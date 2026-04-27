"""
Centerline extraction from a ROS occupancy map (.pgm + .yaml)
Output: CSV compatible with TUMFTM global_racetrajectory_optimization

Approach:
  1. Load PGM + YAML, binarize into free/occupied
  2. Compute EDT on free space → pixel = distance to nearest wall
  3. Skeletonize free space → candidate centerline pixels
  4. Prune skeleton branches using EDT threshold (low EDT = near wall = branch)
  5. Order centerline points via graph traversal
  6. Ray-cast perpendicular to heading for left/right track widths
  7. Convert to world coordinates and write TUM CSV
"""

import numpy as np
import cv2
import yaml
import csv
import argparse
from pathlib import Path
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from skimage.graph import pixel_graph
import networkx as nx


# ---------------------------------------------------------------------------
# 1. Load map
# ---------------------------------------------------------------------------

def load_map(pgm_path: str, yaml_path: str):
    """
    Returns:
        binary  : 2D bool array, True = free space
        resolution : meters per pixel
        origin     : (x, y) world coords of the bottom-left pixel
    """
    img = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read PGM: {pgm_path}")

    with open(yaml_path) as f:
        meta = yaml.safe_load(f)

    resolution  = float(meta["resolution"])           # m/px
    origin      = meta["origin"][:2]                  # [x, y]
    occ_thresh  = float(meta.get("occupied_thresh", 0.65))
    free_thresh = float(meta.get("free_thresh",     0.196))

    # ROS convention: 0 = black = occupied, 254 = white = free, 205 = unknown
    norm = img.astype(np.float32) / 255.0
    free     = norm > (1.0 - free_thresh)     # definitely free
    occupied = norm < (1.0 - occ_thresh)      # definitely occupied

    binary = free & ~occupied  # True where navigable
    return binary, resolution, origin


# ---------------------------------------------------------------------------
# 2. Isolate track corridor via flood-fill from a seed point
# ---------------------------------------------------------------------------

def isolate_track_corridor(binary: np.ndarray, seed_rc: tuple):
    """
    Flood-fill from seed_rc (row, col) on the free-space binary mask.
    Returns a new binary mask containing only the connected region that
    includes the seed — i.e. the track corridor, not the exterior.

    seed_rc: (row, col) pixel coordinate known to be inside the track.
    """
    r, c = seed_rc
    H, W = binary.shape

    if not (0 <= r < H and 0 <= c < W):
        raise ValueError(f"Seed pixel ({r}, {c}) is outside the map bounds ({H}x{W}).")
    if not binary[r, c]:
        raise ValueError(
            f"Seed pixel ({r}, {c}) is not free space. "
            "Pick a point clearly inside the track corridor."
        )

    # cv2.floodFill needs a uint8 image and a mask 2px larger on each side
    canvas = binary.astype(np.uint8) * 255
    flood_mask = np.zeros((H + 2, W + 2), dtype=np.uint8)
    cv2.floodFill(canvas, flood_mask, seedPoint=(c, r), newVal=128)

    # Pixels set to 128 are the flood-filled (track) region
    track_only = (canvas == 128)
    print(f"  Track corridor pixels after flood-fill: {track_only.sum()}")
    return track_only


# ---------------------------------------------------------------------------
# 3. EDT + Skeletonize
# ---------------------------------------------------------------------------

def compute_skeleton_with_edt(binary: np.ndarray):
    """
    Returns:
        skeleton : bool array, centerline candidate pixels
        edt      : float array, distance to nearest wall (pixels)
    """
    edt = distance_transform_edt(binary)
    skeleton = skeletonize(binary)
    return skeleton, edt


# ---------------------------------------------------------------------------
# 3. Prune branches via EDT threshold
# ---------------------------------------------------------------------------

def prune_skeleton(skeleton: np.ndarray, edt: np.ndarray, edt_percentile: float = 25.0):
    """
    Remove skeleton pixels whose EDT value is below a threshold.
    Branches close to walls have low EDT; the true centerline has high EDT.

    edt_percentile: prune pixels below this percentile of EDT values on skeleton.
                    Lower = keep more, Higher = more aggressive pruning.
                    25 is a good default; increase if branches remain.
    """
    skel_edt = edt[skeleton]
    if skel_edt.size == 0:
        raise ValueError("Empty skeleton — check your binary map thresholding.")

    threshold = np.percentile(skel_edt, edt_percentile)
    pruned = skeleton & (edt >= threshold)
    return pruned


# ---------------------------------------------------------------------------
# 4. Order centerline points
# ---------------------------------------------------------------------------

def order_centerline(skeleton: np.ndarray, edt: np.ndarray):
    """
    Build a graph from skeleton adjacency and find the correct traversal order.

    Edges are weighted by 1/EDT so paths near walls are expensive.
    This means weighted-shortest-path prefers routes that stay away from
    walls — fixing the U-turn problem where an unweighted traversal picks
    the short branch toward the outer corner instead of going around.
    """
    rows, cols = np.where(skeleton)
    if len(rows) == 0:
        raise ValueError("Empty skeleton after pruning.")

    pixel_to_idx = {(r, c): i for i, (r, c) in enumerate(zip(rows, cols))}
    idx_to_pixel = list(zip(rows, cols))

    G = nx.Graph()
    G.add_nodes_from(range(len(rows)))

    for (r, c), i in pixel_to_idx.items():
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nb = (r + dr, c + dc)
                if nb in pixel_to_idx:
                    j = pixel_to_idx[nb]
                    if not G.has_edge(i, j):
                        # Weight = 1/EDT at the neighbour pixel.
                        # High EDT (far from wall) → low cost → preferred.
                        edt_val = max(edt[nb[0], nb[1]], 0.1)  # avoid div/0
                        dist = np.hypot(dr, dc)                 # 1 or sqrt(2)
                        G.add_edge(i, j, weight=dist / edt_val)

    # Keep largest connected component
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()

    endpoints = [n for n, d in G.degree() if d == 1]

    if len(endpoints) >= 2:
        # Find the two endpoints that are farthest apart by weighted distance
        # (use a sample of endpoints if there are many, for speed)
        sample = endpoints[:min(len(endpoints), 10)]
        best_len, best_path = 0, None
        for start in sample:
            lengths = nx.single_source_dijkstra_path_length(G, start, weight="weight")
            end = max((n for n in endpoints if n != start),
                      key=lambda n: lengths.get(n, 0))
            path_len = lengths.get(end, 0)
            if path_len > best_len:
                best_len = path_len
                best_path = nx.dijkstra_path(G, start, end, weight="weight")
        ordered_indices = best_path
    else:
        # Already a clean loop — DFS from any node
        ordered_indices = list(nx.dfs_preorder_nodes(G, source=next(iter(G.nodes()))))

    return [idx_to_pixel[i] for i in ordered_indices]


def close_loop(ordered_pixels, max_gap_px=50):
    """
    Check if the first and last skeleton points are close enough to form a
    closed loop. If the gap is <= max_gap_px, interpolate bridging pixels
    along the straight line between them and append to close the loop.

    max_gap_px: maximum pixel distance to attempt bridging (default 50).
                Increase if your track has a larger break at the closure point.
    """
    first = np.array(ordered_pixels[0],  dtype=float)
    last  = np.array(ordered_pixels[-1], dtype=float)
    gap   = np.linalg.norm(last - first)

    if gap > max_gap_px:
        print(f"  Loop gap is {gap:.1f} px — too large to bridge automatically. "
              f"Try lowering --edt-percentile or cleaning the map more.")
        return ordered_pixels

    # Interpolate bridging pixels along the straight line
    n_bridge = int(gap)
    bridge = []
    for t in np.linspace(0, 1, n_bridge, endpoint=False)[1:]:
        pt = last + t * (first - last)
        bridge.append((int(round(pt[0])), int(round(pt[1]))))

    print(f"  Closed loop: bridged {gap:.1f} px gap with {len(bridge)} interpolated points")
    return ordered_pixels + bridge


# ---------------------------------------------------------------------------
# 4b. Auto EDT percentile search
# ---------------------------------------------------------------------------

def score_centerline(ordered_pixels, max_gap_px=50):
    """
    Score a candidate centerline. Higher = better.

    Three signals:
      1. Closing gap   — distance between first and last point (want small)
      2. Max heading change per step — a wrong branch causes a near-180 reversal
      3. Coverage      — number of points (want large, proxy for track coverage)

    Returns (score, gap_px, max_turn_deg) for logging.
    """
    pts = np.array(ordered_pixels, dtype=float)
    N = len(pts)

    if N < 20:
        return -np.inf, np.inf, np.inf

    # 1. Closing gap
    gap = float(np.linalg.norm(pts[-1] - pts[0]))

    # 2. Max consecutive heading change
    # Compute unit heading vectors via central differences
    prev = pts[np.arange(N) - 1]          # wraps: [-1] = last point
    nxt  = pts[(np.arange(N) + 1) % N]
    diffs = nxt - prev
    norms = np.linalg.norm(diffs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    headings = diffs / norms               # (N, 2) unit vectors

    dots = np.clip((headings[:-1] * headings[1:]).sum(axis=1), -1.0, 1.0)
    angles_deg = np.degrees(np.arccos(dots))
    max_turn = float(angles_deg.max()) if len(angles_deg) > 0 else 0.0

    # Score: heavily penalise wrong-branch reversals (> 90 deg is suspicious,
    # > 120 deg almost certainly means a U-turn was mis-traced)
    turn_penalty = max_turn ** 2           # quadratic: 90->8100, 150->22500
    gap_penalty  = gap * 10               # scale to be comparable
    coverage     = N

    score = coverage - turn_penalty - gap_penalty
    return score, gap, max_turn


def auto_find_edt_percentile(skeleton, edt,
                              percentiles=None,
                              max_gap_px=50,
                              verbose=True):
    """
    Try a range of EDT percentiles, score each result, return the best one.

    percentiles: list of values to try. Default: 5 to 30 in steps of 2.5
    Returns: (best_percentile, best_ordered_pixels)
    """
    if percentiles is None:
        percentiles = list(np.arange(5.0, 32.5, 2.5))

    best_score    = -np.inf
    best_pct      = None
    best_pixels   = None

    if verbose:
        print(f"  Searching over {len(percentiles)} EDT percentile values...")
        print(f"  {'PCT':>6}  {'pts':>6}  {'gap_px':>8}  {'max_turn°':>10}  {'score':>10}")

    for pct in percentiles:
        try:
            pruned  = prune_skeleton(skeleton, edt, pct)
            if pruned.sum() < 20:
                continue
            pixels  = order_centerline(pruned, edt)
            pixels  = close_loop(pixels, max_gap_px=max_gap_px)
            score, gap, max_turn = score_centerline(pixels, max_gap_px)

            if verbose:
                print(f"  {pct:>6.1f}  {len(pixels):>6}  {gap:>8.1f}  "
                      f"{max_turn:>10.1f}  {score:>10.1f}")

            if score > best_score:
                best_score  = score
                best_pct    = pct
                best_pixels = pixels

        except Exception as e:
            if verbose:
                print(f"  {pct:>6.1f}  failed: {e}")
            continue

    if best_pixels is None:
        raise RuntimeError("No valid centerline found at any EDT percentile. "
                           "Check that the map is clean and the track corridor is isolated.")

    print(f"Best EDT percentile: {best_pct:.1f}  "
          f"(score={best_score:.1f}, pts={len(best_pixels)})")
    return best_pct, best_pixels


# ---------------------------------------------------------------------------
# 5. Compute left/right widths via perpendicular ray casting
# ---------------------------------------------------------------------------

def raycast_widths(ordered_pixels, edt: np.ndarray, binary: np.ndarray):
    """
    At each centerline point, cast rays perpendicular to the heading direction
    to measure left and right track widths in pixels.
    Returns list of (w_right_px, w_left_px).
    """
    pts = np.array(ordered_pixels, dtype=float)  # (N, 2) as (row, col)
    N = len(pts)
    widths = []

    for i in range(N):
        # Heading vector: use finite difference
        prev_i = max(0, i - 1)
        next_i = min(N - 1, i + 1)
        dr = pts[next_i, 0] - pts[prev_i, 0]
        dc = pts[next_i, 1] - pts[prev_i, 1]
        norm = np.hypot(dr, dc)
        if norm < 1e-6:
            widths.append((edt[int(pts[i, 0]), int(pts[i, 1])],
                           edt[int(pts[i, 0]), int(pts[i, 1])]))
            continue

        # Perpendicular direction (right = +perp, left = -perp)
        # If heading is (dr, dc), right perp is (dc, -dr) in (row, col) space
        perp_r =  dc / norm
        perp_c = -dr / norm

        r0, c0 = pts[i]
        H, W = binary.shape

        def cast(sign):
            for step in range(1, int(edt[int(r0), int(c0)]) + 5):
                r = int(round(r0 + sign * perp_r * step))
                c = int(round(c0 + sign * perp_c * step))
                if not (0 <= r < H and 0 <= c < W):
                    return step - 1
                if not binary[r, c]:
                    return step - 1
            return edt[int(r0), int(c0)]

        w_right = cast(+1)
        w_left  = cast(-1)
        widths.append((w_right, w_left))

    return widths


# ---------------------------------------------------------------------------
# 6. Downsample for smoother output
# ---------------------------------------------------------------------------

def downsample(ordered_pixels, widths, step: int = 5):
    """Keep every `step`-th point to reduce noise and output size."""
    pts = ordered_pixels[::step]
    wts = widths[::step]
    return pts, wts


# ---------------------------------------------------------------------------
# 7. Pixel → world coordinates
# ---------------------------------------------------------------------------

def pixels_to_world_with_shape(ordered_pixels, binary_shape, resolution: float, origin):
    H = binary_shape[0]
    world_pts = []
    for (r, c) in ordered_pixels:
        wx = origin[0] + c * resolution
        wy = origin[1] + (H - r) * resolution
        world_pts.append((wx, wy))
    return world_pts


# ---------------------------------------------------------------------------
# 8. Write TUM CSV
# ---------------------------------------------------------------------------

def write_tum_csv(output_path: str, world_pts, widths_px, resolution: float):
    """
    TUM format:  # x_m,y_m,w_tr_right_m,w_tr_left_m
    """
    with open(output_path, "w", newline="") as f:
        f.write("# x_m,y_m,w_tr_right_m,w_tr_left_m\n")
        writer = csv.writer(f)
        for (wx, wy), (wr, wl) in zip(world_pts, widths_px):
            writer.writerow([
                round(wx, 4),
                round(wy, 4),
                round(wr * resolution, 4),
                round(wl * resolution, 4),
            ])
    print(f"Wrote {len(world_pts)} centerline points to {output_path}")


# ---------------------------------------------------------------------------
# Helpers: discover map files inside a directory
# ---------------------------------------------------------------------------

def find_map_files(map_dir: Path):
    """
    Find the .pgm and .yaml to use in a map directory.
    Prefers *_cleaned.pgm / *_cleaned.yaml (output of clean_map.py),
    falling back to the original if no cleaned file exists.
    """
    def pick(suffix):
        cleaned   = sorted(map_dir.glob(f"*_cleaned{suffix}"))
        originals = [p for p in sorted(map_dir.glob(f"*{suffix}"))
                     if "_cleaned" not in p.stem]
        if cleaned:
            if len(cleaned) > 1:
                raise FileNotFoundError(
                    f"Multiple cleaned {suffix} files in {map_dir}: "
                    f"{[p.name for p in cleaned]}"
                )
            return cleaned[0], True
        if not originals:
            raise FileNotFoundError(f"No {suffix} file found in {map_dir}")
        if len(originals) > 1:
            raise FileNotFoundError(
                f"Multiple {suffix} files in {map_dir}: {[p.name for p in originals]}. "
                f"Run clean_map.py first, or remove duplicates."
            )
        return originals[0], False

    pgm_path,  pgm_cleaned  = pick(".pgm")
    yaml_path, _            = pick(".yaml")
    tag = " (cleaned)" if pgm_cleaned else " (original — run clean_map.py for best results)"
    print(f"  Using map: {pgm_path.name}{tag}")
    return pgm_path, yaml_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract centerline from a ROS map → TUM CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Input modes:
  Directory mode (standalone use):
    python extract_centerline.py maps/levine
    python extract_centerline.py maps/levine --visualize

  Base-path mode (pipeline use, matches generate_centerline.py interface):
    python extract_centerline.py --map /tmp/robo_arp_map --seed 274 512 --no-plot

Output:
  Directory mode  → {map_dir}/centerline.csv
  Base-path mode  → {base}_centerline.csv  (e.g. /tmp/robo_arp_map_centerline.csv)
        """,
    )
    parser.add_argument("map_dir", nargs="?",
                        help="Path to the map directory (contains .pgm and .yaml)")
    parser.add_argument("--map",
                        help="Base path to map files without extension "
                             "(e.g. /tmp/robo_arp_map). Expects <base>.pgm and "
                             "<base>.yaml. Output goes to <base>_centerline.csv.")
    parser.add_argument("--seed", type=int, nargs=2, metavar=("ROW", "COL"),
                        help="Pixel coordinate (row col) of a point inside the track corridor.")
    parser.add_argument("--edt-percentile", type=float, default=None,
                        help="Skeleton pruning percentile (0–100). "
                             "If omitted, auto-searches for the best value.")
    parser.add_argument("--no-auto-edt", action="store_true",
                        help="Disable auto-search and use --edt-percentile directly "
                             "(default percentile becomes 25 if not specified).")
    parser.add_argument("--downsample",  type=int, default=5,
                        help="Keep every N-th point (default: 5)")
    parser.add_argument("--visualize",   action="store_true",
                        help="Save a debug image showing the centerline overlay")
    parser.add_argument("--no-plot", action="store_true",
                        help="Suppress visualisation (equivalent to omitting --visualize)")
    args = parser.parse_args()

    # Resolve input paths and output location
    if args.map:
        base = Path(args.map).resolve()
        pgm_path  = Path(str(base) + ".pgm")
        yaml_path = Path(str(base) + ".yaml")
        if not pgm_path.exists():
            raise FileNotFoundError(f"PGM not found: {pgm_path}")
        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML not found: {yaml_path}")
        output_path = Path(str(base) + "_centerline.csv")
        map_dir = base.parent
    elif args.map_dir:
        map_dir = Path(args.map_dir).resolve()
        if not map_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {map_dir}")
        pgm_path, yaml_path = find_map_files(map_dir)
        output_path = map_dir / "centerline.csv"
    else:
        parser.error("Provide either a map_dir positional argument or --map <base_path>")

    if args.no_plot:
        args.visualize = False

    print(f"Map directory : {map_dir}")
    print(f"PGM           : {pgm_path.name}")
    print(f"YAML          : {yaml_path.name}")
    print(f"Output        : {output_path}")
    print()

    print("Loading map...")
    binary, resolution, origin = load_map(str(pgm_path), str(yaml_path))
    print(f"  Map size: {binary.shape}, resolution: {resolution} m/px")
    print(f"  Free pixels: {binary.sum()}")

    if args.seed:
        print(f"Isolating track corridor (seed: row={args.seed[0]}, col={args.seed[1]})...")
        binary = isolate_track_corridor(binary, tuple(args.seed))
    else:
        print("No --seed provided — using all free space. If the centerline looks wrong,")
        print("run with --visualize to find a pixel inside the track, then rerun with --seed ROW COL.")
        print()

    print("Computing EDT + skeleton...")
    skeleton, edt = compute_skeleton_with_edt(binary)
    print(f"  Skeleton pixels before pruning: {skeleton.sum()}")

    use_auto = not args.no_auto_edt and args.edt_percentile is None
    if use_auto:
        print("Auto-searching for best EDT percentile...")
        best_pct, ordered_pixels = auto_find_edt_percentile(skeleton, edt)
        print(f"  Ordered points: {len(ordered_pixels)}")
    else:
        pct = args.edt_percentile if args.edt_percentile is not None else 25.0
        print(f"Pruning branches (EDT percentile: {pct})...")
        pruned = prune_skeleton(skeleton, edt, pct)
        print(f"  Skeleton pixels after pruning: {pruned.sum()}")
        print("Ordering centerline points...")
        ordered_pixels = order_centerline(pruned, edt)
        ordered_pixels = close_loop(ordered_pixels)
        print(f"  Ordered points: {len(ordered_pixels)}")

    print("Ray-casting track widths...")
    widths = raycast_widths(ordered_pixels, edt, binary)

    print(f"Downsampling (every {args.downsample} points)...")
    ordered_pixels, widths = downsample(ordered_pixels, widths, args.downsample)
    print(f"  Final point count: {len(ordered_pixels)}")

    world_pts = pixels_to_world_with_shape(ordered_pixels, binary.shape, resolution, origin)

    write_tum_csv(str(output_path), world_pts, widths, resolution)

    if args.visualize:
        vis_path = map_dir / "centerline_debug.png"
        img_vis = cv2.cvtColor((binary * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for (r, c) in ordered_pixels:
            cv2.circle(img_vis, (c, r), 1, (0, 0, 255), -1)
        if args.seed:
            cv2.drawMarker(img_vis, (args.seed[1], args.seed[0]),
                           color=(0, 255, 0), markerType=cv2.MARKER_CROSS,
                           markerSize=15, thickness=2)
        cv2.imwrite(str(vis_path), img_vis)
        print(f"  Debug image saved to {vis_path}")
        print("  Tip: open the image in any viewer that shows pixel coordinates")
        print("       to find a ROW COL inside the track for --seed.")


if __name__ == "__main__":
    main()
