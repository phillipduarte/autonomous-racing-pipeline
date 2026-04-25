#!/usr/bin/env python3
"""
Centerline Generation & Verification Tool
============================================
Generates centerline waypoints for F1TENTH maps using corridor-skeleton
analysis with branch pruning.

ALWAYS run this and check the plot BEFORE training on a custom map.

Algorithm:
    1. Distance transform to find corridor regions
    2. Skeletonize corridors to get center paths
    3. Prune dead-end branches to keep only loops/main paths
    4. Trace and order the path
    5. Smooth and subsample

Usage:
    # Generate and verify centerline
    python scripts/generate_centerline.py --map maps/levine_blocked/levine_blocked

    # Adjust for different map types
    python scripts/generate_centerline.py --map maps/my_map/my_map --wall-threshold 200

    # Load existing waypoints (to verify or add speed profile)
    python scripts/generate_centerline.py --map maps/my_map/my_map --from-csv waypoints.csv

    # With speed profile
    python scripts/generate_centerline.py --map maps/my_map/my_map --speed-profile --max-speed 6.0
"""

import argparse
import os
import sys
import shutil
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_map(map_path, map_ext=".png"):
    """Load occupancy grid map and metadata."""
    import yaml
    from PIL import Image

    for candidate in [Path(f"{map_path}_map.yaml"), Path(f"{map_path}.yaml")]:
        if candidate.exists():
            yaml_file = candidate
            break
    else:
        raise FileNotFoundError(f"Map YAML not found for {map_path}")

    with open(yaml_file) as f:
        meta = yaml.safe_load(f)

    resolution = meta.get("resolution", 0.05)
    origin = meta.get("origin", [0.0, 0.0, 0.0])
    image_file = meta.get("image", Path(map_path).stem + map_ext)
    img = np.array(Image.open(yaml_file.parent / image_file).convert("L"))

    print(f"Map: {yaml_file.parent / image_file}")
    print(f"  {img.shape[1]}x{img.shape[0]} px, {resolution} m/px, "
          f"world: {img.shape[1]*resolution:.0f}x{img.shape[0]*resolution:.0f}m")
    return img, resolution, origin, yaml_file


def extract_centerline(img, resolution, origin, wall_threshold=128,
                       min_corridor_width=0.3, max_corridor_width=8.0,
                       smooth_window=11, spacing=0.1):
    """
    Extract centerline using corridor-skeleton + branch-pruning.

    Steps:
        1. Distance transform from walls
        2. Create corridor mask (pixels at correct distance from walls)
        3. Skeletonize corridor mask → corridor center paths
        4. Prune dead-end branches → keep only loops/main paths
        5. Find largest connected component
        6. Trace, smooth, subsample
    """
    from scipy.ndimage import distance_transform_edt
    from skimage.morphology import skeletonize

    walls = img < wall_threshold
    free = ~walls

    # Auto-detect threshold if default doesn't find enough corridor structure
    n_free = free.sum()
    n_total = img.size
    free_ratio = n_free / n_total
    if free_ratio > 0.5:
        # Most pixels are "free" — threshold is too low.
        # Find the dominant background value and set threshold just above it.
        unique_vals, counts = np.unique(img, return_counts=True)
        # The dominant value (mode) is the background
        mode_idx = np.argmax(counts)
        mode_val = int(unique_vals[mode_idx])
        mode_count = counts[mode_idx]

        # If the mode is bright (background gray), threshold should be above it
        if mode_val > wall_threshold and mode_count > n_total * 0.5:
            # Set threshold between the mode and the next distinct group
            auto_threshold = mode_val + 1
            track_pixels = (img > auto_threshold).sum()

            # Only use auto-threshold if it finds a reasonable amount of track
            if 100 < track_pixels < n_total * 0.2:
                print(f"  Auto-threshold: {auto_threshold} "
                      f"(background={mode_val}, track={track_pixels} px = {track_pixels/n_total:.1%})")
                wall_threshold = auto_threshold
                walls = img < wall_threshold
                free = ~walls

    n_walls = walls.sum()
    print(f"\n  Wall threshold: {wall_threshold}")
    print(f"  Wall pixels: {n_walls} ({n_walls/n_total:.1%})")

    dist = distance_transform_edt(free) * resolution

    # Step 1: Corridor mask — free pixels within corridor-width distance from walls
    corridor = free & (dist > min_corridor_width * 0.5) & (dist < max_corridor_width)
    print(f"  Corridor pixels: {corridor.sum()}")

    # Step 2: Skeletonize the corridor mask
    skel = skeletonize(corridor)
    skel_pts = set(map(tuple, np.argwhere(skel)))
    print(f"  Skeleton pixels: {len(skel_pts)}")

    if len(skel_pts) < 20:
        print("  ERROR: Too few skeleton points. Adjust thresholds.")
        return np.zeros((0, 2))

    # Step 3: Build pixel adjacency (8-connected)
    def get_neighbors(p):
        r, c = p
        return [(r+dr, c+dc) for dr in [-1,0,1] for dc in [-1,0,1]
                if (dr or dc) and (r+dr, c+dc) in skel_pts]

    # Step 4: Prune dead-end branches iteratively
    # This removes degree-1 pixels repeatedly, keeping only loop/backbone structure
    pruned = set(skel_pts)
    iterations = 0
    while True:
        to_remove = set()
        for p in pruned:
            nbs = [n for n in get_neighbors_from(p, pruned) if n in pruned]
            if len(nbs) <= 1:
                to_remove.add(p)
        if not to_remove:
            break
        pruned -= to_remove
        iterations += 1
        if iterations > 1000:
            break

    print(f"  After pruning ({iterations} iters): {len(pruned)} points")

    if len(pruned) < 10:
        print("  Pruning removed everything — using unpruned skeleton instead.")
        pruned = skel_pts

    # Step 5: Find connected components
    visited = set()
    components = []
    for p in pruned:
        if p in visited:
            continue
        comp = []
        stack = [p]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            comp.append(cur)
            for nb in get_neighbors_from(cur, pruned):
                if nb not in visited:
                    stack.append(nb)
        components.append(comp)

    sizes = sorted([len(c) for c in components], reverse=True)
    print(f"  Components: {len(components)}, sizes: {sizes[:5]}")

    # Take the largest component
    main = max(components, key=len)
    main_set = set(main)

    # Step 6: Trace the path by following connectivity
    ordered = trace_path(main, main_set)
    print(f"  Traced path: {len(ordered)} points")

    # Convert to world coordinates
    pts = np.array(ordered)
    wx = pts[:, 1] * resolution + origin[0]
    wy = (img.shape[0] - pts[:, 0]) * resolution + origin[1]
    world = np.column_stack([wx, wy])

    # Smooth
    if smooth_window > 1 and len(world) > smooth_window * 3:
        world = smooth_closed(world, smooth_window)

    # Subsample
    world = subsample(world, spacing)

    length = track_length(world)
    print(f"  Final: {len(world)} waypoints, track length: {length:.1f}m")

    return world


def get_neighbors_from(p, point_set):
    """Get 8-connected neighbors that are in point_set."""
    r, c = p
    return [(r+dr, c+dc) for dr in [-1,0,1] for dc in [-1,0,1]
            if (dr or dc) and (r+dr, c+dc) in point_set]


def trace_path(points, point_set):
    """Trace a path through connected pixels, preferring straight-line continuation."""
    # Start from a point with exactly 2 neighbors (regular path point)
    start = points[0]
    for p in points:
        nbs = get_neighbors_from(p, point_set)
        if len(nbs) == 2:
            start = p
            break

    ordered = [start]
    visited = {start}

    current = start
    prev_dir = None

    for _ in range(len(points) + 100):
        candidates = [n for n in get_neighbors_from(current, point_set) if n not in visited]
        if not candidates:
            break

        if prev_dir is not None and len(candidates) > 1:
            # Prefer the direction most aligned with previous direction
            best = candidates[0]
            best_score = -999
            for c in candidates:
                dr = c[0] - current[0]
                dc = c[1] - current[1]
                # Dot product with previous direction
                score = dr * prev_dir[0] + dc * prev_dir[1]
                if score > best_score:
                    best_score = score
                    best = c
            next_pt = best
        else:
            next_pt = candidates[0]

        prev_dir = (next_pt[0] - current[0], next_pt[1] - current[1])
        ordered.append(next_pt)
        visited.add(next_pt)
        current = next_pt

    return ordered


def smooth_closed(points, window=11):
    """Smooth waypoints (handles both open and closed paths)."""
    n = len(points)
    pad = window
    padded = np.concatenate([points[-pad:], points, points[:pad]])
    try:
        from scipy.signal import savgol_filter
        w = min(window, len(padded) - 1) | 1  # must be odd
        sx = savgol_filter(padded[:, 0], w, min(3, w - 1))
        sy = savgol_filter(padded[:, 1], w, min(3, w - 1))
    except Exception:
        kernel = np.ones(window) / window
        sx = np.convolve(padded[:, 0], kernel, mode="same")
        sy = np.convolve(padded[:, 1], kernel, mode="same")
    return np.column_stack([sx[pad:pad+n], sy[pad:pad+n]])


def subsample(points, spacing=0.1):
    """Subsample to approximately uniform spacing."""
    if len(points) < 2:
        return points
    result = [points[0]]
    acc = 0
    for i in range(1, len(points)):
        d = np.sqrt((points[i, 0] - points[i-1, 0])**2 + (points[i, 1] - points[i-1, 1])**2)
        acc += d
        if acc >= spacing:
            result.append(points[i])
            acc = 0
    return np.array(result)


def track_length(points):
    """Total path length."""
    if len(points) < 2:
        return 0
    diffs = np.diff(points, axis=0)
    return np.sqrt((diffs**2).sum(axis=1)).sum()


def compute_speed_profile(waypoints, max_speed=6.0, max_accel=5.0,
                          max_decel=8.0, max_lat_accel=5.0):
    """Forward-backward speed profile from curvature."""
    n = len(waypoints)
    if n < 3:
        return np.column_stack([waypoints, np.full(n, max_speed * 0.5)])

    # Curvature (Menger)
    kappa = np.zeros(n)
    for i in range(n):
        im1, ip1 = (i-1) % n, (i+1) % n
        x1, y1 = waypoints[im1]; x2, y2 = waypoints[i]; x3, y3 = waypoints[ip1]
        dx1, dy1, dx2, dy2 = x2-x1, y2-y1, x3-x2, y3-y2
        cross = abs(dx1*dy2 - dy1*dx2)
        d1, d2 = np.sqrt(dx1**2+dy1**2), np.sqrt(dx2**2+dy2**2)
        d3 = np.sqrt((x3-x1)**2+(y3-y1)**2)
        kappa[i] = 2*cross/(d1*d2*d3) if d1*d2*d3 > 1e-10 else 0

    v_max = np.array([min(max_speed, np.sqrt(max_lat_accel / max(abs(k), 1e-6)))
                       for k in kappa])

    diffs = np.diff(waypoints, axis=0)
    ds = np.sqrt((diffs**2).sum(axis=1))
    ds = np.append(ds, ds[-1])

    v_fwd = np.full(n, max_speed)
    v_fwd[0] = v_max[0]
    for i in range(1, n):
        v_fwd[i] = min(v_max[i], np.sqrt(v_fwd[i-1]**2 + 2*max_accel*ds[i-1]))

    v_bwd = np.full(n, max_speed)
    v_bwd[-1] = v_max[-1]
    for i in range(n-2, -1, -1):
        v_bwd[i] = min(v_max[i], np.sqrt(v_bwd[i+1]**2 + 2*max_decel*ds[i]))

    velocity = np.clip(np.minimum(np.minimum(v_fwd, v_bwd), v_max), 0.5, max_speed)
    return np.column_stack([waypoints, velocity])


def visualize(img, resolution, origin, centerline, output_path, speed_data=None):
    """Generate verification plot."""
    import matplotlib.pyplot as plt

    ncols = 2 if speed_data is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 7))
    if ncols == 1:
        axes = [axes]

    ax = axes[0]
    ax.imshow(img, cmap="gray", origin="upper")
    ax.set_title("Map with Centerline", fontsize=14)

    if len(centerline) > 0:
        px = (centerline[:, 0] - origin[0]) / resolution
        py = img.shape[0] - (centerline[:, 1] - origin[1]) / resolution

        ax.plot(px, py, "r-", linewidth=2, alpha=0.8, label="Centerline")
        ax.plot(px[0], py[0], "go", markersize=12, label="Start", zorder=5)
        ax.plot(px[-1], py[-1], "bs", markersize=10, label="End", zorder=5)

        step = max(1, len(px) // 20)
        for i in range(0, len(px) - 3, step):
            dx, dy = px[min(i+3, len(px)-1)] - px[i], py[min(i+3, len(py)-1)] - py[i]
            ax.annotate("", xy=(px[i] + dx*0.5, py[i] + dy*0.5), xytext=(px[i], py[i]),
                         arrowprops=dict(arrowstyle="->", color="yellow", lw=2))

        margin = 50
        ax.set_xlim(px.min() - margin, px.max() + margin)
        ax.set_ylim(py.max() + margin, py.min() - margin)
        ax.legend(fontsize=10)

    if speed_data is not None and len(centerline) > 0:
        ax2 = axes[1]
        diffs = np.diff(centerline, axis=0)
        s = np.concatenate([[0], np.cumsum(np.sqrt((diffs**2).sum(axis=1)))])
        ax2.plot(s, speed_data[:, 2], "b-", lw=1.5)
        ax2.set_xlabel("Distance (m)")
        ax2.set_ylabel("Speed (m/s)")
        ax2.set_title("Velocity Profile")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved: {output_path}")
    print(f"  >>> CHECK THIS IMAGE TO VERIFY CENTERLINE <<<")


def main():
    parser = argparse.ArgumentParser(description="Generate centerline for F1TENTH maps")
    parser.add_argument("--map", required=True, help="Map path without extension")
    parser.add_argument("--ext", default=".png")
    parser.add_argument("--from-csv", default=None, help="Load from existing CSV")
    parser.add_argument("--wall-threshold", type=int, default=128)
    parser.add_argument("--min-corridor-width", type=float, default=0.3)
    parser.add_argument("--max-corridor-width", type=float, default=8.0)
    parser.add_argument("--smooth-window", type=int, default=11)
    parser.add_argument("--spacing", type=float, default=0.1)
    parser.add_argument("--speed-profile", action="store_true")
    parser.add_argument("--max-speed", type=float, default=6.0)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    map_path = str(Path(project_root) / args.map)
    img, resolution, origin, yaml_file = load_map(map_path, args.ext)

    if args.from_csv:
        data = np.loadtxt(args.from_csv, delimiter=",", skiprows=1)
        centerline = data[:, :2]
        print(f"Loaded {len(centerline)} waypoints from {args.from_csv}")
    else:
        centerline = extract_centerline(
            img, resolution, origin,
            wall_threshold=args.wall_threshold,
            min_corridor_width=args.min_corridor_width,
            max_corridor_width=args.max_corridor_width,
            smooth_window=args.smooth_window,
            spacing=args.spacing,
        )

    if len(centerline) == 0:
        print("\nFAILED. Adjust thresholds or provide --from-csv")
        return

    stem = Path(map_path).stem
    map_dir = Path(map_path).parent
    out_path = str(map_dir / f"{stem}_centerline.csv")

    speed_data = None
    if args.speed_profile:
        speed_data = compute_speed_profile(centerline, max_speed=args.max_speed)
        np.savetxt(out_path, speed_data, delimiter=",", header="x_m,y_m,vx_mps", comments="")
    else:
        np.savetxt(out_path, centerline, delimiter=",", header="x_m,y_m", comments="")
    print(f"  Saved: {out_path}")

    # Ensure _map.yaml exists
    map_yaml = map_dir / f"{stem}_map.yaml"
    if not map_yaml.exists():
        src = map_dir / f"{stem}.yaml"
        if src.exists():
            shutil.copy2(str(src), str(map_yaml))

    if not args.no_plot:
        plot_path = str(map_dir / f"{stem}_centerline_check.png")
        visualize(img, resolution, origin, centerline, plot_path, speed_data)

    print(f"\nNext steps:")
    print(f"  1. Verify: {map_dir / f'{stem}_centerline_check.png'}")
    print(f"  2. Train:  python scripts/train.py --config configs/levine.yaml")


if __name__ == "__main__":
    main()
