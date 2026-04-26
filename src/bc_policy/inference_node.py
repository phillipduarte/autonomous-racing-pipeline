#!/usr/bin/env python3
"""
F1TENTH ROS2 RL Inference Node
================================
Deploys trained RL policies on physical F1TENTH cars.

Supports TWO modes (set via config or --use_localization param):

  END-TO-END (obs_type: lidar_state):
    /scan + /odom → policy → /drive
    No particle filter needed. Fast. Robust.

  LOCALIZATION-BASED (obs_type: lidar_waypoint):
    /scan + /odom + /pf/viz/inferred_pose → policy → /drive
    Follows specific waypoints. Needs particle filter + waypoint CSV.

Hardware: Hokuyo UST-10LX (1080 beams) + Jetson Orin Nano + VESC

Usage:
    # End-to-end (no localization)
    ros2 run f1tenth_rl inference_node --ros-args \\
        -p model_path:=runs/sim2real_e2e_*/final_model.onnx \\
        -p config_path:=configs/sim2real_e2e.yaml \\
        -p use_onnx:=true -p max_speed:=2.0

    # Localization-based (with particle filter)
    ros2 run f1tenth_rl inference_node --ros-args \\
        -p model_path:=runs/sim2real_localized_*/final_model.onnx \\
        -p config_path:=configs/sim2real_localized.yaml \\
        -p use_onnx:=true -p max_speed:=2.0 \\
        -p waypoint_path:=maps/levine_blocked/levine_blocked_centerline.csv

    # Test in f1tenth_gym_ros
    ros2 run f1tenth_rl inference_node --ros-args \\
        -p model_path:=final_model.onnx -p use_onnx:=true \\
        -p max_speed:=4.0 -p odom_topic:=/ego_racecar/odom
"""

import os
import time
import numpy as np
import yaml
from pathlib import Path

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import LaserScan
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import PoseStamped
    from ackermann_msgs.msg import AckermannDriveStamped
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("[INFO] ROS2 not available. Install: sudo apt install ros-humble-desktop")


if ROS2_AVAILABLE:

    class RLInferenceNode(Node):

        def __init__(self):
            super().__init__("rl_inference_node")

            # ---- Parameters ----
            self.declare_parameter("model_path", "")
            self.declare_parameter("config_path", "configs/sim2real_e2e.yaml")
            self.declare_parameter("max_speed", 2.0)
            self.declare_parameter("smoothing_alpha", 0.4)
            self.declare_parameter("max_steer_rate", 2.0)      # rad/s
            self.declare_parameter("use_onnx", False)
            self.declare_parameter("inference_rate", 40.0)
            self.declare_parameter("scan_topic", "/scan")
            self.declare_parameter("odom_topic", "/odom")
            self.declare_parameter("pose_topic", "/pf/viz/inferred_pose")
            self.declare_parameter("drive_topic", "/drive")
            self.declare_parameter("waypoint_path", "")        # For localization mode
            self.declare_parameter("watchdog_timeout", 0.5)
            self.declare_parameter("flip_scan", False)         # Reverse scan direction

            model_path = self.get_parameter("model_path").value
            config_path = self.get_parameter("config_path").value
            self.max_speed = self.get_parameter("max_speed").value
            self.smoothing_alpha = self.get_parameter("smoothing_alpha").value
            self.max_steer_rate = self.get_parameter("max_steer_rate").value
            self.use_onnx = self.get_parameter("use_onnx").value
            inference_rate = self.get_parameter("inference_rate").value
            scan_topic = self.get_parameter("scan_topic").value
            odom_topic = self.get_parameter("odom_topic").value
            pose_topic = self.get_parameter("pose_topic").value
            drive_topic = self.get_parameter("drive_topic").value
            waypoint_path = self.get_parameter("waypoint_path").value
            self.watchdog_timeout = self.get_parameter("watchdog_timeout").value
            self.flip_scan = self.get_parameter("flip_scan").value

            # ---- Load config ----
            if os.path.exists(config_path):
                with open(config_path) as f:
                    self.config = yaml.safe_load(f)
            else:
                self.get_logger().warn(f"Config not found: {config_path}")
                self.config = {}

            obs_cfg = self.config.get("observation", {})
            act_cfg = self.config.get("action", {})

            # ---- Determine mode from config ----
            self.obs_type = obs_cfg.get("type", "lidar_state")
            self.use_localization = (self.obs_type == "lidar_waypoint")

            self.num_beams = obs_cfg.get("lidar_beams", 108)
            self.lidar_clip = obs_cfg.get("lidar_clip", 10.0)
            self.lidar_normalize = obs_cfg.get("lidar_normalize", True)
            self.include_velocity = obs_cfg.get("include_velocity", True)
            self.include_yaw_rate = obs_cfg.get("include_yaw_rate", True)
            self.include_prev_action = obs_cfg.get("include_prev_action", True)
            self.num_waypoints = obs_cfg.get("num_waypoints", 5)
            self.max_steer = act_cfg.get("max_steer", 0.4189)
            self.cfg_max_speed = act_cfg.get("max_speed", 4.0)
            self.cfg_min_speed = act_cfg.get("min_speed", 0.5)
            self.eff_max_speed = min(self.max_speed, self.cfg_max_speed)
            self.eff_min_speed = self.cfg_min_speed

            # ---- Load waypoints (for localization mode) ----
            self.waypoints = None
            if self.use_localization:
                self.waypoints = self._load_waypoints(waypoint_path)
                if self.waypoints is None:
                    self.get_logger().error(
                        "Localization mode requires waypoints! "
                        "Set -p waypoint_path:=<path_to_centerline.csv>"
                    )

            # ---- Load model ----
            if not model_path:
                self.get_logger().error("Set -p model_path:=<path>")
                return
            self._load_model(model_path)

            # ---- State ----
            self.prev_action = np.zeros(2, dtype=np.float32)
            self.current_scan = None
            self.current_vel = 0.0
            self.current_yaw_rate = 0.0
            self.current_pose = None  # (x, y, theta)
            self.last_scan_time = self.get_clock().now()
            self.inference_count = 0
            self.total_inference_time = 0.0

            # ---- QoS ----
            sensor_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST, depth=1,
            )

            # ---- Subscribers ----
            self.create_subscription(LaserScan, scan_topic, self._scan_cb, sensor_qos)
            self.create_subscription(Odometry, odom_topic, self._odom_cb, sensor_qos)

            if self.use_localization:
                self.create_subscription(PoseStamped, pose_topic, self._pose_cb, sensor_qos)
                self.create_subscription(Odometry, "/pf/viz/odom", self._odom_cb, sensor_qos)
                self.get_logger().info(f"LOCALIZATION MODE — subscribing to {pose_topic}")

            # ---- Publisher ----
            self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)

            # ---- Timers ----
            self.create_timer(1.0 / inference_rate, self._inference_cb)
            self.create_timer(0.1, self._watchdog_cb)

            mode_str = "LOCALIZATION (lidar+pose+waypoints)" if self.use_localization else "END-TO-END (lidar only)"
            self.get_logger().info(
                f"\n{'='*50}\n"
                f"  RL Inference Node\n"
                f"  Mode:      {mode_str}\n"
                f"  Model:     {model_path}\n"
                f"  ONNX:      {self.use_onnx}\n"
                f"  Max speed: {self.eff_max_speed:.1f} m/s\n"
                f"  Smoothing: {self.smoothing_alpha}\n"
                f"  Scan:      {scan_topic}\n"
                f"  Odom:      {odom_topic}\n"
                f"  Drive:     {drive_topic}\n"
                f"{'='*50}"
            )

        # ---- Waypoint loading ----

        def _load_waypoints(self, path: str):
            """Load waypoints from CSV for localization-based tracking."""
            if not path or not os.path.exists(path):
                # Try to find from map config
                map_path = self.config.get("env", {}).get("map_path", "")
                for suffix in ["_centerline.csv", "_raceline.csv"]:
                    candidate = map_path + suffix
                    if os.path.exists(candidate):
                        path = candidate
                        break
            if not path or not os.path.exists(path):
                return None

            try:
                data = np.loadtxt(path, delimiter=",", skiprows=1)
                wp = data[:, :2] if data.shape[1] >= 2 else None
                if wp is not None:
                    self.get_logger().info(f"Loaded {len(wp)} waypoints from {path}")
                return wp
            except Exception as e:
                self.get_logger().error(f"Failed to load waypoints: {e}")
                return None

        # ---- Model loading ----

        def _load_model(self, model_path: str):
            self.ort_session = None
            self.sb3_model = None
            self.obs_rms = None

            onnx_path = model_path if model_path.endswith(".onnx") else model_path + ".onnx"
            sb3_path = model_path if model_path.endswith(".zip") else model_path + ".zip"

            if self.use_onnx or os.path.exists(onnx_path):
                try:
                    import onnxruntime as ort
                    providers = []
                    for p in ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]:
                        if p in ort.get_available_providers():
                            providers.append(p)
                    opts = ort.SessionOptions()
                    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    opts.intra_op_num_threads = 2
                    self.ort_session = ort.InferenceSession(onnx_path, opts, providers=providers)
                    self.use_onnx = True
                    self.get_logger().info(f"ONNX model loaded (providers: {providers[:1]})")
                except Exception as e:
                    self.get_logger().error(f"ONNX load failed: {e}")
            elif os.path.exists(sb3_path):
                try:
                    from stable_baselines3 import PPO, SAC, TD3
                    algo = self.config.get("algorithm", {}).get("type", "ppo")
                    self.sb3_model = {"ppo": PPO, "sac": SAC, "td3": TD3}[algo].load(sb3_path, device="cpu")
                    self.get_logger().info(f"SB3 {algo.upper()} model loaded")
                except Exception as e:
                    self.get_logger().error(f"SB3 load failed: {e}")

            # Load normalization stats
            # Try numpy format first (no SB3 needed), then pickle as fallback
            npz_candidates = [
                str(Path(model_path).parent / "obs_norm_stats.npz"),
                model_path.replace(".onnx", "_norm_stats.npz"),
            ]
            pkl_candidates = [
                model_path.replace(".onnx", "") + "_vecnormalize.pkl",
                str(Path(model_path).parent / "final_vecnormalize.pkl"),
            ]

            loaded = False
            for npz_path in npz_candidates:
                if os.path.exists(npz_path):
                    try:
                        data = np.load(npz_path)
                        class ObsRMS:
                            pass
                        self.obs_rms = ObsRMS()
                        self.obs_rms.mean = data["mean"]
                        self.obs_rms.var = data["var"]
                        self.get_logger().info(f"Loaded normalization stats from {npz_path}")
                        loaded = True
                    except Exception as e:
                        self.get_logger().warn(f"Failed to load {npz_path}: {e}")
                    break

            if not loaded:
                for norm_path in pkl_candidates:
                    if os.path.exists(norm_path):
                        try:
                            import pickle
                            with open(norm_path, "rb") as f:
                                self.obs_rms = pickle.load(f).obs_rms
                            self.get_logger().info(f"Loaded normalization stats from {norm_path}")
                        except Exception as e:
                            self.get_logger().warn(f"Failed to load norm stats: {e}")
                        break

            if self.obs_rms is None:
                self.get_logger().info(
                    "No normalization stats found — using raw observations. "
                    "This is correct for policies trained with norm_obs=False."
                )

        # ---- Sensor callbacks ----

        def _scan_cb(self, msg):
            ranges = np.array(msg.ranges, dtype=np.float32)
            self.current_scan = np.where(np.isfinite(ranges), ranges, self.lidar_clip)
            if self.flip_scan:
                self.current_scan = self.current_scan[::-1].copy()
            self.last_scan_time = self.get_clock().now()

            # Auto-detect raw beam count on first scan
            if not hasattr(self, '_raw_beams_detected'):
                self._raw_beams_detected = True
                raw = len(ranges)
                self.get_logger().info(
                    f"LiDAR detected: {raw} beams "
                    f"(downsampling {raw} → {self.num_beams})"
                )

        def _odom_cb(self, msg):
            self.current_vel = float(msg.twist.twist.linear.x)
            self.current_yaw_rate = float(msg.twist.twist.angular.z)

        def _pose_cb(self, msg):
            q = msg.pose.orientation
            theta = np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                               1.0 - 2.0 * (q.y**2 + q.z**2))
            self.current_pose = (msg.pose.position.x, msg.pose.position.y, theta)

        def _watchdog_cb(self):
            elapsed = (self.get_clock().now() - self.last_scan_time).nanoseconds / 1e9
            if elapsed > self.watchdog_timeout and self.current_scan is not None:
                self.get_logger().warn(f"No scan for {elapsed:.1f}s — STOPPING")
                self._stop()

        # ---- Inference ----

        def _inference_cb(self):
            if self.current_scan is None:
                return
            if self.ort_session is None and self.sb3_model is None:
                return

            t0 = time.perf_counter()

            # Build observation (matches training preprocessing exactly)
            obs = self._build_obs()

            # Normalize
            if self.obs_rms is not None:
                obs = np.clip(
                    (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8),
                    -10.0, 10.0
                ).astype(np.float32)

            # Inference
            if self.ort_session is not None:
                name = self.ort_session.get_inputs()[0].name
                out = self.ort_session.run(None, {name: obs.reshape(1, -1)})
                action = out[0].squeeze()
            else:
                action, _ = self.sb3_model.predict(obs, deterministic=True)

            # Rescale [-1,1] → [steer, speed]
            steer = float(action[0]) * self.max_steer
            speed = (float(action[1]) + 1.0) * 0.5 * (self.eff_max_speed - self.eff_min_speed) + self.eff_min_speed

            # EMA smoothing
            raw = np.array([steer, speed], dtype=np.float32)
            smoothed = self.smoothing_alpha * raw + (1 - self.smoothing_alpha) * self.prev_action

            # Steering rate limit
            dt = 1.0 / 40.0
            max_delta = self.max_steer_rate * dt
            delta = smoothed[0] - self.prev_action[0]
            if abs(delta) > max_delta:
                smoothed[0] = self.prev_action[0] + np.sign(delta) * max_delta

            self.prev_action = smoothed.copy()

            # Publish
            msg = AckermannDriveStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"
            msg.drive.steering_angle = np.clip(float(smoothed[0]), -self.max_steer, self.max_steer)
            msg.drive.speed = np.clip(float(smoothed[1]), 0.0, self.eff_max_speed)
            self.drive_pub.publish(msg)

            # Stats
            ms = (time.perf_counter() - t0) * 1000
            self.inference_count += 1
            self.total_inference_time += ms
            if self.inference_count % 200 == 0:
                avg = self.total_inference_time / self.inference_count
                self.get_logger().info(
                    f"[{avg:.1f}ms] steer={msg.drive.steering_angle:.3f} "
                    f"speed={msg.drive.speed:.2f} vel={self.current_vel:.1f}"
                )

        def _build_obs(self) -> np.ndarray:
            """Build observation vector — matches training preprocessing exactly."""
            parts = []

            # 1. Lidar: downsample 1080 → num_beams, clip, normalize
            stride = max(1, len(self.current_scan) // self.num_beams)
            scan = self.current_scan[::stride][:self.num_beams]
            scan = np.clip(scan, 0.0, self.lidar_clip)
            if self.lidar_normalize:
                scan = scan / self.lidar_clip
            parts.append(scan.astype(np.float32))

            # 2. Velocity
            if self.include_velocity:
                parts.append(np.array([self.current_vel / 10.0], dtype=np.float32))

            # 3. Yaw rate
            if self.include_yaw_rate:
                parts.append(np.array([self.current_yaw_rate / 3.14], dtype=np.float32))

            # 4. Previous action
            if self.include_prev_action:
                parts.append(self.prev_action.astype(np.float32))

            # 5. Waypoint features (LOCALIZATION MODE ONLY)
            if self.use_localization and self.waypoints is not None:
                wp_features = self._compute_waypoint_features()
                parts.append(wp_features)

            return np.concatenate(parts)

        def _compute_waypoint_features(self) -> np.ndarray:
            """
            Compute waypoint-relative features from particle filter pose.
            Matches ObservationBuilder._compute_waypoint_features() exactly.

            Returns: [dist_1, heading_1, dist_2, heading_2, ...] for N waypoints
            """
            if self.current_pose is None:
                return np.zeros(self.num_waypoints * 2, dtype=np.float32)

            x, y, theta = self.current_pose

            # Find closest waypoint
            dists = np.sqrt(
                (self.waypoints[:, 0] - x)**2 + (self.waypoints[:, 1] - y)**2
            )
            closest = np.argmin(dists)

            features = []
            n_wp = len(self.waypoints)
            for i in range(self.num_waypoints):
                wp_idx = (closest + i + 1) % n_wp
                wp = self.waypoints[wp_idx]

                dx, dy = wp[0] - x, wp[1] - y
                dist = np.sqrt(dx**2 + dy**2)
                wp_angle = np.arctan2(dy, dx)
                heading_err = wp_angle - theta
                while heading_err > np.pi: heading_err -= 2 * np.pi
                while heading_err < -np.pi: heading_err += 2 * np.pi

                features.extend([dist / 10.0, heading_err / np.pi])

            return np.array(features, dtype=np.float32)

        def _stop(self):
            msg = AckermannDriveStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.drive.speed = 0.0
            msg.drive.steering_angle = 0.0
            self.drive_pub.publish(msg)


def main():
    if not ROS2_AVAILABLE:
        print("Install ROS2 Humble: sudo apt install ros-humble-desktop")
        return
    rclpy.init()
    node = RLInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Stopping car")
        node._stop()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
