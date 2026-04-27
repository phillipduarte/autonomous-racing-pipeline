import csv
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped, PointStamped
from nav_msgs.msg import Path
from std_msgs.msg import String
from ackermann_msgs.msg import AckermannDriveStamped

from robo_arp_interfaces.srv import SetActive


class PurePursuitNode(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')

        self.declare_parameter('raceline_path', '')
        self.declare_parameter('lookahead_distance', 0.8)
        self.declare_parameter('speed', 1.5)
        self.declare_parameter('wheelbase', 0.33)

        self._active = False
        self._path: list[tuple[float, float]] = []
        self._pending_path = ''   # path received on topic, loaded on next activation
        self._closest_idx = 0

        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0
        self._odom_received = False

        self._drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self._lookahead_pub = self.create_publisher(PointStamped, 'pure_pursuit/lookahead_point', 10)

        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._path_pub = self.create_publisher(Path, 'pure_pursuit/path', latched_qos)

        self.create_subscription(PoseStamped, '/pf/viz/inferred_pose', self._odom_cb, 10)
        self.create_subscription(String, 'coordinator/current_raceline', self._raceline_cb, 10)

        self.create_service(SetActive, 'pure_pursuit/set_active', self._set_active_handler)

        self._timer = self.create_timer(0.02, self._control_loop)  # 50 Hz

        self.get_logger().info('PurePursuitNode ready — starting INACTIVE')

    # ------------------------------------------------------------------
    # Subscriptions

    def _odom_cb(self, msg: PoseStamped):
        self._x = msg.pose.position.x
        self._y = msg.pose.position.y
        q = msg.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._yaw = math.atan2(siny_cosp, cosy_cosp)
        self._odom_received = True

    def _raceline_cb(self, msg: String):
        path = msg.data.strip()
        if path:
            self._pending_path = path

    # ------------------------------------------------------------------
    # Service

    def _set_active_handler(self, request: SetActive.Request, response: SetActive.Response):
        prev = self._active
        self._active = request.active

        if request.active:
            path_to_load = self._pending_path or self.get_parameter('raceline_path').value
            if not path_to_load:
                response.success = False
                response.message = 'No raceline path available'
                self._active = False
                return response
            if not self._load_path(path_to_load):
                response.success = False
                response.message = f'Failed to load raceline from {path_to_load}'
                self._active = False
                return response
            self._closest_idx = 0
            self.get_logger().info(
                f'Pure pursuit activated — {len(self._path)} waypoints from {path_to_load}')
        elif prev:
            self._publish_stop()
            self.get_logger().info('Pure pursuit deactivated')

        response.success = True
        response.message = f'Pure pursuit {"activated" if self._active else "deactivated"}'
        return response

    # ------------------------------------------------------------------
    # Path loading

    def _load_path(self, path: str) -> bool:
        try:
            with open(path, newline='') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if not rows:
                self.get_logger().error(f'Raceline CSV is empty: {path}')
                return False
            # Support both x_m/y_m (from generate_centerline.py) and x/y column names
            if 'x_m' in rows[0] and 'y_m' in rows[0]:
                self._path = [(float(r['x_m']), float(r['y_m'])) for r in rows]
            elif 'x' in rows[0] and 'y' in rows[0]:
                self._path = [(float(r['x']), float(r['y'])) for r in rows]
            else:
                self.get_logger().error(
                    f'CSV has no x/y or x_m/y_m columns. Headers: {list(rows[0].keys())}')
                return False
            self._publish_path()
            return True
        except Exception as e:
            self.get_logger().error(f'Error loading raceline {path}: {e}')
            return False

    def _publish_path(self):
        msg = Path()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        for x, y in self._path:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = x
            ps.pose.position.y = y
            msg.poses.append(ps)
        self._path_pub.publish(msg)

    # ------------------------------------------------------------------
    # Control loop

    def _control_loop(self):
        if not self._active or not self._path or not self._odom_received:
            return

        lookahead = self.get_parameter('lookahead_distance').value
        speed = self.get_parameter('speed').value
        wheelbase = self.get_parameter('wheelbase').value

        goal = self._find_lookahead_point(lookahead)
        if goal is None:
            return

        pt = PointStamped()
        pt.header.frame_id = 'map'
        pt.header.stamp = self.get_clock().now().to_msg()
        pt.point.x = goal[0]
        pt.point.y = goal[1]
        self._lookahead_pub.publish(pt)

        # Transform goal to vehicle frame
        dx = goal[0] - self._x
        dy = goal[1] - self._y
        local_x = math.cos(-self._yaw) * dx - math.sin(-self._yaw) * dy
        local_y = math.sin(-self._yaw) * dx + math.cos(-self._yaw) * dy

        # Pure pursuit curvature → steering angle
        dist_sq = local_x * local_x + local_y * local_y
        if dist_sq < 1e-6:
            steering = 0.0
        else:
            curvature = 2.0 * local_y / dist_sq
            steering = math.atan(curvature * wheelbase)
        steering = max(-0.4, min(0.4, steering))

        msg = AckermannDriveStamped()
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steering)
        self._drive_pub.publish(msg)

    def _find_lookahead_point(self, lookahead: float):
        """Return the first waypoint at least lookahead distance ahead, starting from
        the nearest waypoint to the current position."""
        n = len(self._path)
        if n == 0:
            return None

        # Update closest index
        best_dist = float('inf')
        best_idx = self._closest_idx
        search_range = min(n, 50)
        for i in range(search_range):
            idx = (self._closest_idx + i) % n
            px, py = self._path[idx]
            d = math.hypot(px - self._x, py - self._y)
            if d < best_dist:
                best_dist = d
                best_idx = idx
        self._closest_idx = best_idx

        # Find lookahead point
        for i in range(n):
            idx = (self._closest_idx + i) % n
            px, py = self._path[idx]
            if math.hypot(px - self._x, py - self._y) >= lookahead:
                return self._path[idx]

        # Fallback: return the farthest point
        return self._path[(self._closest_idx + n // 2) % n]

    def _publish_stop(self):
        stop = AckermannDriveStamped()
        stop.drive.speed = 0.0
        stop.drive.steering_angle = 0.0
        self._drive_pub.publish(stop)


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitNode()
    rclpy.spin(node)
    rclpy.shutdown()
