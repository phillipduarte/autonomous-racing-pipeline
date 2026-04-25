import math
import csv
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Float32, String


class SafetyMonitorNode(Node):
    def __init__(self):
        super().__init__('safety_monitor_node')

        self.declare_parameter('cross_track_error_threshold', 0.4)
        self.declare_parameter('check_rate_hz', 20.0)

        self._cte_threshold = self.get_parameter('cross_track_error_threshold').value
        self._check_rate_hz = self.get_parameter('check_rate_hz').value

        self._raceline: list[tuple[float, float]] = []
        self._current_x = 0.0
        self._current_y = 0.0
        self._odom_received = False

        self._cte_pub = self.create_publisher(Float32, 'safety_monitor/cross_track_error', 10)
        self._emergency_pub = self.create_publisher(Bool, 'safety_monitor/emergency', 10)

        self._odom_sub = self.create_subscription(
            PoseStamped, '/pf/viz/inferred_pose', self._odom_callback, 10)
        self._raceline_sub = self.create_subscription(
            String, 'coordinator/current_raceline', self._raceline_callback, 10)

        self._timer = self.create_timer(1.0 / self._check_rate_hz, self._check_cte)

    def _odom_callback(self, msg: PoseStamped):
        self._current_x = msg.pose.position.x
        self._current_y = msg.pose.position.y
        self._odom_received = True

    def _raceline_callback(self, msg: String):
        path = msg.data.strip()
        if not path:
            self._raceline = []
            return
        try:
            with open(path, newline='') as f:
                reader = csv.DictReader(f)
                self._raceline = [(float(row['x']), float(row['y'])) for row in reader]
            self.get_logger().info(f'Loaded raceline with {len(self._raceline)} points from {path}')
        except Exception as e:
            self.get_logger().error(f'Failed to load raceline from {path}: {e}')
            self._raceline = []

    def _check_cte(self):
        cte_msg = Float32()

        if not self._raceline or not self._odom_received:
            cte_msg.data = 0.0
            self._cte_pub.publish(cte_msg)
            return

        min_dist = min(
            math.hypot(self._current_x - px, self._current_y - py)
            for px, py in self._raceline
        )

        cte_msg.data = float(min_dist)
        self._cte_pub.publish(cte_msg)

        if min_dist > self._cte_threshold:
            emergency_msg = Bool()
            emergency_msg.data = True
            self._emergency_pub.publish(emergency_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SafetyMonitorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
