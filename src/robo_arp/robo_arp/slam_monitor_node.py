import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool, Float32
from robo_arp_interfaces.srv import SaveMap
from slam_toolbox.srv import SaveMap as SlamSaveMap


class SlamMonitorNode(Node):
    def __init__(self):
        super().__init__('slam_monitor_node')

        self.declare_parameter('stability_threshold_m2', 0.05)
        self.declare_parameter('stable_frames_required', 10)
        self.declare_parameter('min_map_area_m2', 2.0)
        self.declare_parameter('map_save_path', '/tmp/robo_arp_map')

        self._stability_threshold = self.get_parameter('stability_threshold_m2').value
        self._stable_frames_required = self.get_parameter('stable_frames_required').value
        self._min_map_area = self.get_parameter('min_map_area_m2').value
        self._map_save_path = self.get_parameter('map_save_path').value

        self._previous_area = 0.0
        self._stable_count = 0
        self._already_converged = False

        self._cb_group = ReentrantCallbackGroup()

        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self._converged_pub = self.create_publisher(Bool, 'slam_monitor/converged', latched_qos)
        self._area_pub = self.create_publisher(Float32, 'slam_monitor/map_area', 10)

        self._map_sub = self.create_subscription(
            OccupancyGrid, '/map', self._map_callback, 10,
            callback_group=self._cb_group)

        self._save_map_srv = self.create_service(
            SaveMap, 'slam_monitor/save_map', self._save_map_handler,
            callback_group=self._cb_group)

        self._slam_save_client = self.create_client(
            SlamSaveMap, '/slam_toolbox/save_map', callback_group=self._cb_group)
        if not self._slam_save_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().warn(
                '/slam_toolbox/save_map service not available at startup — will retry on request')

    def _map_callback(self, msg: OccupancyGrid):
        resolution = msg.info.resolution
        free_cells = int(np.count_nonzero(np.asarray(msg.data, dtype=np.int8) == 0))
        area = free_cells * resolution * resolution

        delta = abs(area - self._previous_area)

        self.get_logger().info(
            f'Received map: area={area:.2f}m², delta={delta:.2f}m²')

        if delta < self._stability_threshold:
            self._stable_count += 1
            self.get_logger().info(
                f'Map stable frame logged: delta={delta:.2f}, '
                f'stable for stable_count={self._stable_count} frames, '
                f'need {self._stable_frames_required} for convergence')
        else:
            self._stable_count = 0
        self._previous_area = area

        area_msg = Float32()
        area_msg.data = float(area)
        self._area_pub.publish(area_msg)

        if (self._stable_count >= self._stable_frames_required
                and area >= self._min_map_area
                and not self._already_converged):
            self._already_converged = True
            converged_msg = Bool()
            converged_msg.data = True
            self._converged_pub.publish(converged_msg)
            self.get_logger().info(
                f'Map converged: area={area:.2f}m², '
                f'stable for {self._stable_frames_required} frames')

    def _save_map_handler(self, request: SaveMap.Request, response: SaveMap.Response):
        map_path = request.map_path if request.map_path else self._map_save_path

        if not self._slam_save_client.wait_for_service(timeout_sec=5.0):
            response.success = False
            response.message = '/slam_toolbox/save_map service not available'
            return response

        slam_req = SlamSaveMap.Request()
        slam_req.name.data = map_path

        future = self._slam_save_client.call_async(slam_req)
        deadline = time.monotonic() + 10.0
        while not future.done():
            if time.monotonic() > deadline:
                break
            time.sleep(0.005)

        if not future.done() or future.result() is None:
            response.success = False
            response.message = 'slam_toolbox save_map call timed out'
            return response

        response.success = True
        response.message = 'Map saved successfully'
        response.pgm_path = map_path + '.pgm'
        response.yaml_path = map_path + '.yaml'
        self.get_logger().info(f'Map saved to {response.pgm_path}')
        return response


def main(args=None):
    rclpy.init(args=args)
    node = SlamMonitorNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
