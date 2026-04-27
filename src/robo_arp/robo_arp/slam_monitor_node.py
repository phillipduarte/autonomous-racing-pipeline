import os
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool, Float32
from robo_arp_interfaces.srv import SaveMap


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
        self._last_map: OccupancyGrid = None

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

    def _map_callback(self, msg: OccupancyGrid):
        self._last_map = msg
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

        if self._last_map is None:
            response.success = False
            response.message = 'No map received yet'
            return response

        try:
            pgm_path, yaml_path = self._write_map(self._last_map, map_path)
        except Exception as e:
            response.success = False
            response.message = f'Map write failed: {e}'
            return response

        response.success = True
        response.message = 'Map saved successfully'
        response.pgm_path = pgm_path
        response.yaml_path = yaml_path
        self.get_logger().info(f'Map saved to {pgm_path}')
        return response

    def _write_map(self, map_msg: OccupancyGrid, map_path: str):
        info = map_msg.info
        width, height = info.width, info.height
        resolution = info.resolution
        ox = info.origin.position.x
        oy = info.origin.position.y

        data = np.array(map_msg.data, dtype=np.int8).reshape(height, width)

        # ROS occupancy → PGM: free=254, occupied=0, unknown=205
        pixels = np.full((height, width), 205, dtype=np.uint8)
        pixels[data == 0] = 254
        pixels[data == 100] = 0
        mid = (data > 0) & (data < 100)
        pixels[mid] = (255 - (data[mid].astype(np.float32) / 100.0) * 254).astype(np.uint8)

        # ROS maps are stored bottom-row-first; PGM is top-row-first
        pixels = np.flipud(pixels)

        os.makedirs(os.path.dirname(map_path) or '.', exist_ok=True)
        pgm_path = map_path + '.pgm'
        with open(pgm_path, 'wb') as f:
            f.write(f'P5\n{width} {height}\n255\n'.encode())
            f.write(pixels.tobytes())

        yaml_path = map_path + '.yaml'
        with open(yaml_path, 'w') as f:
            f.write(
                f'image: {os.path.basename(pgm_path)}\n'
                f'resolution: {resolution}\n'
                f'origin: [{ox:.6f}, {oy:.6f}, 0.0]\n'
                f'negate: 0\n'
                f'occupied_thresh: 0.65\n'
                f'free_thresh: 0.196\n'
            )

        return pgm_path, yaml_path


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
