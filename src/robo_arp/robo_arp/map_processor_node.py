import os
import subprocess

import rclpy
from rclpy.node import Node

from robo_arp_interfaces.srv import ProcessMap


class MapProcessorNode(Node):
    def __init__(self):
        super().__init__('map_processor_node')
        self.declare_parameter('centerline_script_path', '')

        self.create_service(ProcessMap, 'map_processor/process', self.process_cb)
        self.get_logger().info('MapProcessorNode ready')

    def process_cb(self, request, response):
        script_path = self.get_parameter('centerline_script_path').value

        if not script_path or not os.path.isfile(script_path):
            response.success = False
            response.message = 'centerline_script_path not set or file not found'
            return response

        map_base = request.pgm_path.replace('.pgm', '')
        centerline_path = map_base + '_centerline.csv'

        cmd = ['python3', script_path, '--map', map_base, '--no-plot']

        self.get_logger().info(f'Running centerline extraction: {" ".join(cmd)}')
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60.0)
        except subprocess.TimeoutExpired:
            response.success = False
            response.message = 'Centerline script timed out after 60 seconds'
            self.get_logger().error(response.message)
            return response

        if result.returncode == 0:
            response.success = True
            response.centerline_path = centerline_path
            response.message = 'Centerline extracted successfully'
            self.get_logger().info(f'Centerline written to {centerline_path}')
        else:
            response.success = False
            response.message = (
                f'Script failed (exit {result.returncode}): {result.stderr.strip()}')
            self.get_logger().error(
                f'Centerline extraction failed: {result.stderr.strip()}')

        return response


def main(args=None):
    rclpy.init(args=args)
    node = MapProcessorNode()
    rclpy.spin(node)
    rclpy.shutdown()
