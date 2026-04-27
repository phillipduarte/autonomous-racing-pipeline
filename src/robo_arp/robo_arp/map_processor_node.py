import os
import subprocess

import yaml
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
            response.message = 'centerline_script_path not set or file not found' + script_path
            return response

        map_base = request.pgm_path.replace('.pgm', '')
        centerline_path = map_base + '_centerline.csv'

        cmd = ['python3', script_path, '--map', map_base, '--no-plot']

        if request.seed_x != 0.0 or request.seed_y != 0.0:
            seed_px, seed_py = self._meters_to_pixels(
                map_base + '.yaml', request.seed_x, request.seed_y)
            if seed_px is not None:
                cmd += ['--seed', str(seed_py), str(seed_px)]  # script expects row, col
                self.get_logger().info(
                    f'Seed: map=({request.seed_x:.3f}m, {request.seed_y:.3f}m) '
                    f'-> pixel=(row={seed_py}, col={seed_px})')
            else:
                self.get_logger().warn('Could not read map YAML — running without seed')

        self.get_logger().info(f'Running centerline extraction: {" ".join(cmd)}')
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60.0)
        except subprocess.TimeoutExpired:
            response.success = False
            response.message = 'Centerline script timed out after 60 seconds'
            self.get_logger().error(response.message)
            return response

        if result.stdout.strip():
            self.get_logger().info(f'Centerline script output:\n{result.stdout.strip()}')
        if result.returncode == 0:
            response.success = True
            response.centerline_path = centerline_path
            response.message = 'Centerline extracted successfully'
            self.get_logger().info(f'Centerline written to {centerline_path}')
        else:
            if result.stderr.strip():
                self.get_logger().error(
                    f'Centerline script stderr:\n{result.stderr.strip()}')
            response.success = False
            response.message = (
                f'Script failed (exit {result.returncode}): {result.stderr.strip()}')
            self.get_logger().error(
                f'Centerline extraction failed (exit {result.returncode})')

        return response

    def _meters_to_pixels(self, yaml_path, mx, my):
        try:
            with open(yaml_path) as f:
                meta = yaml.safe_load(f)
            res = float(meta['resolution'])
            ox, oy = float(meta['origin'][0]), float(meta['origin'][1])
            pgm = yaml_path.replace('.yaml', '.pgm')
            height = self._pgm_height(pgm)
            px = int((mx - ox) / res)
            py = int(height - 1 - (my - oy) / res)
            return px, py
        except Exception as e:
            self.get_logger().error(f'meters_to_pixels failed: {e}')
            return None, None

    @staticmethod
    def _pgm_height(pgm_path):
        with open(pgm_path, 'rb') as f:
            # skip magic line and any comment lines
            while True:
                line = f.readline().decode('ascii').strip()
                if not line.startswith('#'):
                    break
            # next non-comment line is "width height"
            while True:
                line = f.readline().decode('ascii').strip()
                if not line.startswith('#'):
                    break
            return int(line.split()[1])


def main(args=None):
    rclpy.init(args=args)
    node = MapProcessorNode()
    rclpy.spin(node)
    rclpy.shutdown()
