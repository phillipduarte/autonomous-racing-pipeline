import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from robo_arp_interfaces.srv import SetActive


class WallFollowerWrapperNode(Node):
    """
    Thin gate between the upstream wall_follower_node and the /drive topic.
    Subscribes to /drive_raw (remapped from the real wall follower's /drive output),
    exposes wall_follower/set_active, and forwards messages only when active.
    Starts inactive; the coordinator activates it.
    """

    def __init__(self):
        super().__init__('wall_follower_wrapper_node')

        self._active = False

        self._drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        self._drive_raw_sub = self.create_subscription(
            AckermannDriveStamped, '/drive_raw', self._drive_raw_callback, 10)

        self._set_active_srv = self.create_service(
            SetActive, 'wall_follower/set_active', self._set_active_handler)

        self.get_logger().info('WallFollowerWrapper ready — starting INACTIVE')

    def _drive_raw_callback(self, msg: AckermannDriveStamped):
        if self._active:
            self._drive_pub.publish(msg)

    def _set_active_handler(self, request: SetActive.Request, response: SetActive.Response):
        prev = self._active
        self._active = request.active

        if not request.active and prev:
            stop = AckermannDriveStamped()
            stop.drive.speed = 0.0
            stop.drive.steering_angle = 0.0
            self._drive_pub.publish(stop)

        response.success = True
        response.message = f'Wall follower {"activated" if self._active else "deactivated"}'
        self.get_logger().info(response.message)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = WallFollowerWrapperNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
