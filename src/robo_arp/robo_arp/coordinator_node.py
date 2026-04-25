import enum
import subprocess
import time
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import Bool, String
from ackermann_msgs.msg import AckermannDriveStamped
from robo_arp_interfaces.srv import SetActive, SaveMap


class PipelineState(enum.Enum):
    IDLE = 'IDLE'
    EXPLORE = 'EXPLORE'
    SAVING = 'SAVING'
    DONE = 'DONE'
    EMERGENCY = 'EMERGENCY'


class CoordinatorNode(Node):
    def __init__(self):
        super().__init__('coordinator_node')

        self.declare_parameter('explore_wall_follow_speed', 1.5)
        self.declare_parameter('map_save_path', '/tmp/robo_arp_map')

        self._map_save_path = self.get_parameter('map_save_path').value

        self._state = PipelineState.IDLE
        self._cb_group = ReentrantCallbackGroup()

        self._state_pub = self.create_publisher(String, 'coordinator/state', 10)
        self._raceline_pub = self.create_publisher(String, 'coordinator/current_raceline', 10)
        self._drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._converged_sub = self.create_subscription(
            Bool, 'slam_monitor/converged', self._converged_callback, latched_qos,
            callback_group=self._cb_group)
        self._emergency_sub = self.create_subscription(
            Bool, 'safety_monitor/emergency', self._emergency_callback, 10,
            callback_group=self._cb_group)

        self._wall_follower_client = self.create_client(
            SetActive, 'wall_follower/set_active', callback_group=self._cb_group)
        self._save_map_client = self.create_client(
            SaveMap, 'slam_monitor/save_map', callback_group=self._cb_group)

        if not self._wall_follower_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn(
                'wall_follower/set_active not available at startup — will retry on use')
        if not self._save_map_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn(
                'slam_monitor/save_map not available at startup — will retry on use')

        # Transition immediately from IDLE to EXPLORE
        self._initial_timer = self.create_timer(
            0.1, self._initial_transition, callback_group=self._cb_group)

    def _initial_transition(self):
        self._initial_timer.cancel()
        self._transition_to_explore()

    def _publish_state(self, state: PipelineState):
        msg = String()
        msg.data = state.value
        self._state_pub.publish(msg)

    def _publish_stop(self):
        stop = AckermannDriveStamped()
        stop.drive.speed = 0.0
        stop.drive.steering_angle = 0.0
        self._drive_pub.publish(stop)

    def kill_node(self, node_name):
        result = subprocess.run(
            ['ros2', 'node', 'kill', node_name],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            self.get_logger().info(f'Stopped node {node_name}')
        else:
            self.get_logger().warn(
                f'Could not stop {node_name} (may already be stopped): {result.stderr.strip()}')

    def call_service_sync(self, client, request, timeout=5.0):
        if not client.wait_for_service(timeout_sec=timeout):
            self.get_logger().error(f'Service {client.srv_name} not available')
            return None, False
        future = client.call_async(request)
        deadline = time.monotonic() + timeout
        while not future.done():
            if time.monotonic() > deadline:
                self.get_logger().error(f'Service {client.srv_name} timed out')
                return None, False
            time.sleep(0.005)
        return future.result(), True

    def _transition_to_explore(self):
        if self._state != PipelineState.IDLE:
            return
        self._state = PipelineState.EXPLORE
        self._publish_state(PipelineState.EXPLORE)

        req = SetActive.Request()
        req.active = True
        resp, ok = self.call_service_sync(self._wall_follower_client, req)
        if not ok or not resp.success:
            self.get_logger().error(
                'Failed to activate wall follower — transitioning to EMERGENCY')
            self._transition_to_emergency()
            return

        raceline_msg = String()
        raceline_msg.data = ''
        self._raceline_pub.publish(raceline_msg)

        self.get_logger().info('Entering EXPLORE: wall follower active, SLAM running')

    def _converged_callback(self, msg: Bool):
        if not msg.data:
            return
        if self._state != PipelineState.EXPLORE:
            return
        self._transition_to_saving()

    def _transition_to_saving(self):
        if self._state != PipelineState.EXPLORE:
            return
        self._state = PipelineState.SAVING
        self._publish_state(PipelineState.SAVING)

        req = SetActive.Request()
        req.active = False
        resp, ok = self.call_service_sync(self._wall_follower_client, req)
        if not ok or (resp is not None and not resp.success):
            self.get_logger().warn('Failed to deactivate wall follower — continuing with map save')

        self.get_logger().info('Map converged, stopping wall follower, saving map...')

        save_req = SaveMap.Request()
        save_req.map_path = self._map_save_path
        save_resp, ok = self.call_service_sync(self._save_map_client, save_req, timeout=15.0)

        if ok and save_resp is not None and save_resp.success:
            self.get_logger().info(f'Map saved to {save_resp.pgm_path}')
            self._transition_to_done(save_resp.pgm_path)
        else:
            msg = save_resp.message if (save_resp is not None) else 'unknown error'
            self.get_logger().error(f'Map save failed: {msg}')
            self._transition_to_emergency()

    def _transition_to_done(self, map_path: str):
        if self._state not in (PipelineState.SAVING,):
            return
        self._state = PipelineState.DONE
        self._publish_state(PipelineState.DONE)

        # TODO: Stop car safely, and then record baseline/odom from slam
        # Then, when we transition to next phase, we pass in this position
        # as starting position in particle filter before we let the car
        # start driving again.

        self.kill_node('/slam_toolbox')
        self.kill_node('/wall_follower')
        self.get_logger().info(
            f'Phase 1 complete. Map at: {map_path}. slam_toolbox and wall_follower stopped.')

    def _emergency_callback(self, msg: Bool):
        if not msg.data:
            return
        if self._state not in (PipelineState.EXPLORE,):
            return
        self._transition_to_emergency()

    def _transition_to_emergency(self):
        if self._state == PipelineState.EMERGENCY:
            return
        self._state = PipelineState.EMERGENCY
        self._publish_state(PipelineState.EMERGENCY)

        req = SetActive.Request()
        req.active = False
        self.call_service_sync(self._wall_follower_client, req)

        self._publish_stop()
        self.kill_node('/slam_toolbox')
        self.kill_node('/wall_follower')
        self.get_logger().error(
            'Emergency stop triggered — cross-track error exceeded threshold')


def main(args=None):
    rclpy.init(args=args)
    node = CoordinatorNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
