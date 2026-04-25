import enum
import subprocess
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import Bool, String
from ackermann_msgs.msg import AckermannDriveStamped
from robo_arp_interfaces.srv import SetActive, SaveMap, ProcessMap


class PipelineState(enum.Enum):
    IDLE               = 'IDLE'
    EXPLORE            = 'EXPLORE'
    SAVING             = 'SAVING'
    PLANNING           = 'PLANNING'
    RACING_CENTERLINE  = 'RACING_CENTERLINE'
    EMERGENCY          = 'EMERGENCY'


class CoordinatorNode(Node):
    def __init__(self):
        super().__init__('coordinator_node')

        self.declare_parameter('explore_wall_follow_speed', 1.5)
        self.declare_parameter('map_save_path', '/tmp/robo_arp_map')

        self._map_save_path = self.get_parameter('map_save_path').value
        self._map_pgm_path = ''
        self._map_yaml_path = ''
        self._centerline_path = ''

        self._state = PipelineState.IDLE

        self._state_pub = self.create_publisher(String, 'coordinator/state', 10)
        self._raceline_pub = self.create_publisher(String, 'coordinator/current_raceline', 10)
        self._drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._converged_sub = self.create_subscription(
            Bool, 'slam_monitor/converged', self._converged_callback, latched_qos)
        self._emergency_sub = self.create_subscription(
            Bool, 'safety_monitor/emergency', self._emergency_callback, 10)

        self._wall_follower_client = self.create_client(
            SetActive, 'wall_follower/set_active')
        self._save_map_client = self.create_client(
            SaveMap, 'slam_monitor/save_map')
        self._process_map_client = self.create_client(
            ProcessMap, 'map_processor/process')
        self._pure_pursuit_active_client = self.create_client(
            SetActive, 'pure_pursuit/set_active')

        for client, name in [
            (self._wall_follower_client, 'wall_follower/set_active'),
            (self._save_map_client, 'slam_monitor/save_map'),
            (self._process_map_client, 'map_processor/process'),
            (self._pure_pursuit_active_client, 'pure_pursuit/set_active'),
        ]:
            if not client.wait_for_service(timeout_sec=5.0):
                self.get_logger().warn(
                    f'{name} not available at startup — will retry on use')

        self.create_timer(0.1, self._initial_transition)

    def _initial_transition(self):
        self._initial_timer = None
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
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if future.result() is None:
            self.get_logger().error(f'Service {client.srv_name} timed out')
            return None, False
        return future.result(), True

    # ------------------------------------------------------------------
    # IDLE → EXPLORE

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

    # ------------------------------------------------------------------
    # EXPLORE → SAVING

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
            self.get_logger().warn(
                'Failed to deactivate wall follower — continuing with map save')

        self.get_logger().info('Map converged, stopping wall follower, saving map...')

        save_req = SaveMap.Request()
        save_req.map_path = self._map_save_path
        save_resp, ok = self.call_service_sync(self._save_map_client, save_req, timeout=15.0)

        if ok and save_resp is not None and save_resp.success:
            self.get_logger().info(f'Map saved to {save_resp.pgm_path}')
            self._transition_to_planning(save_resp.pgm_path, save_resp.yaml_path)
        else:
            msg = save_resp.message if (save_resp is not None) else 'unknown error'
            self.get_logger().error(f'Map save failed: {msg}')
            self._transition_to_emergency()

    # ------------------------------------------------------------------
    # SAVING → PLANNING

    def _transition_to_planning(self, pgm_path: str, yaml_path: str):
        if self._state != PipelineState.SAVING:
            return
        self._state = PipelineState.PLANNING
        self._publish_state(PipelineState.PLANNING)

        self.kill_node('/slam_toolbox')
        self.kill_node('/wall_follower')

        self._map_pgm_path = pgm_path
        self._map_yaml_path = yaml_path

        proc_req = ProcessMap.Request()
        proc_req.pgm_path = self._map_pgm_path
        proc_resp, ok = self.call_service_sync(
            self._process_map_client, proc_req, timeout=90.0)

        if ok and proc_resp is not None and proc_resp.success:
            self._centerline_path = proc_resp.centerline_path
            self.get_logger().info(f'Centerline ready at {self._centerline_path}')
            self._transition_to_racing_centerline()
        else:
            msg = proc_resp.message if (proc_resp is not None) else 'unknown error'
            self.get_logger().error(f'Centerline extraction failed: {msg}')
            self._transition_to_emergency()

    # ------------------------------------------------------------------
    # PLANNING → RACING_CENTERLINE

    def _transition_to_racing_centerline(self):
        if self._state != PipelineState.PLANNING:
            return
        self._state = PipelineState.RACING_CENTERLINE
        self._publish_state(PipelineState.RACING_CENTERLINE)

        # Publish centerline path so pure pursuit and safety monitor receive it
        raceline_msg = String()
        raceline_msg.data = self._centerline_path
        self._raceline_pub.publish(raceline_msg)

        req = SetActive.Request()
        req.active = True
        resp, ok = self.call_service_sync(self._pure_pursuit_active_client, req)
        if not ok or (resp is not None and not resp.success):
            self.get_logger().error(
                'Failed to activate pure pursuit — transitioning to EMERGENCY')
            self._transition_to_emergency()
            return

        self.get_logger().info(f'Racing on centerline: {self._centerline_path}')

    # ------------------------------------------------------------------
    # EMERGENCY

    def _emergency_callback(self, msg: Bool):
        if not msg.data:
            return
        if self._state not in {
            PipelineState.EXPLORE,
            PipelineState.PLANNING,
            PipelineState.RACING_CENTERLINE,
        }:
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

        # Deactivate pure pursuit if it may be running
        pp_req = SetActive.Request()
        pp_req.active = False
        self.call_service_sync(self._pure_pursuit_active_client, pp_req)

        self._publish_stop()
        self.kill_node('/slam_toolbox')
        self.kill_node('/wall_follower')
        self.get_logger().error('Emergency stop triggered')


def main(args=None):
    rclpy.init(args=args)
    node = CoordinatorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
