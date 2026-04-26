import enum
import math
import subprocess
import time
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import Bool, String
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
import tf2_ros
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
        self._cb_group = ReentrantCallbackGroup()

        self._initial_pose = None  # (x, y, yaw) captured from SLAM TF before kill
        self._pf_proc = None       # Popen handle for particle filter process

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._state_pub = self.create_publisher(String, 'coordinator/state', 10)
        self._raceline_pub = self.create_publisher(String, 'coordinator/current_raceline', 10)
        self._drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self._initialpose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/initialpose', 10)

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

        self._initial_timer = self.create_timer(
            0.1, self._initial_transition, callback_group=self._cb_group)

    def _initial_transition(self):
        self._initial_timer.cancel()
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
        pattern = f'__node:={node_name.lstrip("/")}'
        result = subprocess.run(
            ['pkill', '-TERM', '-f', pattern],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            self.get_logger().info(f'Stopped node {node_name}')
        else:
            self.get_logger().warn(
                f'Could not stop {node_name} (may already be stopped)')

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

        # Stop the car and wait for it to settle before reading TF
        self._publish_stop()
        time.sleep(0.5)

        # Capture map→base_link pose from SLAM TF before killing the toolbox
        try:
            tf = self._tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time(),
                timeout=Duration(seconds=1.0))
            tx = tf.transform.translation.x
            ty = tf.transform.translation.y
            q = tf.transform.rotation
            yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            self._initial_pose = (tx, ty, yaw)
            self.get_logger().info(
                f'Captured PF init pose: ({tx:.2f}, {ty:.2f}, {math.degrees(yaw):.1f}°)')
        except Exception as e:
            self.get_logger().warn(
                f'TF lookup failed — PF will use global init: {e}')
            self._initial_pose = None

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

        # Launch particle filter against the map saved during the SAVING phase
        self._pf_proc = subprocess.Popen([
            'ros2', 'launch', 'particle_filter', 'localize_launch.py',
            f'map_yaml:={self._map_yaml_path}',
        ])
        self.get_logger().info('Particle filter launched, waiting for startup...')
        time.sleep(3.0)

        # Seed the particle filter with the pose captured just before SLAM was killed
        if self._initial_pose is not None:
            x, y, yaw = self._initial_pose
            pose_msg = PoseWithCovarianceStamped()
            pose_msg.header.frame_id = 'map'
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.pose.pose.position.x = x
            pose_msg.pose.pose.position.y = y
            cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
            pose_msg.pose.pose.orientation.w = cy
            pose_msg.pose.pose.orientation.z = sy
            self._initialpose_pub.publish(pose_msg)
            self.get_logger().info('Initial pose published to /initialpose')
            time.sleep(0.5)

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
        if self._pf_proc is not None and self._pf_proc.poll() is None:
            self._pf_proc.terminate()
            self._pf_proc = None
        self.get_logger().error('Emergency stop triggered')


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
