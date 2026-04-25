from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('robo_arp'),
        'config', 'params.yaml'
    )

    centerline_script = os.path.join(
        get_package_share_directory('robo_arp'),
        'scripts', 'generate_centerline.py'
    )

    return LaunchDescription([

        # slam_toolbox — runs the whole time
        Node(
            package='slam_toolbox',
            executable='async_slam_toolbox_node',
            name='slam_toolbox',
            parameters=[
                get_package_share_directory('robo_arp') + '/config/slam_params.yaml'
            ],
            output='screen',
        ),

        # wall follower — remapped so wrapper gates its output
        Node(
            package='wall_follow_arp',
            executable='wall_follow_node',
            name='wall_follow',
            remappings=[('/drive', '/drive_raw')],
            output='screen',
        ),

        # wrapper exposes set_active and gates /drive_raw → /drive
        Node(
            package='robo_arp',
            executable='wall_follower_wrapper_node',
            name='wall_follower_wrapper',
            output='screen',
        ),

        # pipeline nodes
        Node(
            package='robo_arp',
            executable='slam_monitor_node',
            name='slam_monitor_node',
            parameters=[config],
            output='screen',
        ),
        Node(
            package='robo_arp',
            executable='safety_monitor_node',
            name='safety_monitor_node',
            parameters=[config],
            output='screen',
        ),
        Node(
            package='robo_arp',
            executable='coordinator_node',
            name='coordinator_node',
            parameters=[config],
            output='screen',
        ),
        Node(
            package='robo_arp',
            executable='map_processor_node',
            name='map_processor_node',
            parameters=[
                config,
                {'centerline_script_path': centerline_script},
            ],
            output='screen',
        ),
        Node(
            package='robo_arp',
            executable='pure_pursuit_node',
            name='pure_pursuit_node',
            parameters=[config],
            output='screen',
        ),
    ])
