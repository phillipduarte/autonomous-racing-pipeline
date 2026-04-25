# SIM LAUNCH - currently identical to hardware, simulator integration TODO
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('robo_arp'),
        'config', 'params.yaml'
    )

    return LaunchDescription([

        # TODO: add simulator launch
        # Replace slam_toolbox with a static map server for sim testing,
        # and add the F1TENTH gym simulator node when available.

        Node(
            package='slam_toolbox',
            executable='async_slam_toolbox_node',
            name='slam_toolbox',
            parameters=[
                get_package_share_directory('robo_arp') + '/config/slam_params.yaml'
            ],
            output='screen',
        ),

        Node(
            package='wall_follower',
            executable='wall_follower_node',
            name='wall_follower',
            remappings=[('/drive', '/drive_raw')],
            output='screen',
        ),

        Node(
            package='robo_arp',
            executable='wall_follower_wrapper_node',
            name='wall_follower_wrapper',
            output='screen',
        ),

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
    ])
