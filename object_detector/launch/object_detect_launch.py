from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    config_path = os.path.join(
        get_package_share_directory('object_detector'),
        'config',
        'params.yaml'
    )

    return LaunchDescription([
        Node(
            package='object_detector',
            executable='scan_processor_node',
            name='scan_processor_node',
            output='screen',
            parameters=[config_path]
        ),
        Node(
            package='object_detector',
            executable='obstacle_detector_node',
            name='obstacle_detector_node',
            output='screen',
            parameters=[config_path]
        ),
        Node(
            package='object_detector',
            executable='visualization_node',
            name='visualization_node',
            output='screen',
            parameters=[config_path]
        ),
        Node(
            package='object_detector',
            executable='delay_monitor_node',
            name='delay_monitor_node',
            output='screen',
            parameters=[config_path]
        )
    ])
