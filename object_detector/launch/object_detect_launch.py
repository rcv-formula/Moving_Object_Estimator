"""
ROS2 launch file for object detection system.

This launch file starts the complete object detection pipeline:
- scan_processor_node: Processes laser scans and detects obstacle candidates
- obstacle_detector_node: Classifies and tracks static/dynamic obstacles
- visualization_node: Visualizes detection results (if available)
- delay_monitor_node: Monitors system latency (if available)

Author: Object Detection Team
License: Apache 2.0
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    """Generate launch description for object detection system."""
    
    # Declare launch arguments
    package_name_arg = DeclareLaunchArgument(
        'package_name',
        default_value='object_detector',
        description='Name of the ROS2 package'
    )
    
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value='params.yaml',
        description='Name of the configuration file'
    )
    
    log_level_arg = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        description='Logging level (debug, info, warn, error, fatal)',
        choices=['debug', 'info', 'warn', 'error', 'fatal']
    )
    
    # Get launch configurations
    package_name = LaunchConfiguration('package_name')
    config_file = LaunchConfiguration('config_file')
    log_level = LaunchConfiguration('log_level')
    
    # Construct path to configuration file
    config_path = PathJoinSubstitution([
        FindPackageShare(package_name),
        'config',
        config_file
    ])
    
    # Common node arguments
    common_args = {
        'package': package_name,
        'output': 'screen',
        'parameters': [config_path],
        'arguments': ['--ros-args', '--log-level', log_level]
    }
    
    # Define nodes
    scan_processor_node = Node(
        executable='scan_processor_node',
        name='scan_processor_node',
        namespace='object_detection',
        **common_args
    )
    
    obstacle_detector_node = Node(
        executable='obstacle_detector_node',
        name='obstacle_detector_node',
        namespace='object_detection',
        **common_args
    )
    
    visualization_node = Node(
        executable='visualization_node',
        name='visualization_node',
        namespace='object_detection',
        **common_args,
        # Optional: This node might not exist in all configurations
        # You can add 'condition' parameter if needed
    )
    
    delay_monitor_node = Node(
        executable='delay_monitor_node',
        name='delay_monitor_node',
        namespace='object_detection',
        **common_args,
        # Optional: This node might not exist in all configurations
    )
    
    # Log startup information
    log_info = LogInfo(
        msg=['Starting object detection system with config: ', config_path]
    )
    
    return LaunchDescription([
        # Launch arguments
        package_name_arg,
        config_file_arg,
        log_level_arg,
        
        # Log
        log_info,
        
        # Core nodes (required)
        scan_processor_node,
        obstacle_detector_node,
        
        # Optional nodes (comment out if not available)
        visualization_node,
        delay_monitor_node,
    ])


if __name__ == '__main__':
    generate_launch_description()