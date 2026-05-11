from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    package_share = get_package_share_directory('object_detector')
    wall_map_share = get_package_share_directory('wall_map')
    config_path = os.path.join(
        package_share,
        'config',
        'params.yaml'
    )
    default_track_csv_path = os.path.join(
        package_share,
        'track',
        '0120_track.csv'
    )
    default_wall_map_config_path = os.path.join(
        wall_map_share,
        'config',
        'config.yaml'
    )
    default_wall_map_yaml_path = os.path.join(
        wall_map_share,
        'maps',
        '0120.yaml'
    )
    track_csv_path = LaunchConfiguration('track_csv_path')
    wall_map_config_path = LaunchConfiguration('wall_map_config_path')
    wall_map_yaml_path = LaunchConfiguration('wall_map_yaml_path')
    only_static = LaunchConfiguration('only_static')

    return LaunchDescription([
        DeclareLaunchArgument(
            'track_csv_path',
            default_value=default_track_csv_path,
            description='Path to track CSV used for wall/track filtering.'
        ),
        DeclareLaunchArgument(
            'wall_map_config_path',
            default_value=default_wall_map_config_path,
            description='Path to wall_map config.yaml.'
        ),
        DeclareLaunchArgument(
            'wall_map_yaml_path',
            default_value=default_wall_map_yaml_path,
            description='Path to wall_map map YAML.'
        ),
        DeclareLaunchArgument(
            'only_static',
            default_value='false',
            description='When true, classify all valid obstacles as static.'
        ),
        Node(
            package='object_detector',
            executable='scan_processor_node',
            name='scan_processor_node',
            output='log',
            parameters=[config_path]
        ),
        Node(
            package='object_detector',
            executable='obstacle_detector_node',
            name='obstacle_detector_node',
            output='screen',
            parameters=[
                config_path,
                {
                    'track_csv_path': track_csv_path,
                    'wall_map.config_path': wall_map_config_path,
                    'wall_map.yaml_path': wall_map_yaml_path,
                    'only_static': ParameterValue(only_static, value_type=bool),
                }
            ]
        ),
        Node(
            package='object_detector',
            executable='visualization_node',
            name='visualization_node',
            output='log',
            parameters=[config_path]
        ),
        Node(
            package='object_detector',
            executable='delay_monitor_node',
            name='delay_monitor_node',
            output='log',
            parameters=[config_path]
        )
    ])
