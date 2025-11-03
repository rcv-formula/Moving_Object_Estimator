from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import ThisLaunchFileDir

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='carto_eskf',
            executable='carto_eskf_node',
            name='carto_eskf_node',
            output='screen',
            parameters=['config/carto_eskf.yaml'],
            remappings=[
                ('/corrected_imu', '/airimu/corrected_imu'),  # 필요에 맞게 변경
                ('/odom', '/odom')                            # Cartographer odom
            ]
        )
    ])
