from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import PathJoinSubstitution

def generate_launch_description():
    pkg = 'airio_imu_odometry'
    param_file = PathJoinSubstitution([get_package_share_directory(pkg), 'config', 'carto_airio_ekf_params.yaml'])
    return LaunchDescription([
        Node(
            package=pkg,
            executable='carto_airio_ekf_node',
            name='carto_airio_ekf_node',   # ★ YAML 최상단 키와 반드시 동일
            output='screen',
            parameters=[param_file],
        )
    ])
