import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Package directory
    pkg_dir = get_package_share_directory('object_detector')
    
    # Launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )
    
    icp_method_arg = DeclareLaunchArgument(
        'icp_method',
        default_value='point_to_point',
        description='ICP method: point_to_point, point_to_plane, point_to_line, gicp'
    )
    
    # Node
    icp_comparator_node = Node(
        package='object_detector',
        executable='icp_comparator_node',
        name='icp_comparator_node',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            
            # General parameters
            'icp_method': LaunchConfiguration('icp_method'),
            'max_history_frames': 10,
            'throttle_processing': False,
            'process_every_n_frames': 1,  # 40Hz -> 20Hz
            'min_delta_trans': 0.0,       # meters
            'min_delta_rot': 0.0,          # radians (~5.7 degrees)
            'enable_icp': True,
            'max_history_for_icp': 10,     # Apply ICP to last 3 frames
            
            # Common ICP parameters
            'icp.max_iterations': 5,
            'icp.max_corr_dist': 0.2,
            'icp.trans_eps': 1e-4,
            'icp.fit_eps': 1e-5,
            'icp.voxel_leaf': 0.05,
            'icp.use_downsample': True,
            'icp.reject_far': True,
            'icp.reject_radius': 5.0,
            'icp.min_points': 20,
            
            # Point-to-Plane specific
            'pt2pl.normal_use_radius': False,
            'pt2pl.normal_radius': 0.3,
            'pt2pl.normal_k': 10,
            'pt2pl.assume_planar': True,  # For 2D laser scans
            
            # Point-to-Line specific
            'pt2line.line_fitting_dist': 0.1,
            'pt2line.min_line_points': 5,
            'pt2line.outlier_threshold': 0.1,
            
            # GICP specific
            'gicp.correspondence_randomness': 20,
            'gicp.max_optimizer_iterations': 20,
            'gicp.rotation_epsilon': 2e-3,
            'gicp.use_reciprocal': False,
        }],
        remappings=[
            ('/scan', '/scan'),
            ('/odom', '/odom'),
        ]
    )
    
    return LaunchDescription([
        use_sim_time_arg,
        icp_method_arg,
        icp_comparator_node,
    ])