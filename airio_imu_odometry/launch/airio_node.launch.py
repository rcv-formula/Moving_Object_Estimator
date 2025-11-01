# launch/carto_airio_ekf.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    params = "/root/moving_object_estimator_ws/airio_imu_odometry/config/params.yaml"

    airimu_root = "/root/moving_object_estimator_ws/airio_imu_odometry/external/AirIMU-for-Damvi"
    airimu_ckpt = "/root/moving_object_estimator_ws/airio_imu_odometry/config/airimu/best_model.ckpt"
    airimu_conf = "/root/moving_object_estimator_ws/airio_imu_odometry/config/airimu/codenet.conf"

    airio_root = "/root/moving_object_estimator_ws/airio_imu_odometry/external/Air-IO-for-Damvi"
    airio_ckpt = "/root/moving_object_estimator_ws/airio_imu_odometry/config/airio/best_model.ckpt"
    airio_conf = "/root/moving_object_estimator_ws/airio_imu_odometry/config/airio/motion_body_rot.conf"

    timming_logging_output_path = "/root/moving_object_estimator_ws/airio_imu_odometry/TL_output/"

    airio_node = Node(
        package="airio_imu_odometry",
        executable="airio_node",              
        name="airio_imu_odometry",
        output="screen",
        parameters=[
            params,
            {
                'use_sim_time': use_sim_time,
                "airimu_root": airimu_root,
                "airimu_ckpt": airimu_ckpt,
                "airimu_conf": airimu_conf,
                "airio_root": airio_root,
                "airio_ckpt": airio_ckpt,
                "airio_conf": airio_conf,
                "device": "cuda",
                "airimu_seqlen": 30,
                "airimu_stride": 1,
                "airio_model": "CodeNetMotion",
                "airio_seqlen": 10,
                "airio_interval": 5,
                "publish_rate": 40.0,  
                "timming_logging_mode": False,
                "timming_logging_outputpath": timming_logging_output_path,
                "use_odom_init": True,
                "use_odom_realign": True,
            }
        ]
    )

    ekf_node = Node(
        package='airio_imu_odometry',
        executable='carto_airio_ekf_node',
        name='carto_airio_ekf',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'odom_pub_rate': 50.0,
            'carto_Rp_diag': [0.05, 0.05, 0.10],                 # m^2
            'carto_Rth_diag': [(3.14159265/180)**2]*3,           # (1 deg)^2 in rad^2
            'chi2_thresh': 22.46,                                 # dof=6, p≈0.999
            'alpha_max': 50.0,
            'use_body_vel_update': True,
        }],
        remappings=[
            ('/airimu_imu_data', '/airimu_imu_data'),
            ('/odom_airio', '/odom_airio'),
            ('/odom', '/odom'),
            ('/odom_fused', '/odom_fused'),
        ]
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        airio_node,
        ekf_node,  
    ])
