from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    params = "/root/moving_object_estimator_ws/airio_imu_odometry/config/params.yaml"

    airimu_root = "/root/moving_object_estimator_ws/airio_imu_odometry/external/AirIMU-for-Damvi"
    airimu_ckpt = "/root/moving_object_estimator_ws/airio_imu_odometry/config/airimu/best_model.ckpt"
    airimu_conf = "/root/moving_object_estimator_ws/airio_imu_odometry/config/airimu/codenet.conf"

    airio_root = "/root/moving_object_estimator_ws/airio_imu_odometry/external/Air-IO-for-Damvi"
    airio_ckpt = "/root/moving_object_estimator_ws/airio_imu_odometry/config/airio/best_model.ckpt"
    airio_conf = "/root/moving_object_estimator_ws/airio_imu_odometry/config/airio/motion_body_rot.conf"

    timming_logging_output_path = "/root/moving_object_estimator_ws/airio_imu_odometry/TL_output/"
    return LaunchDescription([
        Node(
            package="airio_imu_odometry",
            executable="airio_node",
            name="airio_imu_odometry",
            output="screen",
            parameters=[ 
                params, {
                "airimu_root": airimu_root,
                "airimu_ckpt": airimu_ckpt,
                "airimu_conf": airimu_conf,
                "airio_root": airio_root,
                "airio_ckpt": airio_ckpt,
                "airio_conf": airio_conf,
                "device": "cuda",
                "airimu_seqlen": 5,
                "airimu_stride": 1,
                "airio_model": "CodeNetMotion",
                "airio_seqlen": 10,                     
                "airio_interval": 5,                     
                "publish_rate": 40.0,
                "timming_logging_mode": False,
                "timming_logging_outputpath" : timming_logging_output_path,
            }]
        )
    ])
