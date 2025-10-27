# launch/ate_rte.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    # 1) 인자 선언 (기본값)
    args = [
        DeclareLaunchArgument('est_topic', default_value='/odom_airio'),
        DeclareLaunchArgument('gt_topic',  default_value='/odom'),
        DeclareLaunchArgument('assoc_max_dt', default_value='0.002'),
        DeclareLaunchArgument('window_sec',   default_value='20.0'),
        DeclareLaunchArgument('align_mode',   default_value='se2_xyyaw'),
        DeclareLaunchArgument('align_with_scale', default_value='false'),
        DeclareLaunchArgument('recompute_alignment_every', default_value='2.0'),
        DeclareLaunchArgument('rte_delta_sec', default_value='1.0'),
        DeclareLaunchArgument('publish_rate',  default_value='5.0'),
        DeclareLaunchArgument('output_csv',    default_value='./result.csv'),
        DeclareLaunchArgument('save_plots',    default_value='true'),
        DeclareLaunchArgument('eval_stride',    default_value='5'),
    ]

    # 2) Node에 LaunchConfiguration을 “타입 지정”해서 꽂기
    params = [{
        'est_topic': LaunchConfiguration('est_topic'),
        'gt_topic':  LaunchConfiguration('gt_topic'),
        'assoc_max_dt': ParameterValue(LaunchConfiguration('assoc_max_dt'), value_type=float),
        'window_sec':   ParameterValue(LaunchConfiguration('window_sec'), value_type=float),
        'align_mode':   LaunchConfiguration('align_mode'),
        'align_with_scale': ParameterValue(LaunchConfiguration('align_with_scale'), value_type=bool),
        'recompute_alignment_every': ParameterValue(LaunchConfiguration('recompute_alignment_every'), value_type=float),
        'rte_delta_sec': ParameterValue(LaunchConfiguration('rte_delta_sec'), value_type=float),
        'publish_rate':  ParameterValue(LaunchConfiguration('publish_rate'), value_type=float),
        'output_csv':    LaunchConfiguration('output_csv'),
        'save_plots':    ParameterValue(LaunchConfiguration('save_plots'), value_type=bool),
        'eval_stride':  ParameterValue(LaunchConfiguration('eval_stride'), value_type=int),

    }]

    node = Node(
        package='airio_imu_odometry',
        executable='ate_rte_evaluator',     # setup.py console_scripts 이름
        name='ate_rte_evaluator',
        output='screen',
        parameters=params,
    )

    return LaunchDescription(args + [node])
