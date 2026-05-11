#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry


class AirIoImuNode(Node):
    def __init__(self):
        super().__init__('airio_imu_with_odom_and_timer_mt')

        # === 콜백 그룹 (스레드 분산용) ===
        self.cbgroup_imu   = MutuallyExclusiveCallbackGroup()
        self.cbgroup_odom  = MutuallyExclusiveCallbackGroup()
        self.cbgroup_timer = MutuallyExclusiveCallbackGroup()

        # --- 파라미터 (타이머 주기) ---
        self.declare_parameter("publish_rate", 30.0)
        hz = float(self.get_parameter("publish_rate").get_parameter_value().double_value)
        period = 1.0 / max(1.0, hz)

        # --- 퍼블리셔 ---
        self.filtered_pub = self.create_publisher(Imu, '/airimu_imu_data', qos_profile_sensor_data)
        self.timer_pub    = self.create_publisher(Imu, '/airimu_imu_data_timer', qos_profile_sensor_data)

        # --- 구독 ---
        self.create_subscription(
            Imu, '/imu/data_raw', self.imu_callback,
            qos_profile_sensor_data, callback_group=self.cbgroup_imu
        )
        self.create_subscription(
            Odometry, '/odom', self.odom_callback,
            10, callback_group=self.cbgroup_odom
        )

        # --- 타이머 ---
        self.create_timer(period, self.on_timer, callback_group=self.cbgroup_timer)

        # --- 상태(가벼운 기록만) ---
        self.last_imu_stamp = None
        self.last_odom_stamp = None

        self.get_logger().info(
            f"MT node started: IMU direct republish + Odom callback + Timer({hz:.1f} Hz) [threads=3]"
        )

    # === 콜백들 ===
    def imu_callback(self, msg: Imu):
        """IMU 수신 시 바로 리퍼블리시 (고주파: 로깅/무거운 연산 금지)."""
        self.last_imu_stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)

        out = Imu()
        out.header = msg.header
        out.orientation = msg.orientation
        out.angular_velocity = msg.angular_velocity
        out.linear_acceleration = msg.linear_acceleration
        self.filtered_pub.publish(out)

    def odom_callback(self, msg: Odometry):
        """ODOM 수신: 최소 상태만 저장."""
        self.last_odom_stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)

    def on_timer(self):
        """주기 발행(테스트/하트비트용)."""
        hb = Imu()
        hb.header.stamp = self.get_clock().now().to_msg()
        hb.header.frame_id = "base_link"
        self.timer_pub.publish(hb)


def main(args=None):
    rclpy.init(args=args)
    node = AirIoImuNode()

    # === 멀티스레드 실행기 (imu / odom / timer 병렬 처리) ===
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
