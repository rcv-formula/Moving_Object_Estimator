#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import numpy as np
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
from airio_imu_odometry.ekf_wrapper import AirIOEKFWrapper, ImuSample, EkfParams

def _diag3_from_cov(cov_list):
        # cov_list: 9개 원소의 row-major 3x3
        if cov_list is None or len(cov_list) != 9:
            return None
        d0, d1, d2 = float(cov_list[0]), float(cov_list[4]), float(cov_list[8])
        # ROS 규약: -1 은 unknown
        if d0 < 0.0 or d1 < 0.0 or d2 < 0.0:
            return None
        return (d0, d1, d2)

def quat_to_np(q: Quaternion):
    return np.array([q.x, q.y, q.z, q.w], dtype=float)

def rot_from_quat(q: np.ndarray):
    x, y, z, w = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1 - 2*(yy+zz), 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),   1 - 2*(xx+zz),   2*(yz - wx)],
        [2*(xz - wy),   2*(yz + wx),     1 - 2*(xx+yy)]
    ], dtype=float)

class OdomFusionEkfNode(Node):
    """
    /airimu_imu_data  : 보정된 IMU (predict)
    /odom_airio       : AirIO 오도메트리(속도 + 불확실도) (update - velocity)
    /odom             : Cartographer 오도메트리(pose)   (update - pose, 게이팅/인플레이트)
    출력: /odom_fused
    """
    def __init__(self):
        super().__init__('odom_fusion_ekf')

        # ---- params ----
        self.declare_parameter("odom_pub_rate", 50.0)
        self.declare_parameter("carto_Rp_diag", [0.05, 0.05, 0.1])     # m^2
        self.declare_parameter("carto_Rth_diag",[ (np.deg2rad(1))**2 ]*3)  # rad^2
        self.declare_parameter("chi2_thresh", 22.46)   # dof=6, 0.999
        self.declare_parameter("alpha_max", 50.0)
        self.declare_parameter("use_body_vel_update", True)

        self.pub_rate = float(self.get_parameter("odom_pub_rate").value)
        self.dt_pub = 1.0/max(1e-6, self.pub_rate)
        self.last_pub_sec = 0.0

        Rp = np.array(self.get_parameter("carto_Rp_diag").value, dtype=float)
        Rth= np.array(self.get_parameter("carto_Rth_diag").value, dtype=float)
        self.Rp = np.diag(Rp)
        self.Rth= np.diag(Rth)
        self.chi2_thresh = float(self.get_parameter("chi2_thresh").value)
        self.alpha_max   = float(self.get_parameter("alpha_max").value)
        self.use_body_vel_update = bool(self.get_parameter("use_body_vel_update").value)

        # ---- EKF ----
        self.ekf = AirIOEKFWrapper(airio_root="", use_repo=False,
                                   params=EkfParams(gyro_noise=0.02, acc_noise=0.20))
        self.initialized = False

        # ---- subs/pubs ----
        self.sub_imu  = self.create_subscription(Imu,      "/airimu_imu_data", self.cb_imu,  qos_profile_sensor_data)
        self.sub_air  = self.create_subscription(Odometry, "/odom_airio",      self.cb_airio, 10)
        self.sub_carto= self.create_subscription(Odometry, "/odom",            self.cb_carto, 10)
        self.pub_odom = self.create_publisher(Odometry,    "/odom_fused",      10)

        # caches
        self.last_eta_v = np.array([0.05,0.05,0.05], float)  # AirIO 속도 불확실도 기본
        self.deadband_ms = 5.0


    # ---------- callbacks ----------
    def cb_imu(self, m: Imu):
        stamp = m.header.stamp.sec + m.header.stamp.nanosec*1e-9

        if not self.initialized:
            self.ekf.set_init_state({
                "pos":[0,0,0], "vel":[0,0,0], "rot":[0,0,0,1], "stamp":stamp
            })
            self.initialized = True

        try:
            gyro_var = _diag3_from_cov(m.angular_velocity_covariance)
            acc_var  = _diag3_from_cov(m.linear_acceleration_covariance)

            self.ekf.add_imu(
                ImuSample(
                    wx=float(m.angular_velocity.x),
                    wy=float(m.angular_velocity.y),
                    wz=float(m.angular_velocity.z),
                    ax=float(m.linear_acceleration.x),
                    ay=float(m.linear_acceleration.y),
                    az=float(m.linear_acceleration.z),
                    stamp=float(stamp)
                ),
                gyro_var=gyro_var,
                acc_var=acc_var,
            )
        except Exception as e:
            self.get_logger().warn(f"EKF predict failed: {e}")
            return

        self._EKF_publisher()

    def cb_airio(self, m: Odometry):
        try:
            v_w = np.array([m.twist.twist.linear.x,
                            m.twist.twist.linear.y,
                            m.twist.twist.linear.z], dtype=float)

            if self.use_body_vel_update:
                q = self.ekf.get_state()["rot"]
                Rwb = rot_from_quat(np.asarray(q, float))
                v_b = Rwb.T @ v_w

                if np.linalg.norm(v_b)*1000.0 < self.deadband_ms:
                    v_b[:] = 0.0
                Rm = np.diag(self.last_eta_v**2)  # 필요시 토픽으로 받아 반영
                self.ekf.update_velocity_body(tuple(v_b.tolist()), Rm)
            else:
                pass
        except Exception as e:
            self.get_logger().warn(f"AirIO velocity update failed: {e}")
            return
        self._EKF_publisher()

    def cb_carto(self, m: Odometry):
        try:
            p = np.array([m.pose.pose.position.x,
                          m.pose.pose.position.y,
                          m.pose.pose.position.z], dtype=float)
            q = quat_to_np(m.pose.pose.orientation)

            lam = self.ekf.update_pose_world_adaptive(
                p_meas=p, q_meas=q,
                R_p=self.Rp, R_theta=self.Rth,
                chi2_thresh=self.chi2_thresh, alpha_max=self.alpha_max
            )
            if lam > self.chi2_thresh:
                self.get_logger().warn(f"Carto pose gated/inflated. Mahalanobis={lam:.2f}")
        except Exception as e:
            self.get_logger().warn(f"Carto pose update failed: {e}")
            return
        self._EKF_publisher()

    # ---------- publish ----------
    def _EKF_publisher(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if (now - self.last_pub_sec) < self.dt_pub:
            return
        self.last_pub_sec = now

        s = self.ekf.get_state()
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = "map"
        odom.child_frame_id  = "base_link"
        odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = s["pos"].tolist()
        odom.pose.pose.orientation.x, odom.pose.pose.orientation.y, odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = s["rot"].tolist()
        odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z = s["vel"].tolist()
        self.pub_odom.publish(odom)

def main(args=None):
    rclpy.init(args=args)
    node = OdomFusionEkfNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
