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

# ----------------- helpers -----------------
def _ang_deg(q_meas, q_pred):
    # 각도차(deg)
    from math import acos, degrees
    import numpy as np
    qm = np.asarray(q_meas, float); qp = np.asarray(q_pred, float)
    dot = abs(np.dot(qm, qp))
    dot = max(min(dot, 1.0), -1.0)
    return degrees(2*acos(dot))

def _cov6_from_odom(msg: Odometry):
    """Odometry.pose.covariance(36개)를 6x6으로. 0/음수/비정상은 None."""
    C = np.array(msg.pose.covariance, dtype=float)
    if C.size != 36:
        return None
    C = C.reshape(6, 6)

    # 대각 유효성: 너무 작거나(≈0) 음수면 무효
    diag = np.array([C[0,0], C[1,1], C[2,2], C[3,3], C[4,4], C[5,5]], dtype=float)
    if np.any(diag <= 1e-12):   # ★ 0 포함 차단
        return None

    # 수치 안정화(대칭화 + 바닥값)
    C = 0.5*(C + C.T)
    C += 1e-12*np.eye(6)
    return C

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

def _cov6_from_odom(msg: Odometry):
    """Odometry.pose.covariance(36개)를 6x6으로 반환. 실패 시 None."""
    C = np.array(msg.pose.covariance, dtype=float)
    if C.shape[0] != 36:
        return None
    C = C.reshape(6, 6)
    # 유효성 검사(대각 -1이면 unknown으로 보는 경우가 있음)
    d = np.array([C[0,0], C[1,1], C[2,2], C[3,3], C[4,4], C[5,5]])
    if np.any(d < 0.0):
        return None
    # 수치 안정화
    return C + 1e-12*np.eye(6)

# ----------------- node -----------------
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
        self.declare_parameter("carto_Rp_diag", [0.05, 0.05, 0.1])           # m^2
        self.declare_parameter("carto_Rth_diag", [(np.deg2rad(1))**2]*3)     # rad^2
        self.declare_parameter("chi2_thresh", 22.46)     # dof=6, 0.999
        self.declare_parameter("alpha_max", 50.0)
        self.declare_parameter("use_body_vel_update", True)

        # ★ NEW: 초기 정렬/웜업/스킵 관련
        self.declare_parameter("warmup_sec", 1.0)        # 웜업 동안 R 인플레이트/임계 완화
        self.declare_parameter("carto_boot_skip", 2)     # Carto 초기 N프레임 업데이트 스킵
        self.declare_parameter("skip_thresh", 80.0)      # 이 값 넘으면 pose 업데이트 스킵
        self.declare_parameter("cooldown_sec", 0.2)      # 스킵 후 쿨다운
        self.declare_parameter("use_carto_cov_from_msg", True)  # Odometry covariance 사용

        self.pub_rate = float(self.get_parameter("odom_pub_rate").value)
        self.dt_pub = 1.0/max(1e-6, self.pub_rate)
        self.last_pub_sec = 0.0

        Rp_diag = np.array(self.get_parameter("carto_Rp_diag").value, dtype=float)
        Rth_diag= np.array(self.get_parameter("carto_Rth_diag").value, dtype=float)
        self.Rp = np.diag(Rp_diag)
        self.Rth= np.diag(Rth_diag)
        self.chi2_thresh = float(self.get_parameter("chi2_thresh").value)
        self.alpha_max   = float(self.get_parameter("alpha_max").value)
        self.use_body_vel_update = bool(self.get_parameter("use_body_vel_update").value)

        # ★ NEW: 초기/웜업/스킵 변수
        self.warmup_sec   = float(self.get_parameter("warmup_sec").value)
        self.carto_boot_skip = int(self.get_parameter("carto_boot_skip").value)
        self.skip_thresh  = float(self.get_parameter("skip_thresh").value)
        self.cooldown_sec = float(self.get_parameter("cooldown_sec").value)
        self.use_carto_cov_from_msg = bool(self.get_parameter("use_carto_cov_from_msg").value)

        # ---- EKF ----
        self.ekf = AirIOEKFWrapper(airio_root="", use_repo=False,
                                   params=EkfParams(gyro_noise=0.02, acc_noise=0.20))
        self.initialized = False            # ★ IMU에서 초기화하지 않고 Carto로 초기화할 것
        self.start_sec   = None             # ★ 시작 시각(웜업 계산용)

        # ---- subs/pubs ----
        self.sub_imu  = self.create_subscription(Imu,      "/airimu_imu_data", self.cb_imu,  10)
        self.sub_air  = self.create_subscription(Odometry, "/odom_airio",      self.cb_airio, 10)
        self.sub_carto= self.create_subscription(Odometry, "/odom",            self.cb_carto, 10)
        self.pub_odom = self.create_publisher(Odometry,    "/odom_fused",      10)

        # caches
        self.last_eta_v = np.array([0.05,0.05,0.05], float)  # AirIO 속도 불확실도 기본
        self.deadband_ms = 5.0

        # ★ NEW: Carto 초기 프레임 스킵/쿨다운 상태
        self.carto_seen = 0
        self.skip_until_sec = 0.0

    # ---------- callbacks ----------
    def cb_imu(self, m: Imu):
        stamp = m.header.stamp.sec + m.header.stamp.nanosec*1e-9

        # ★ NEW: 시작 시각 기록(웜업 기준)
        if self.start_sec is None:
            self.start_sec = stamp

        # ★ 변경: IMU에서 초기화하지 않음. Carto에서 초기화.
        if not self.initialized:
            # Carto 들어오기 전까지는 predict만 누적(단, 내부 last_t가 None이면 _propagate가 건너뛴다)
            # 원하면 여기서 아주 느슨한 초기 상태를 줄 수도 있음.
            pass

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

            if self.use_body_vel_update and self.initialized:
                q = self.ekf.get_state()["rot"]
                Rwb = rot_from_quat(np.asarray(q, float))
                v_b = Rwb.T @ v_w

                if np.linalg.norm(v_b)*1000.0 < self.deadband_ms:
                    v_b[:] = 0.0
                Rm = np.diag(self.last_eta_v**2)  # 필요시 토픽으로 받아 반영
                self.ekf.update_velocity_body(tuple(v_b.tolist()), Rm)
        except Exception as e:
            self.get_logger().warn(f"AirIO velocity update failed: {e}")
            return
        self._EKF_publisher()

    def cb_carto(self, m: Odometry):
        t = m.header.stamp.sec + m.header.stamp.nanosec*1e-9
        p = np.array([m.pose.pose.position.x,
                      m.pose.pose.position.y,
                      m.pose.pose.position.z], dtype=float)
        q = quat_to_np(m.pose.pose.orientation)

        # --- 초기화 ---
        if not self.initialized:
            self.ekf.set_init_state({"pos": p.tolist(),
                                     "vel": [0.0, 0.0, 0.0],
                                     "rot": q.tolist(),
                                     "stamp": t})
            # 초기 P 약간 키움
            self.ekf.P[0:3,0:3] *= 10.0
            self.ekf.P[6:9,6:9] *= 10.0
            self.initialized = True
            self.carto_seen = 0
            self.get_logger().info("EKF initialized from first Carto pose")
            return

        # --- 초기 N프레임 스킵 ---
        self.carto_seen += 1
        if self.carto_seen <= self.carto_boot_skip:
            return

        # --- 쿨다운 체크 ---
        now = self.get_clock().now().nanoseconds*1e-9
        if now < self.skip_until_sec:
            return

        # --- 기본 R 세팅 ---
        Rp  = self.Rp.copy()
        Rth = self.Rth.copy()
        chi2 = self.chi2_thresh
        amax = self.alpha_max

        # 웜업 동안 인플레이트
        elapsed = (now - self.start_sec) if (self.start_sec is not None) else 999.0
        if elapsed < self.warmup_sec:
            Rp  *= 5.0
            Rth *= 5.0
            chi2 = max(chi2, 40.0)
            amax = max(amax, 100.0)

        # --- 메시지 공분산 사용: 0이면 무시 ---
        if self.use_carto_cov_from_msg:
            C6 = _cov6_from_odom(m)  # ★ 위에서 강화한 함수
            if C6 is None:
                
                # 모두 0이거나 비정상 → 디폴트 사용
                self.get_logger().warn("Carto covariance invalid/zero → using default Rp/Rθ")
            else:
                Rp  = C6[:3,:3]
                Rth = C6[3:,3:]

        Rp  = np.diag([0.05, 0.05, 0.10])               
        Rth = np.diag([np.deg2rad(1)**2]*3)             
        try:
            lam = self.ekf.update_pose_world_adaptive(
                p_meas=p, q_meas=q,
                R_p=Rp, R_theta=Rth,
                chi2_thresh=chi2, alpha_max=amax,
                skip_thresh=self.skip_thresh
            )
            # lam 계산 직후:
            s = self.ekf.get_state()
            pos_pred = np.array(s["pos"])
            rot_pred = np.array(s["rot"])
            rp = np.linalg.norm(p - pos_pred)
            rth_deg = _ang_deg(q, rot_pred)

            self.get_logger().info(
                f"NIS λ={lam:.2f} | rp={rp:.3f} m, rθ={rth_deg:.2f} deg | "
                f"Rp_diag={np.diag(Rp)} Rθ_diag={np.diag(Rth)} | "
            )

            if lam > self.skip_thresh:
                self.skip_until_sec = now + self.cooldown_sec
                self.get_logger().warn(
                    f"Carto pose skipped: λ={lam:.1f} (cooldown {self.cooldown_sec:.2f}s)"
                )
            elif lam > chi2:
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
