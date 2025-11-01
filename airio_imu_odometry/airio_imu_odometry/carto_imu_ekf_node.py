#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import numpy as np
from math import sin, cos, sqrt
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion

# ==========================
# Quaternion / SO(3) helpers
# ==========================
def quat_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ], dtype=float)

def quat_normalize(q):
    n = np.linalg.norm(q)
    if n < 1e-12:  # fallback
        return np.array([0.0, 0.0, 0.0, 1.0], float)
    return q / n

def so3_exp(phi):
    """ small-angle axis-angle (phi in R^3) -> quat """
    th = np.linalg.norm(phi)
    if th < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], float)
    axis = phi / th
    s = sin(th/2.0)
    return np.array([axis[0]*s, axis[1]*s, axis[2]*s, cos(th/2.0)], float)

def so3_log(q):
    """ quat -> small-angle axis-angle (R^3) """
    q = quat_normalize(q)
    v = q[:3]; w = q[3]
    s = np.linalg.norm(v)
    if s < 1e-12:
        return np.zeros(3, float)
    axis = v / s
    th = 2.0*np.arctan2(s, w)
    return axis * th

def rot_from_quat(q):
    x, y, z, w = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1 - 2*(yy+zz), 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),   1 - 2*(xx+zz),   2*(yz - wx)],
        [2*(xz - wy),   2*(yz + wx),     1 - 2*(xx+yy)]
    ], dtype=float)

def quat_from_msg(qm: Quaternion):
    return np.array([qm.x, qm.y, qm.z, qm.w], float)

def quat_to_msg(q):
    out = Quaternion()
    out.x, out.y, out.z, out.w = q.tolist()
    return out

def ang_diff_deg(q_meas, q_pred):
    # 각도차(deg) from quaternion dot
    dot = abs(float(np.dot(q_meas, q_pred)))
    dot = max(min(dot, 1.0), -1.0)
    return np.degrees(2*np.arccos(dot))

# ==========================
# Odometry covariance helper
# ==========================
def cov6_from_odom(msg: Odometry):
    C = np.array(msg.pose.covariance, dtype=float)
    if C.size != 36:
        return None
    C = C.reshape(6, 6)
    diag = np.array([C[0,0], C[1,1], C[2,2], C[3,3], C[4,4], C[5,5]], float)
    # -1 or <= ~0 는 unknown 으로 간주
    if np.any(diag <= 1e-12):
        return None
    C = 0.5*(C + C.T)
    C += 1e-12*np.eye(6)
    return C

def diag3_from_cov9(lst):
    if lst is None or len(lst) != 9:
        return None
    d0, d1, d2 = float(lst[0]), float(lst[4]), float(lst[8])
    if d0 < 0.0 or d1 < 0.0 or d2 < 0.0:  # -1 처리
        return None
    return np.array([d0, d1, d2], float)

# ==========================
# Simple 9-state EKF: x=[p(3), v(3), theta(3)]
# ==========================
class SimpleImuPoseEKF:
    def __init__(self, gyro_noise=0.02, acc_noise=0.20, g=9.80665):
        # nominal state (kept separately): position, velocity, orientation(quaternion)
        self.p = np.zeros(3, float)
        self.v = np.zeros(3, float)
        self.q = np.array([0.0, 0.0, 0.0, 1.0], float)  # world<-body
        # error-state covariance (9x9) for [δp, δv, δθ]
        self.P = np.eye(9, dtype=float) * 1e-3
        # process noise (tuned as random-walk-ish)
        self.qp = 1e-4
        self.qv = acc_noise**2
        self.qth = gyro_noise**2
        self.g = np.array([0.0, 0.0, -g], float)
        self.last_t = None

    def predict(self, imu: Imu, t: float):
        if self.last_t is None:
            self.last_t = t
            return
        dt = max(1e-4, min(0.05, t - self.last_t))
        self.last_t = t

        # 1) attitude propagate: q <- q ⊗ Exp(ω*dt)
        wm = np.array([imu.angular_velocity.x,
                       imu.angular_velocity.y,
                       imu.angular_velocity.z], float)
        dq = so3_exp(wm * dt)
        self.q = quat_normalize(quat_multiply(self.q, dq))
        R = rot_from_quat(self.q)

        # 2) specific force -> world accel (IMU a = proper accel)
        am = np.array([imu.linear_acceleration.x,
                       imu.linear_acceleration.y,
                       imu.linear_acceleration.z], float)
        a_w = R @ am + self.g  # world accel

        # 3) velocity & position
        self.v = self.v + a_w * dt
        self.p = self.p + self.v * dt + 0.5 * a_w * (dt**2)

        # ---- covariance propagate (simple block model) ----
        I3 = np.eye(3)
        Phi = np.block([
            [I3, dt*I3, np.zeros((3,3))],
            [np.zeros((3,3)), I3, np.zeros((3,3))],
            [np.zeros((3,3)), np.zeros((3,3)), I3]
        ])
        Q = np.block([
            [self.qp*np.eye(3)*dt, np.zeros((3,3)), np.zeros((3,3))],
            [np.zeros((3,3)), self.qv*np.eye(3)*dt, np.zeros((3,3))],
            [np.zeros((3,3)), np.zeros((3,3)), self.qth*np.eye(3)*dt]
        ])
        self.P = Phi @ self.P @ Phi.T + Q

    def update_pose_world_adaptive(self, p_meas, q_meas, Rp, Rth,
                                   chi2_thresh=22.46, alpha_max=50.0,
                                   skip_thresh=80.0):
        """
        p_meas: (3,) 위치 (world)
        q_meas: (4,) 자세 (world<-body)
        Rp: (3,3), Rth: (3,3)  (θ는 소각도 공분산)
        반환: Mahalanobis distance (λ)
        """
        p_meas = np.asarray(p_meas, float)
        q_meas = quat_normalize(np.asarray(q_meas, float))

        # innovation
        dp = p_meas - self.p
        q_err = quat_multiply(q_meas, np.array([-self.q[0], -self.q[1], -self.q[2], self.q[3]]))
        dth = so3_log(q_err)  # small angle

        z = np.hstack([dp, dth])  # 6x1
        H = np.block([
            [np.eye(3), np.zeros((3,3)), np.zeros((3,3))],
            [np.zeros((3,3)), np.zeros((3,3)), np.eye(3)]
        ])  # 6x9
        Rm = np.block([
            [Rp, np.zeros((3,3))],
            [np.zeros((3,3)), Rth]
        ])

        S = H @ self.P @ H.T + Rm
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # 수치 안전장치
            S = S + 1e-9*np.eye(6)
            S_inv = np.linalg.inv(S)

        lam = float(z.T @ S_inv @ z)  # NIS
        if lam > skip_thresh:
            # 너무 동떨어진 측정 → 스킵
            return lam

        # 게이팅/인플레이트
        if lam > chi2_thresh:
            # 측정 공분산 인플레이트
            infl = min(alpha_max, lam/chi2_thresh)
            Rm = np.block([
                [Rp*infl, np.zeros((3,3))],
                [np.zeros((3,3)), Rth*infl]
            ])
            S = H @ self.P @ H.T + Rm
            S_inv = np.linalg.inv(S)

        K = self.P @ H.T @ S_inv  # 9x6
        dx = K @ z                 # 9x1 (δp, δv, δθ)
        self.P = (np.eye(9) - K @ H) @ self.P

        # apply correction
        self.p += dx[0:3]
        self.v += dx[3:6]
        dth = dx[6:9]
        self.q = quat_normalize(quat_multiply(self.q, so3_exp(dth)))

        return lam

    def get_state(self):
        return {"pos": self.p.copy(),
                "vel": self.v.copy(),
                "rot": self.q.copy()}

    def set_init(self, p, q, t0=None):
        self.p = np.asarray(p, float)
        self.v = np.zeros(3, float)
        self.q = quat_normalize(np.asarray(q, float))
        self.P = np.eye(9, dtype=float) * 1e-2
        self.last_t = t0

# ==========================
# ROS2 Node
# ==========================
class CartoIMUEKFNode(Node):
    """
    /imu    : 일반 IMU (predict)
    /odom   : Cartographer 오돔 (update - pose)
    출력    : /odom_fused
    """
    def __init__(self):
        super().__init__('carto_imu_ekf')

        # ---- parameters ----
        self.declare_parameter("imu_topic", "/imu")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("odom_pub_topic", "/odom_fused")
        self.declare_parameter("odom_pub_rate", 50.0)

        # 기본 R (fallback)
        self.declare_parameter("carto_Rp_diag", [0.05, 0.05, 0.10])   # m^2
        self.declare_parameter("carto_Rth_deg", [1.0, 1.0, 1.0])      # deg (converted to rad^2)
        self.declare_parameter("use_carto_cov_from_msg", True)

        # 게이팅/웜업
        self.declare_parameter("chi2_thresh", 22.46)   # dof=6, 0.999
        self.declare_parameter("alpha_max", 50.0)
        self.declare_parameter("skip_thresh", 80.0)
        self.declare_parameter("warmup_sec", 1.0)
        self.declare_parameter("boot_skip", 2)
        self.declare_parameter("cooldown_sec", 0.2)

        # process noises
        self.declare_parameter("gyro_noise", 0.02)     # rad/s stdev (approx)
        self.declare_parameter("acc_noise", 0.20)      # m/s^2 stdev

        # ---- read params ----
        self.imu_topic   = self.get_parameter("imu_topic").value
        self.odom_topic  = self.get_parameter("odom_topic").value
        self.odom_pub_topic = self.get_parameter("odom_pub_topic").value
        self.pub_rate = float(self.get_parameter("odom_pub_rate").value)
        self.dt_pub = 1.0/max(1e-6, self.pub_rate)

        Rp_diag = np.array(self.get_parameter("carto_Rp_diag").value, float)
        Rth_deg = np.array(self.get_parameter("carto_Rth_deg").value, float)
        self.Rp_default  = np.diag(Rp_diag)
        self.Rth_default = np.diag(np.deg2rad(Rth_deg)**2)

        self.use_carto_cov = bool(self.get_parameter("use_carto_cov_from_msg").value)
        self.chi2_thresh = float(self.get_parameter("chi2_thresh").value)
        self.alpha_max   = float(self.get_parameter("alpha_max").value)
        self.skip_thresh = float(self.get_parameter("skip_thresh").value)
        self.warmup_sec  = float(self.get_parameter("warmup_sec").value)
        self.boot_skip   = int(self.get_parameter("boot_skip").value)
        self.cooldown_sec= float(self.get_parameter("cooldown_sec").value)

        gyro_n = float(self.get_parameter("gyro_noise").value)
        acc_n  = float(self.get_parameter("acc_noise").value)

        # ---- filter ----
        self.ekf = SimpleImuPoseEKF(gyro_noise=gyro_n, acc_noise=acc_n)
        self.initialized = False
        self.start_sec = None
        self.carto_seen = 0
        self.skip_until_sec = 0.0
        self.last_pub_sec = 0.0

        # ---- subs/pubs ----
        self.sub_imu  = self.create_subscription(Imu, self.imu_topic, self.cb_imu, qos_profile_sensor_data)
        self.sub_odom = self.create_subscription(Odometry, self.odom_topic, self.cb_odom, 10)
        self.pub_odom = self.create_publisher(Odometry, self.odom_pub_topic, 10)

        self.get_logger().info(f"CartoIMUEKFNode start: imu={self.imu_topic}, carto={self.odom_topic}, out={self.odom_pub_topic}")

    # ---------- callbacks ----------
    def cb_imu(self, m: Imu):
        t = m.header.stamp.sec + m.header.stamp.nanosec*1e-9
        if self.start_sec is None:
            self.start_sec = t
        try:
            self.ekf.predict(m, t)
        except Exception as e:
            self.get_logger().warn(f"EKF predict failed: {e}")
        self._publish_if_due()

    def cb_odom(self, m: Odometry):
        t_now = self.get_clock().now().nanoseconds * 1e-9
        p = np.array([m.pose.pose.position.x,
                      m.pose.pose.position.y,
                      m.pose.pose.position.z], float)
        q = quat_from_msg(m.pose.pose.orientation)

        # 초기화: Cartographer 첫 pose로
        if not self.initialized:
            self.ekf.set_init(p, q, t0=m.header.stamp.sec + m.header.stamp.nanosec*1e-9)
            self.initialized = True
            self.carto_seen = 0
            self.get_logger().info("EKF initialized from Cartographer /odom.")
            return

        # 부트 스킵
        self.carto_seen += 1
        if self.carto_seen <= self.boot_skip:
            return

        # 쿨다운
        if t_now < self.skip_until_sec:
            return

        # R 설정 (웜업 시 인플레이트)
        Rp  = self.Rp_default.copy()
        Rth = self.Rth_default.copy()
        chi2 = self.chi2_thresh
        amax = self.alpha_max

        elapsed = (t_now - self.start_sec) if (self.start_sec is not None) else 999.0
        if elapsed < self.warmup_sec:
            Rp  *= 5.0
            Rth *= 5.0
            chi2 = max(chi2, 40.0)
            amax = max(amax, 100.0)

        # 메시지 공분산 사용
        if self.use_carto_cov:
            C6 = cov6_from_odom(m)
            if C6 is None:
                self.get_logger().warn("Carto covariance invalid/zero → using default Rp/Rθ")
            else:
                Rp  = C6[:3,:3]
                Rth = C6[3:,3:]

        try:
            lam = self.ekf.update_pose_world_adaptive(
                p_meas=p, q_meas=q,
                Rp=Rp, Rth=Rth,
                chi2_thresh=chi2, alpha_max=amax,
                skip_thresh=self.skip_thresh
            )
            # 로깅
            s = self.ekf.get_state()
            rp = float(np.linalg.norm(p - s["pos"]))
            rth = ang_diff_deg(q, s["rot"])
            if lam is not None:
                self.get_logger().info(
                    f"NIS λ={lam:.2f} | rp={rp:.3f} m, rθ={rth:.2f} deg | "
                    f"Rp_diag={np.diag(Rp)} Rθ_diag={np.diag(Rth)}"
                )
                if lam > self.skip_thresh:
                    self.skip_until_sec = t_now + self.cooldown_sec
                    self.get_logger().warn(f"Carto pose skipped: λ={lam:.1f} (cooldown {self.cooldown_sec:.2f}s)")
                elif lam > chi2:
                    self.get_logger().warn(f"Carto pose gated/inflated. Mahalanobis={lam:.2f}")
        except Exception as e:
            self.get_logger().warn(f"Carto pose update failed: {e}")

        self._publish_if_due()

    # ---------- publisher ----------
    def _publish_if_due(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if (now - self.last_pub_sec) < self.dt_pub:
            return
        self.last_pub_sec = now

        s = self.ekf.get_state()
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = "map"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = s["pos"].tolist()
        odom.pose.pose.orientation = quat_to_msg(s["rot"])
        odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z = s["vel"].tolist()

        self.pub_odom.publish(odom)

# ==========================
def main(args=None):
    rclpy.init(args=args)
    node = CartoIMUEKFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
