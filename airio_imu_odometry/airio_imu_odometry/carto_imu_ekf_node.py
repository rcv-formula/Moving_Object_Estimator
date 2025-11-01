#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import numpy as np
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion

# ==========================
# SO(3) / Quaternion helpers
# ==========================
def quat_multiply(q1, q2):
    x1, y1, z1, w1 = q1; x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ], float)

def quat_normalize(q):
    n = np.linalg.norm(q)
    return q/n if n > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0], float)

def so3_exp(phi):
    th = np.linalg.norm(phi)
    if th < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], float)
    axis = phi / th
    s = np.sin(th/2.0)
    return np.array([axis[0]*s, axis[1]*s, axis[2]*s, np.cos(th/2.0)], float)

def so3_log(q):
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
    ], float)

def quat_from_msg(qm: Quaternion):
    return np.array([qm.x, qm.y, qm.z, qm.w], float)

def quat_to_msg(q):
    m = Quaternion()
    m.x, m.y, m.z, m.w = q.tolist()
    return m

def ang_diff_deg(q_meas, q_pred):
    dot = abs(float(np.dot(q_meas, q_pred)))
    dot = max(min(dot, 1.0), -1.0)
    return np.degrees(2*np.arccos(dot))

# ==========================
# Odometry covariance helper
# ==========================
def cov6_from_odom(msg: Odometry):
    C = np.array(msg.pose.covariance, float)
    if C.size != 36:
        return None
    C = C.reshape(6, 6)
    # 0 또는 음수(-1 등)는 invalid로 간주
    if np.any(np.diag(C) <= 1e-12):
        return None
    C = 0.5*(C + C.T) + 1e-12*np.eye(6)
    return C

# ==========================
# Simple 9-state error EKF
# x = [δp, δv, δθ], nominal = {p, v, q}
# ==========================
class SimpleImuPoseEKF:
    def __init__(self, gyro_noise=0.02, acc_noise=0.20, g=9.80665):
        self.p = np.zeros(3, float)
        self.v = np.zeros(3, float)
        self.q = np.array([0.0, 0.0, 0.0, 1.0], float)  # world<-body
        # 초기 P를 넉넉히: 측정을 잘 받아들이도록 함
        self.P = np.eye(9, dtype=float) * 1e-2
        # 과정 잡음(튜닝)
        self.qp  = 1e-3                 # position random walk
        self.qv  = float(acc_noise**2)  # acceleration drive
        self.qth = float(gyro_noise**2) # angle random walk
        self.g = np.array([0.0, 0.0, 0], float)
        self.last_t = None

    def predict(self, imu: Imu, t: float, q_scale=1.0, imu_has_gravity=False):
        if self.last_t is None:
            self.last_t = t
            return
        dt = max(1e-4, min(0.05, t - self.last_t))
        self.last_t = t

        # 1) orientation integrate
        wm = np.array([imu.angular_velocity.x,
                       imu.angular_velocity.y,
                       imu.angular_velocity.z], float)
        dq = so3_exp(wm * dt)
        self.q = quat_normalize(quat_multiply(self.q, dq))
        R = rot_from_quat(self.q)

        # 2) specific force to world accel
        am = np.array([imu.linear_acceleration.x,
                       imu.linear_acceleration.y,
                       imu.linear_acceleration.z], float)
        if imu_has_gravity:
            a_w = R @ am + self.g    # raw IMU(-g 포함)
        else:
            a_w = R @ am             # 이미 중력 보정된 IMU

        # 3) integrate velocity & position
        self.v = self.v + a_w * dt
        self.p = self.p + self.v * dt + 0.5 * a_w * (dt**2)

        # 4) covariance propagate
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
        ]) * float(q_scale)  # MD 모드에서 인플레이트
        self.P = Phi @ self.P @ Phi.T + Q

    def update_pose(self, p_meas, q_meas, Rp, Rth, H_mask=None,
                    chi2_thresh=30.0, alpha_max=100.0, skip_thresh=-1.0):
        p_meas = np.asarray(p_meas, float)
        q_meas = quat_normalize(np.asarray(q_meas, float))

        # innovation
        dp = p_meas - self.p
        q_err = quat_multiply(q_meas, np.array([-self.q[0], -self.q[1], -self.q[2], self.q[3]]))
        dth = so3_log(q_err)

        z = np.hstack([dp, dth])  # 6,
        H = np.block([
            [np.eye(3), np.zeros((3,3)), np.zeros((3,3))],
            [np.zeros((3,3)), np.zeros((3,3)), np.eye(3)]
        ])  # 6x9
        Rm = np.block([
            [Rp, np.zeros((3,3))],
            [np.zeros((3,3)), Rth]
        ])

        # 선택적 2D 업데이트(특정 축 무시)
        if H_mask is not None:
            mask = np.array(H_mask, dtype=bool)
            H = H[mask]
            z = z[mask]
            Rm = Rm[np.ix_(mask, mask)]

        S = H @ self.P @ H.T + Rm
        S = 0.5*(S + S.T) + 1e-9*np.eye(S.shape[0])
        S_inv = np.linalg.inv(S)

        lam = float(z.T @ S_inv @ z) if z.size > 0 else 0.0
        if skip_thresh > 0 and lam > skip_thresh:
            return lam  # 스킵

        # 게이팅/인플레이트
        if lam > chi2_thresh and z.size > 0:
            infl = min(alpha_max, max(1.0, lam/chi2_thresh))
            Rm = Rm * infl
            S = H @ self.P @ H.T + Rm
            S = 0.5*(S + S.T) + 1e-9*np.eye(S.shape[0])
            S_inv = np.linalg.inv(S)

        if z.size > 0:
            K = self.P @ H.T @ S_inv
            dx = K @ z
            self.P = (np.eye(9) - K @ H) @ self.P
            self.p += dx[0:3]
            self.v += dx[3:6]
            self.q = quat_normalize(quat_multiply(self.q, so3_exp(dx[6:9])))

        return lam

    def hard_snap_to_measurement(self, p_meas, q_meas, vel_damping=0.5):
        self.p = np.asarray(p_meas, float)
        self.q = quat_normalize(np.asarray(q_meas, float))
        self.v *= float(vel_damping)
        # 스냅 후 P를 약간 키워 다음 업데이트 수용성 ↑
        self.P = self.P + np.diag([
            0.1,0.1,0.1,
            0.1,0.1,0.1,
            np.deg2rad(5)**2, np.deg2rad(5)**2, np.deg2rad(5)**2
        ])

    def soft_blend_toward_measurement(self, p_meas, q_meas, beta=0.6):
        p_meas = np.asarray(p_meas, float)
        q_meas = quat_normalize(np.asarray(q_meas, float))
        self.p = (1.0 - beta)*self.p + beta*p_meas
        q_err = quat_multiply(q_meas, np.array([-self.q[0], -self.q[1], -self.q[2], self.q[3]]))
        self.q = quat_normalize(quat_multiply(self.q, so3_exp(beta * so3_log(q_err))))

    def get_state(self):
        return {"pos": self.p.copy(), "vel": self.v.copy(), "rot": self.q.copy()}

# ==========================
# ROS2 Node
# ==========================
class CartoIMUEKFNode(Node):
    """
    /imu   : 일반 IMU (predict)
    /odom  : Cartographer 오돔 (update - pose)
    출력    : /odom_fused (map←base_link)
    """
    def __init__(self):
        super().__init__('carto_imu_ekf')

        # --- Topics / Rates ---
        self.declare_parameter("imu_topic", "/imu")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("odom_pub_topic", "/odom_fused")
        self.declare_parameter("odom_pub_rate", 50.0)

        # --- Gravity handling ---
        self.declare_parameter("imu_has_gravity", False)  # IMU a가 -g 포함시 True

        # --- MD(Measurement-Dominant) mode when IMU timeout ---
        self.declare_parameter("imu_timeout_sec", 0.25)
        self.declare_parameter("md_q_scale", 50.0)

        # --- Measurement noise fallback (std) ---
        self.declare_parameter("default_pos_std_m", [0.15, 0.15, 0.50])
        self.declare_parameter("default_rpy_std_deg", [1.5, 1.5, 1.5])
        self.declare_parameter("use_carto_cov_from_msg", True)

        # --- Gating / Warmup / Skips ---
        self.declare_parameter("chi2_thresh", 30.0)
        self.declare_parameter("alpha_max", 100.0)
        self.declare_parameter("skip_thresh", -1.0)   # -1이면 스킵 사용 안함
        self.declare_parameter("warmup_sec", 1.5)
        self.declare_parameter("boot_skip", 1)
        self.declare_parameter("cooldown_sec", 0.0)

        # --- 2D mode ---
        self.declare_parameter("use_2d_mode", True)

        # --- Process noises (approx stdev) ---
        self.declare_parameter("gyro_noise", 0.02)
        self.declare_parameter("acc_noise", 0.20)

        # --- Reset thresholds for sudden motion ---
        self.declare_parameter("pos_reset_thresh_m", 0.8)
        self.declare_parameter("yaw_reset_thresh_deg", 15.0)
        self.declare_parameter("nis_reset_thresh", 200.0)
        self.declare_parameter("snap_beta", 0.6)
        self.declare_parameter("vel_damping_on_reset", 0.5)

        # --- Read params ---
        self.imu_topic = self.get_parameter("imu_topic").value
        self.odom_topic = self.get_parameter("odom_topic").value
        self.odom_pub_topic = self.get_parameter("odom_pub_topic").value
        self.pub_rate = float(self.get_parameter("odom_pub_rate").value)
        self.dt_pub = 1.0 / max(1e-6, self.pub_rate)

        self.imu_has_gravity = bool(self.get_parameter("imu_has_gravity").value)
        self.imu_timeout_sec = float(self.get_parameter("imu_timeout_sec").value)
        self.md_q_scale = float(self.get_parameter("md_q_scale").value)

        pos_std = np.array(self.get_parameter("default_pos_std_m").value, float)
        rpy_std_deg = np.array(self.get_parameter("default_rpy_std_deg").value, float)
        self.Rp_default = np.diag(pos_std**2)
        self.Rth_default = np.diag(np.deg2rad(rpy_std_deg)**2)
        self.use_carto_cov = bool(self.get_parameter("use_carto_cov_from_msg").value)

        self.chi2_thresh = float(self.get_parameter("chi2_thresh").value)
        self.alpha_max = float(self.get_parameter("alpha_max").value)
        self.skip_thresh = float(self.get_parameter("skip_thresh").value)
        self.warmup_sec = float(self.get_parameter("warmup_sec").value)
        self.boot_skip = int(self.get_parameter("boot_skip").value)
        self.cooldown_sec = float(self.get_parameter("cooldown_sec").value)

        self.use_2d_mode = bool(self.get_parameter("use_2d_mode").value)
        gyro_n = float(self.get_parameter("gyro_noise").value)
        acc_n  = float(self.get_parameter("acc_noise").value)

        self.pos_reset_thresh_m = float(self.get_parameter("pos_reset_thresh_m").value)
        self.yaw_reset_thresh_deg = float(self.get_parameter("yaw_reset_thresh_deg").value)
        self.nis_reset_thresh = float(self.get_parameter("nis_reset_thresh").value)
        self.snap_beta = float(self.get_parameter("snap_beta").value)
        self.vel_damping_on_reset = float(self.get_parameter("vel_damping_on_reset").value)

        # --- Filter / State ---
        self.ekf = SimpleImuPoseEKF(gyro_noise=gyro_n, acc_noise=acc_n)
        self.initialized = False
        self.start_sec = self.get_clock().now().nanoseconds * 1e-9
        self.last_pub_sec = 0.0
        self.last_imu_sec = None
        self.measurement_dominant = False
        self.carto_seen = 0
        self.skip_until_sec = 0.0

        # --- ROS I/O ---
        self.sub_imu  = self.create_subscription(Imu, self.imu_topic, self.cb_imu, qos_profile_sensor_data)
        self.sub_odom = self.create_subscription(Odometry, self.odom_topic, self.cb_odom, 10)
        self.pub_odom = self.create_publisher(Odometry, self.odom_pub_topic, 10)

        # 더미 predict 타이머(100 Hz): IMU가 끊겨도 시간 전진 보장
        self.timer = self.create_timer(1.0/100.0, self.timer_predict)

        self.get_logger().info(f"[CartoIMUEKF] imu={self.imu_topic} odom={self.odom_topic} out={self.odom_pub_topic} | imu_has_gravity={self.imu_has_gravity}")

    # ---------- Callbacks ----------
    def cb_imu(self, m: Imu):
        t = m.header.stamp.sec + m.header.stamp.nanosec*1e-9
        self.last_imu_sec = t
        self.measurement_dominant = False  # 정상 모드
        try:
            self.ekf.predict(m, t, q_scale=1.0, imu_has_gravity=self.imu_has_gravity)
        except Exception as e:
            self.get_logger().warn(f"predict failed: {e}")
        self._publish_if_due()

    def timer_predict(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.last_imu_sec is None or (now - self.last_imu_sec) > self.imu_timeout_sec:
            # IMU 타임아웃 → 측정우선(MD) 모드로 더미 predict + Q 인플레이트
            self.measurement_dominant = True
            dummy = Imu()  # 0 각속/가속
            try:
                self.ekf.predict(dummy, now, q_scale=self.md_q_scale, imu_has_gravity=self.imu_has_gravity)
            except Exception as e:
                self.get_logger().warn(f"dummy predict failed: {e}")
            self._publish_if_due()

    def cb_odom(self, m: Odometry):
        t_now = self.get_clock().now().nanoseconds * 1e-9
        p = np.array([m.pose.pose.position.x,
                      m.pose.pose.position.y,
                      m.pose.pose.position.z], float)
        q = quat_from_msg(m.pose.pose.orientation)

        # 초기화: Cartographer 첫 pose로
        if not self.initialized:
            t0 = m.header.stamp.sec + m.header.stamp.nanosec*1e-9
            self.ekf.p = p.copy()
            self.ekf.v = np.zeros(3, float)
            self.ekf.q = quat_normalize(q.copy())
            self.ekf.P = np.eye(9, dtype=float) * 1e-2
            self.ekf.last_t = t0
            self.initialized = True
            self.carto_seen = 0
            self.get_logger().info("EKF initialized from /odom.")
            return

        # 부트 스킵
        self.carto_seen += 1
        if self.carto_seen <= self.boot_skip:
            return

        # 쿨다운(스킵 이후)
        if t_now < self.skip_until_sec:
            return

        # 기본 R
        Rp  = self.Rp_default.copy()
        Rth = self.Rth_default.copy()

        # 웜업 인플레이트
        elapsed = t_now - self.start_sec
        chi2 = self.chi2_thresh
        amax = self.alpha_max
        if elapsed < self.warmup_sec:
            Rp  *= 5.0
            Rth *= 5.0
            chi2 = max(chi2, 50.0)
            amax = max(amax, 200.0)

        # 메시지 공분산 사용 (유효할 때만)
        if self.use_carto_cov:
            C6 = cov6_from_odom(m)
            if C6 is None:
                self.get_logger().warn("Carto covariance invalid/zero → using relaxed default Rp/Rθ")
            else:
                Rp = C6[:3,:3]; Rth = C6[3:,3:]

        # 2D 모드: z/roll/pitch 제외
        H_mask = [True, True, False,  False, False, True] if self.use_2d_mode else None

        # 업데이트 수행
        try:
            lam = self.ekf.update_pose(
                p_meas=p, q_meas=q,
                Rp=Rp, Rth=Rth,
                H_mask=H_mask,
                chi2_thresh=chi2, alpha_max=amax,
                skip_thresh=self.skip_thresh
            )
        except Exception as e:
            self.get_logger().warn(f"pose update failed: {e}")
            return

        # 급작 이동 대응: 소프트/하드 리셋
        s = self.ekf.get_state()
        rp = float(np.linalg.norm((p - s["pos"])[0:2])) if self.use_2d_mode else float(np.linalg.norm(p - s["pos"]))
        rth = ang_diff_deg(q, s["rot"])

        if lam is not None and lam > self.nis_reset_thresh:
            self.ekf.hard_snap_to_measurement(p, q, vel_damping=self.vel_damping_on_reset)
            self.skip_until_sec = t_now + self.cooldown_sec
            self.get_logger().warn(f"[HARD RESET] λ={lam:.1f} → snap to Carto (cooldown {self.cooldown_sec:.2f}s)")
        elif (rp > self.pos_reset_thresh_m) or (rth > self.yaw_reset_thresh_deg):
            self.ekf.soft_blend_toward_measurement(p, q, beta=self.snap_beta)
            self.get_logger().warn(f"[SOFT RESET] rp={rp:.2f}m rθ={rth:.1f}° → blend β={self.snap_beta:.2f}")

        # 로깅
        mode = "MD" if self.measurement_dominant else "N"
        lam_txt = f"{lam:.2f}" if lam is not None else "nan"
        self.get_logger().info(f"upd[{mode}] λ={lam_txt} | rp={rp:.3f} m, rθ={rth:.2f}°")

        self._publish_if_due()

    # ---------- Publisher ----------
    def _publish_if_due(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if (now - self.last_pub_sec) < self.dt_pub:
            return
        self.last_pub_sec = now

        s = self.ekf.get_state()
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = "map"      # 필요 시 "odom"으로 변경
        odom.child_frame_id  = "base_link"
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
