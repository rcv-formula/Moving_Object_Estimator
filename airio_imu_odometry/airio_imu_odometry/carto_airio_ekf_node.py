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
    x1,y1,z1,w1 = q1; x2,y2,z2,w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ], float)

def quat_normalize(q):
    n = np.linalg.norm(q)
    return q/n if n > 1e-12 else np.array([0,0,0,1], float)

def so3_exp(phi):
    th = np.linalg.norm(phi)
    if th < 1e-12: return np.array([0,0,0,1], float)
    axis = phi/th; s = np.sin(th/2.0)
    return np.array([axis[0]*s, axis[1]*s, axis[2]*s, np.cos(th/2.0)], float)

def so3_log(q):
    q = quat_normalize(q); v = q[:3]; w = q[3]
    s = np.linalg.norm(v)
    if s < 1e-12: return np.zeros(3, float)
    axis = v/s; th = 2.0*np.arctan2(s, w)
    return axis*th

def rot_from_quat(q):
    x,y,z,w = q
    xx,yy,zz = x*x, y*y, z*z
    xy,xz,yz = x*y, x*z, y*z
    wx,wy,wz = w*x, w*y, w*z
    return np.array([
        [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],
        [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)],
        [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)]
    ], float)

def quat_from_msg(qm: Quaternion):
    return np.array([qm.x, qm.y, qm.z, qm.w], float)

def quat_to_msg(q):
    m = Quaternion(); m.x,m.y,m.z,m.w = q.tolist(); return m

def ang_diff_deg(q_meas, q_pred):
    dot = abs(float(np.dot(q_meas, q_pred)))
    dot = max(min(dot,1.0), -1.0)
    return np.degrees(2*np.arccos(dot))

# ==========================
# Odometry covariance helper
# ==========================
def cov6_from_odom(msg: Odometry):
    C = np.array(msg.pose.covariance, float)
    if C.size != 36: return None
    C = C.reshape(6,6)
    if np.any(np.diag(C) <= 1e-12): return None
    C = 0.5*(C + C.T) + 1e-12*np.eye(6)
    return C

# ==========================
# Simple 9-state error EKF
# x = [δp, δv, δθ], nominal = {p, v, q}
# ==========================
class SimpleImuPoseEKF:
    def __init__(self, gyro_noise=0.02, acc_noise=0.20, g=9.80665):
        self.p = np.zeros(3); self.v = np.zeros(3); self.q = np.array([0,0,0,1], float)
        self.P = np.eye(9)*1e-2
        self.qp  = 1e-3
        self.qv  = float(acc_noise**2)
        self.qth = float(gyro_noise**2)
        self.g = np.array([0,0,-g], float)
        self.last_t = None

    def predict(self, imu: Imu, t: float, q_scale=1.0, imu_has_gravity=True):
        if self.last_t is None:
            self.last_t = t; return
        dt = max(1e-4, min(0.05, t - self.last_t))
        self.last_t = t

        wm = np.array([imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z], float)
        dq = so3_exp(wm*dt)
        self.q = quat_normalize(quat_multiply(self.q, dq))
        R = rot_from_quat(self.q)

        am = np.array([imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z], float)
        if imu_has_gravity:
            a_w = R @ am + self.g   # raw IMU(-g 포함)
        else:
            a_w = R @ am            # 이미 중력 보정된 IMU

        v_old = self.v.copy()
        self.p = self.p + v_old*dt + 0.5*a_w*(dt**2)
        self.v = v_old + a_w*dt

        I3 = np.eye(3)
        Phi = np.block([[I3, dt*I3, np.zeros((3,3))],
                        [np.zeros((3,3)), I3, np.zeros((3,3))],
                        [np.zeros((3,3)), np.zeros((3,3)), I3]])
        Q = np.block([
            [self.qp*np.eye(3)*dt, np.zeros((3,3)), np.zeros((3,3))],
            [np.zeros((3,3)), self.qv*np.eye(3)*dt, np.zeros((3,3))],
            [np.zeros((3,3)), np.zeros((3,3)), self.qth*np.eye(3)*dt]
        ]) * float(q_scale)
        self.P = Phi @ self.P @ Phi.T + Q

    def update_pose(self, p_meas, q_meas, Rp, Rth, H_mask=None,
                    chi2_thresh=30.0, alpha_max=100.0, skip_thresh=-1.0):
        p_meas = np.asarray(p_meas, float)
        q_meas = quat_normalize(np.asarray(q_meas, float))

        dp = p_meas - self.p
        q_err = quat_multiply(q_meas, np.array([-self.q[0], -self.q[1], -self.q[2], self.q[3]]))
        dth = so3_log(q_err)

        z = np.hstack([dp, dth])
        H = np.block([[np.eye(3), np.zeros((3,3)), np.zeros((3,3))],
                      [np.zeros((3,3)), np.zeros((3,3)), np.eye(3)]])  # 6x9
        Rm = np.block([[Rp, np.zeros((3,3))],
                       [np.zeros((3,3)), Rth]])

        if H_mask is not None:
            mask = np.array(H_mask, bool)
            H = H[mask]; z = z[mask]; Rm = Rm[np.ix_(mask, mask)]

        S = H @ self.P @ H.T + Rm
        S = 0.5*(S + S.T) + 1e-9*np.eye(S.shape[0])
        S_inv = np.linalg.inv(S)

        lam = float(z.T @ S_inv @ z) if z.size > 0 else 0.0
        if skip_thresh > 0 and lam > skip_thresh:
            return lam

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

    def update_velocity_body(self, v_b, Rv_b):
        """Body-frame 속도 업데이트 (AirIO)"""
        v_b = np.asarray(v_b, float).reshape(3)
        Rv_b = np.asarray(Rv_b, float).reshape(3,3)
        # 측정 모델: z = v_b_meas - R^T v_world  ≈ H dx
        Rwb = rot_from_quat(self.q)
        z = v_b - (Rwb.T @ self.v)

        H = np.zeros((3,9), float)
        H[:,3:6] = -Rwb.T     # δv_world
        # δθ 항(선형화)도 넣을 수 있으나 안정성을 위해 생략/작게
        S = H @ self.P @ H.T + Rv_b
        S = 0.5*(S + S.T) + 1e-9*np.eye(3)
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ z
        self.P = (np.eye(9) - K @ H) @ self.P
        self.p += dx[0:3]
        self.v += dx[3:6]
        self.q = quat_normalize(quat_multiply(self.q, so3_exp(dx[6:9])))

    def hard_snap_to_measurement(self, p_meas, q_meas, vel_damping=0.5):
        self.p = np.asarray(p_meas, float)
        self.q = quat_normalize(np.asarray(q_meas, float))
        self.v *= float(vel_damping)
        self.P = self.P + np.diag([0.1,0.1,0.1, 0.1,0.1,0.1, np.deg2rad(5)**2]*3)[:9,:9]

    def soft_blend_toward_measurement(self, p_meas, q_meas, beta=0.6):
        p_meas = np.asarray(p_meas, float)
        q_meas = quat_normalize(np.asarray(q_meas, float))
        self.p = (1.0-beta)*self.p + beta*p_meas
        q_err = quat_multiply(q_meas, np.array([-self.q[0], -self.q[1], -self.q[2], self.q[3]]))
        self.q = quat_normalize(quat_multiply(self.q, so3_exp(beta*so3_log(q_err))))

    def get_state(self):
        return {"pos": self.p.copy(), "vel": self.v.copy(), "rot": self.q.copy()}

# ==========================
# ROS2 Node (AirIMU + AirIO + Carto)
# ==========================
class AirioAirimuEKFNode(Node):
    """
    Predict:  /airimu_imu_data (fallback: /imu)
    Update v: /odom_airio (Body velocity)
    Update p: /odom (Cartographer pose)
    Output :  /odom_fused
    """
    def __init__(self):
        super().__init__('airio_airimu_ekf')

        # --- Topics / Rate ---
        self.declare_parameter("imu_topic", "/imu")
        self.declare_parameter("airimu_topic", "/airimu_imu_data")
        self.declare_parameter("odom_carto_topic", "/odom")
        self.declare_parameter("odom_airio_topic", "/odom_airio")
        self.declare_parameter("odom_pub_topic", "/odom_fused")
        self.declare_parameter("odom_pub_rate", 50.0)

        # --- Gravity handling ---
        self.declare_parameter("imu_has_gravity", False)   # 보정된 IMU라면 False 권장

        # --- MD mode when IMU timeout ---
        self.declare_parameter("imu_timeout_sec", 0.25)
        self.declare_parameter("md_q_scale", 150.0)

        # --- Measurement noise fallback (std) ---
        self.declare_parameter("default_pos_std_m", [0.15, 0.15, 0.50])
        self.declare_parameter("default_rpy_std_deg", [1.5, 1.5, 1.5])
        self.declare_parameter("use_carto_cov_from_msg", False)  # Carto 신뢰 ↑를 위해 기본 False

        # --- Gating / Warmup / Skips ---
        self.declare_parameter("chi2_thresh", 50.0)
        self.declare_parameter("alpha_max", 150.0)
        self.declare_parameter("skip_thresh", -1.0)
        self.declare_parameter("warmup_sec", 0.5)
        self.declare_parameter("boot_skip", 0)
        self.declare_parameter("cooldown_sec", 0.0)

        # --- 2D mode ---
        self.declare_parameter("use_2d_mode", True)

        # --- Process noises ---
        self.declare_parameter("gyro_noise", 0.04)
        self.declare_parameter("acc_noise", 0.35)

        # --- Velocity update (AirIO) ---
        self.declare_parameter("use_body_vel_update", True)
        self.declare_parameter("vel_deadband_mmps", 5.0)   # 5 mm/s 이하면 0
        self.declare_parameter("vel_std_body_mps", [0.15, 0.15, 0.30])  # AirIO 속도 표준편차(Body)

        # --- Reset thresholds ---
        self.declare_parameter("pos_reset_thresh_m", 1.2)
        self.declare_parameter("yaw_reset_thresh_deg", 25.0)
        self.declare_parameter("nis_reset_thresh", 400.0)
        self.declare_parameter("snap_beta", 0.5)
        self.declare_parameter("vel_damping_on_reset", 0.7)

        # --- Read params ---
        self.imu_topic = self.get_parameter("imu_topic").value
        self.airimu_topic = self.get_parameter("airimu_topic").value
        self.odom_carto_topic = self.get_parameter("odom_carto_topic").value
        self.odom_airio_topic = self.get_parameter("odom_airio_topic").value
        self.odom_pub_topic = self.get_parameter("odom_pub_topic").value
        self.pub_rate = float(self.get_parameter("odom_pub_rate").value)
        self.dt_pub = 1.0/max(1e-6, self.pub_rate)

        self.imu_has_gravity = bool(self.get_parameter("imu_has_gravity").value)
        self.imu_timeout_sec = float(self.get_parameter("imu_timeout_sec").value)
        self.md_q_scale = float(self.get_parameter("md_q_scale").value)

        pos_std = np.asarray(self.get_parameter("default_pos_std_m").value, float)
        rpy_std_deg = np.asarray(self.get_parameter("default_rpy_std_deg").value, float)
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

        self.use_body_vel_update = bool(self.get_parameter("use_body_vel_update").value)
        self.vel_deadband_mmps = float(self.get_parameter("vel_deadband_mmps").value)
        vel_std_body = np.asarray(self.get_parameter("vel_std_body_mps").value, float)
        self.Rv_body = np.diag(vel_std_body**2)

        self.pos_reset_thresh_m = float(self.get_parameter("pos_reset_thresh_m").value)
        self.yaw_reset_thresh_deg = float(self.get_parameter("yaw_reset_thresh_deg").value)
        self.nis_reset_thresh = float(self.get_parameter("nis_reset_thresh").value)
        self.snap_beta = float(self.get_parameter("snap_beta").value)
        self.vel_damping_on_reset = float(self.get_parameter("vel_damping_on_reset").value)

        # --- Filter / State ---
        self.ekf = SimpleImuPoseEKF(gyro_noise=gyro_n, acc_noise=acc_n)
        self.initialized = False
        self.start_sec = self.get_clock().now().nanoseconds*1e-9
        self.last_pub_sec = 0.0
        self.last_imu_sec = None
        self.last_carto_sec = 0.0
        self.measurement_dominant = False
        self.carto_seen = 0
        self.skip_until_sec = 0.0

        # --- ROS I/O ---
        # AirIMU 보정된 IMU가 우선. 없으면 raw /imu로 predict
        self.sub_airimu = self.create_subscription(Imu, self.airimu_topic, self.cb_imu, 10)
        self.sub_carto  = self.create_subscription(Odometry, self.odom_carto_topic, self.cb_odom_carto, 10)
        self.sub_airio  = self.create_subscription(Odometry, self.odom_airio_topic, self.cb_odom_airio, 10)
        self.pub_odom   = self.create_publisher(Odometry, self.odom_pub_topic, 10)

        # Dummy predict 타이머(100Hz)
        self.timer = self.create_timer(1.0/100.0, self.timer_predict)

        self.get_logger().info(f"[AirioAirimuEKF] airimu={self.airimu_topic} imu={self.imu_topic} carto={self.odom_carto_topic} airio={self.odom_airio_topic} out={self.odom_pub_topic} | imu_has_gravity={self.imu_has_gravity}")

    # ---------- IMU predict ----------
    def cb_imu(self, m: Imu):
        t = m.header.stamp.sec + m.header.stamp.nanosec*1e-9
        # AirIMU와 raw IMU 모두 이 콜백 공유: 마지막 수신 갱신
        self.last_imu_sec = t
        self.measurement_dominant = False
        try:
            self.ekf.predict(m, t, q_scale=1.0, imu_has_gravity=self.imu_has_gravity)
            if self.use_2d_mode:
                # z 위치/속도 고정
                self.ekf.p[2] = 0.0
                self.ekf.v[2] = 0.0
                # yaw-only quaternion으로 투영
                R = rot_from_quat(self.ekf.q)
                yaw = float(np.arctan2(R[1,0], R[0,0]))
                half = 0.5*yaw
                self.ekf.q = np.array([0.0, 0.0, np.sin(half), np.cos(half)], dtype=float)

        except Exception as e:
            self.get_logger().warn(f"predict failed: {e}")
        self._publish_if_due()

    # ---------- Dummy predict (IMU timeout) ----------
    def timer_predict(self):
        now = self.get_clock().now().nanoseconds*1e-9
        imu_lost = (self.last_imu_sec is None) or ((now - self.last_imu_sec) > self.imu_timeout_sec)
        carto_lost = (not self.initialized) or ((now - self.last_carto_sec) > 0.5)

        if imu_lost and carto_lost:
            # 둘 다 끊겼으면 정지 상태 유지(필요 시 속도 감쇠/0 클램프)
            self.ekf.v[:] = 0.0
            return

        if imu_lost:
            self.measurement_dominant = True
            dummy = Imu()
            try:
                self.ekf.predict(dummy, now, q_scale=self.md_q_scale, imu_has_gravity=self.imu_has_gravity)
                if self.use_2d_mode:
                    self.ekf.p[2] = 0.0
                    self.ekf.v[2] = 0.0
                    R = rot_from_quat(self.ekf.q)
                    yaw = float(np.arctan2(R[1,0], R[0,0]))
                    half = 0.5*yaw
                    self.ekf.q = np.array([0.0, 0.0, np.sin(half), np.cos(half)], dtype=float)

            except Exception as e:
                self.get_logger().warn(f"dummy predict failed: {e}")
            self._publish_if_due()

    # ---------- AirIO velocity update ----------
    def cb_odom_airio(self, m: Odometry):
        if not self.use_body_vel_update or not self.initialized:
            return
        try:
            v_b = np.array([m.twist.twist.linear.x,
                            m.twist.twist.linear.y,
                            m.twist.twist.linear.z], float)

            if np.linalg.norm(v_b) * 1000.0 < self.vel_deadband_mmps:
                v_b[:] = 0.0

            self.ekf.update_velocity_body(v_b, self.Rv_body)
        except Exception as e:
            self.get_logger().warn(f"AirIO velocity update failed: {e}")
        self._publish_if_due()

    # ---------- Cartographer pose update ----------
    def cb_odom_carto(self, m: Odometry):
        t_now = self.get_clock().now().nanoseconds*1e-9
        self.last_carto_sec = t_now

        p = np.array([m.pose.pose.position.x,
                      m.pose.pose.position.y,
                      m.pose.pose.position.z], float)
        q = quat_from_msg(m.pose.pose.orientation)

        if not self.initialized:
            t0 = m.header.stamp.sec + m.header.stamp.nanosec*1e-9
            self.ekf.p = p.copy()
            self.ekf.v = np.zeros(3, float)
            self.ekf.q = quat_normalize(q.copy())
            self.ekf.P = np.eye(9)*1e-2
            self.ekf.last_t = t0
            self.initialized = True
            self.carto_seen = 0
            self.get_logger().info("EKF initialized from Cartographer /odom.")
            return

        # 부트 스킵/쿨다운
        self.carto_seen += 1
        if self.carto_seen <= self.boot_skip:
            return
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
            Rp *= 5.0; Rth *= 5.0
            chi2 = max(chi2, 50.0); amax = max(amax, 200.0)

        # Carto 공분산
        if self.use_carto_cov:
            C6 = cov6_from_odom(m)
            if C6 is None:
                # Carto를 더 신뢰: 공격적 R로 강제
                Rp  = np.diag([0.15**2, 0.15**2, 0.50**2])
                Rth = np.diag(np.deg2rad([1.5,1.5,1.5])**2)
                self.get_logger().warn("Carto covariance invalid/zero → using aggressive Rp/Rθ")
            else:
                Rp = C6[:3,:3]; Rth = C6[3:,3:]

        # 2D 마스크
        H_mask = [True, True, False,  False, False, True] if self.use_2d_mode else None

        # 업데이트
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

        # 리셋 로직
        s = self.ekf.get_state()
        rp = float(np.linalg.norm((p - s["pos"])[0:2])) if self.use_2d_mode else float(np.linalg.norm(p - s["pos"]))
        rth = ang_diff_deg(q, s["rot"])

        if lam is not None and lam > self.nis_reset_thresh:
            self.ekf.hard_snap_to_measurement(p, q, vel_damping=self.vel_damping_on_reset)
            self.skip_until_sec = t_now + self.cooldown_sec
            self.get_logger().warn(f"[HARD RESET] λ={lam:.1f} → snap (cooldown {self.cooldown_sec:.2f}s)")
        elif (rp > self.pos_reset_thresh_m) or (rth > self.yaw_reset_thresh_deg):
            self.ekf.soft_blend_toward_measurement(p, q, beta=self.snap_beta)
            self.get_logger().warn(f"[SOFT RESET] rp={rp:.2f}m rθ={rth:.1f}° → β={self.snap_beta:.2f}")

        # 로깅
        mode = "MD" if self.measurement_dominant else "N"
        lam_txt = f"{lam:.2f}" if lam is not None else "nan"
        self.get_logger().info(f"upd[{mode}] λ={lam_txt} | rp={rp:.3f} m, rθ={rth:.2f}°")

        self._publish_if_due()

    # ---------- Publish ----------
    def _publish_if_due(self):
        now = self.get_clock().now().nanoseconds*1e-9
        if (now - self.last_pub_sec) < self.dt_pub:
            return
        self.last_pub_sec = now

        s = self.ekf.get_state()
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = "map"
        odom.child_frame_id  = "base_link"
        odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = s["pos"].tolist()
        odom.pose.pose.orientation = quat_to_msg(s["rot"])
        odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z = s["vel"].tolist()
        self.pub_odom.publish(odom)

# ==========================
def main(args=None):
    rclpy.init(args=args)
    node = AirioAirimuEKFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
