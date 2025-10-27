#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ros2 run airio_imu_odometry odom_error_evaluator_node.py \
  --ros-args \
  -p gt_topic:=/odom \
  -p pred_topic:=/odom_airio \
  -p time_window_sec:=3.0 \
  -p segment_length_m:=5.0 \
  -p ate_thresh_m:=0.2 \
  -p rte_trans_thresh_m:=0.1 \
  -p rte_rot_thresh_deg:=3.0
"""
import math
import numpy as np
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from nav_msgs.msg import Odometry
from std_msgs.msg import Float32

import os
import csv
import matplotlib.pyplot as plt                                                                                                                                                                             

# ------------------------------
# Quaternion / SE3 Utilities
# ------------------------------
def quat_to_np(q):
    return np.array([q.x, q.y, q.z, q.w], dtype=np.float64)

def quat_normalize(q):
    n = np.linalg.norm(q)
    if n < 1e-12: 
        return np.array([0., 0., 0., 1.], dtype=np.float64)
    return q / n

def quat_multiply(q1, q2):
    x1,y1,z1,w1 = q1; x2,y2,z2,w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ], dtype=np.float64)

def quat_conjugate(q):
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)

def quat_slerp(q0, q1, t):
    # q0, q1 normalized
    q0 = quat_normalize(q0); q1 = quat_normalize(q1)
    dot = np.dot(q0, q1)
    # ensure shortest path
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    DOT_THRESH = 0.9995
    if dot > DOT_THRESH:
        # nearly linear
        q = q0 + t*(q1 - q0)
        return quat_normalize(q)
    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    s0 = math.sin(theta_0 - theta) / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return s0*q0 + s1*q1

def rot_from_quat(q):
    # q = [x,y,z,w]
    x,y,z,w = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    R = np.array([
        [1.0 - 2*(yy+zz),     2*(xy - wz),         2*(xz + wy)],
        [    2*(xy + wz), 1.0 - 2*(xx+zz),         2*(yz - wx)],
        [    2*(xz - wy),     2*(yz + wx),     1.0 - 2*(xx+yy)]
    ], dtype=np.float64)
    return R

def se3_from_odom(msg):
    p = np.array([msg.pose.pose.position.x,
                  msg.pose.pose.position.y,
                  msg.pose.pose.position.z], dtype=np.float64)
    q = quat_to_np(msg.pose.pose.orientation)
    q = quat_normalize(q)
    R = rot_from_quat(q)
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = R
    T[:3, 3] = p
    return T, p, q

def se3_inv(T):
    R = T[:3,:3]
    t = T[:3,3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3,:3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti

def se3_rel(T_a, T_b):
    # T_a^(-1) * T_b
    return se3_inv(T_a) @ T_b

def se3_trans_angle(T):
    t = T[:3,3]
    R = T[:3,:3]
    # rotation angle from trace
    tr = np.trace(R)
    angle = math.acos(max(min((tr - 1.0)/2.0, 1.0), -1.0))
    return np.linalg.norm(t), angle  # (translation norm [m], angle [rad])


# ------------------------------
# Interpolation Buffer
# ------------------------------
class PoseBuffer:
    """ Timestamped pose buffer for interpolation. """
    def __init__(self, maxlen=5000):
        self.buf = deque(maxlen=maxlen)  # each: (t:float, p:np(3,), q:np(4,))

    def push(self, t, p, q):
        self.buf.append((t, p, q))

    def interpolate(self, t_query, max_gap=0.2):
        """Return (p,q) interpolated at t_query. If cannot, return None."""
        if len(self.buf) < 2:
            return None
        # find two samples surrounding t_query
        # buffer is append-right; not guaranteed sorted by time -> ensure time non-decreasing use scan
        # But Odometry times should be non-decreasing; still, scan for bracketing pair
        prev = None
        for cur in self.buf:
            if prev is None:
                prev = cur
                continue
            t0, p0, q0 = prev
            t1, p1, q1 = cur
            if t0 <= t_query <= t1:
                dt = t1 - t0
                if dt <= 0.0:
                    return None
                if (t_query - t0) > max_gap or (t1 - t_query) > max_gap:
                    return None
                u = (t_query - t0) / dt
                p = (1.0 - u) * p0 + u * p1
                q = quat_slerp(q0, q1, u)
                return p, q
            prev = cur
        return None


# ------------------------------
# Evaluator Node
# ------------------------------
class OdomErrorEvaluator(Node):
    """
    Subscribes:
      - /odom         : Ground Truth (GT)
      - /odom_airio   : Prediction (Pred)
    Publishes:
      - /odom_eval/ate_rmse_window              (Float32)
      - /odom_eval/rte_trans_rmse_window        (Float32)
      - /odom_eval/rte_rot_deg_rmse_window      (Float32)
      - /odom_eval/confidence                   (Float32)
    """
    def __init__(self):
        super().__init__("odom_error_evaluator")

        # ---- Parameters ----
        self.declare_parameter("gt_topic", "/odom")
        self.declare_parameter("pred_topic", "/odom_airio")
        self.declare_parameter("time_window_sec", 3.0)       # ATE 윈도우(초)
        self.declare_parameter("segment_length_m", 5.0)      # RTE 세그먼트 길이(미터)
        self.declare_parameter("max_interp_gap_sec", 0.2)    # 보간 허용 간격(초)
        self.declare_parameter("max_buffer_sec", 30.0)       # 버퍼 유지 시간(초)

        # 신뢰도 임계값(적절히 조정)
        self.declare_parameter("ate_thresh_m", 0.20)         # 최근 ATE RMSE 임계
        self.declare_parameter("rte_trans_thresh_m", 0.10)   # RTE 번들 변위 임계
        self.declare_parameter("rte_rot_thresh_deg", 3.0)    # RTE 번들 회전 임계

        self.gt_topic = self.get_parameter("gt_topic").get_parameter_value().string_value
        self.pred_topic = self.get_parameter("pred_topic").get_parameter_value().string_value
        self.time_window = float(self.get_parameter("time_window_sec").get_parameter_value().double_value)
        self.seg_len = float(self.get_parameter("segment_length_m").get_parameter_value().double_value)
        self.max_interp_gap = float(self.get_parameter("max_interp_gap_sec").get_parameter_value().double_value)
        self.max_buffer_sec = float(self.get_parameter("max_buffer_sec").get_parameter_value().double_value)

        self.ate_thresh = float(self.get_parameter("ate_thresh_m").get_parameter_value().double_value)
        self.rte_trans_thresh = float(self.get_parameter("rte_trans_thresh_m").get_parameter_value().double_value)
        self.rte_rot_thresh_deg = float(self.get_parameter("rte_rot_thresh_deg").get_parameter_value().double_value)
        
        self.log_dir = "./odom_eval_results"
        os.makedirs(self.log_dir, exist_ok=True)

        # --- 로그 버퍼 ---
        self.log_ate = []
        self.log_rte_trans = []
        self.log_rte_rot = []
        self.log_conf = []
        
        # ---- Buffers ----
        self.gt_buf = PoseBuffer(maxlen=20000)
        self.pred_buf = PoseBuffer(maxlen=20000)

        # ATE window buffer: (t, error_m)
        self.ate_window = deque(maxlen=10000)

        # For RTE we keep GT cumulative distances and time-ordered indices
        self.gt_traj = deque(maxlen=20000)   # (t, p(3,), q(4,))
        self.pred_traj = deque(maxlen=20000) # (t, p(3,), q(4,))
        self.gt_cumdist = deque(maxlen=20000)  # cumulative distance along GT

        # QoS
        qos = QoSProfile(depth=50)
        qos.reliability = QoSReliabilityPolicy.BEST_EFFORT
        qos.history = QoSHistoryPolicy.KEEP_LAST

        # Subs
        self.create_subscription(Odometry, self.gt_topic, self.gt_cb, qos)
        self.create_subscription(Odometry, self.pred_topic, self.pred_cb, qos)

        # Pubs
        self.pub_ate = self.create_publisher(Float32, "/odom_eval/ate_rmse_window", 10)
        self.pub_rte_trans = self.create_publisher(Float32, "/odom_eval/rte_trans_rmse_window", 10)
        self.pub_rte_rotdeg = self.create_publisher(Float32, "/odom_eval/rte_rot_deg_rmse_window", 10)
        self.pub_conf = self.create_publisher(Float32, "/odom_eval/confidence", 10)

        # Timers
        self.timer = self.create_timer(0.05, self.on_tick)  # 20 Hz 평가

        self.get_logger().info(f"Start OdomErrorEvaluator: GT={self.gt_topic}, Pred={self.pred_topic}")

    # ------------------ Callbacks ------------------
    def gt_cb(self, msg: Odometry):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        _, p, q = se3_from_odom(msg)
        self.gt_buf.push(t, p, q)
        self.gt_traj.append((t, p, q))
        # cumulative distance
        if len(self.gt_cumdist) == 0:
            self.gt_cumdist.append(0.0)
        else:
            prev_t, prev_p, _ = self.gt_traj[-2]
            self.gt_cumdist.append(self.gt_cumdist[-1] + float(np.linalg.norm(p - prev_p)))
        self.prune_buffers(t)

    def pred_cb(self, msg: Odometry):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        _, p, q = se3_from_odom(msg)
        self.pred_buf.push(t, p, q)
        self.pred_traj.append((t, p, q))
        # prune with latest known GT time if exists
        t_ref = self.gt_traj[-1][0] if len(self.gt_traj) else t
        self.prune_buffers(t_ref)

    # ------------------ Periodic Evaluation ------------------
    def on_tick(self):
        if len(self.gt_traj) < 2 or len(self.pred_traj) < 2:
            return
        t_now = self.gt_traj[-1][0]

        # ---- ATE (time-window RMSE) ----
        # ATE(t) := || p_pred(t) - p_gt(t) || ; window over [t_now - W, t_now]
        t_start = t_now - self.time_window
        # find GT samples inside window, compute pointwise error by interpolating Pred at each GT time
        ate_samples = []
        for t, p_gt, q_gt in reversed(self.gt_traj):
            if t < t_start:
                break
            interp = self.pred_buf.interpolate(t, max_gap=self.max_interp_gap)
            if interp is None:
                continue
            p_pred, q_pred = interp
            ate_samples.append(float(np.linalg.norm(p_pred - p_gt)))
        if ate_samples:
            rmse_ate = math.sqrt(np.mean(np.square(ate_samples)))
            self.ate_window.append((t_now, rmse_ate))
            self.pub_ate.publish(Float32(data=float(rmse_ate)))

        # ---- RTE (segment-length L) ----
        # For the latest GT sample at t_now, find past GT pose at distance ≈ L behind.
        rte_trans_vals = []
        rte_rot_vals = []
        if len(self.gt_traj) >= 2:
            idx_b = len(self.gt_traj) - 1
            cum_b = self.gt_cumdist[-1]
            # walk backward until cumdist difference >= seg_len
            idx_a = idx_b
            while idx_a > 0 and (cum_b - self.gt_cumdist[idx_a-1]) < self.seg_len:
                idx_a -= 1
            if idx_a != idx_b and (cum_b - self.gt_cumdist[idx_a]) >= 0.5 * self.seg_len:
                t_a, p_a_gt, q_a_gt = self.gt_traj[idx_a]
                t_b, p_b_gt, q_b_gt = self.gt_traj[idx_b]
                # interpolate prediction at t_a and t_b
                interp_a = self.pred_buf.interpolate(t_a, max_gap=self.max_interp_gap)
                interp_b = self.pred_buf.interpolate(t_b, max_gap=self.max_interp_gap)
                if interp_a is not None and interp_b is not None:
                    p_a_pr, q_a_pr = interp_a
                    p_b_pr, q_b_pr = interp_b
                    # build SE3
                    T_a_gt = np.eye(4); T_a_gt[:3,:3]=rot_from_quat(q_a_gt); T_a_gt[:3,3]=p_a_gt
                    T_b_gt = np.eye(4); T_b_gt[:3,:3]=rot_from_quat(q_b_gt); T_b_gt[:3,3]=p_b_gt
                    T_a_pr = np.eye(4); T_a_pr[:3,:3]=rot_from_quat(q_a_pr); T_a_pr[:3,3]=p_a_pr
                    T_b_pr = np.eye(4); T_b_pr[:3,:3]=rot_from_quat(q_b_pr); T_b_pr[:3,3]=p_b_pr
                    # relative transforms
                    dT_gt = se3_rel(T_a_gt, T_b_gt)
                    dT_pr = se3_rel(T_a_pr, T_b_pr)
                    # error between relatives: eT = dT_pr^(-1) * dT_gt
                    eT = se3_rel(dT_pr, dT_gt)
                    trans_err, rot_err = se3_trans_angle(eT)  # m, rad
                    rte_trans_vals.append(trans_err)
                    rte_rot_vals.append(rot_err * 180.0 / math.pi)  # deg

        # Publish RTE windowed RMS (use recent N samples in ~time_window)
        # Keep small deques of last K RTE samples using same ate_window time scope
        if not hasattr(self, "_rte_trans_window"):
            self._rte_trans_window = deque(maxlen=500)
            self._rte_rotdeg_window = deque(maxlen=500)

        if rte_trans_vals:
            self._rte_trans_window.append((t_now, float(np.mean(rte_trans_vals))))
        if rte_rot_vals:
            self._rte_rotdeg_window.append((t_now, float(np.mean(rte_rot_vals))))

        # prune RTE windows by time
        self._prune_time_window(self._rte_trans_window, t_now, self.time_window)
        self._prune_time_window(self._rte_rotdeg_window, t_now, self.time_window)

        # publish RMS
        if len(self._rte_trans_window) > 2:
            trans_rmse = math.sqrt(np.mean([v*v for _, v in self._rte_trans_window]))
            self.pub_rte_trans.publish(Float32(data=float(trans_rmse)))
        if len(self._rte_rotdeg_window) > 2:
            rot_rmse = math.sqrt(np.mean([v*v for _, v in self._rte_rotdeg_window]))
            self.pub_rte_rotdeg.publish(Float32(data=float(rot_rmse)))

        # ---- Confidence (0~1) ----
        # Combine normalized penalties of ATE and RTE
        ate_term = 1.0
        if self.ate_window:
            cur_ate = self.ate_window[-1][1]
            ate_term = max(0.0, 1.0 - (cur_ate / max(self.ate_thresh, 1e-6)))

        trans_term = 1.0
        if len(self._rte_trans_window) > 0:
            cur_rte_t = self._rte_trans_window[-1][1]
            trans_term = max(0.0, 1.0 - (cur_rte_t / max(self.rte_trans_thresh, 1e-6)))

        rot_term = 1.0
        if len(self._rte_rotdeg_window) > 0:
            cur_rte_r = self._rte_rotdeg_window[-1][1]
            rot_term = max(0.0, 1.0 - (cur_rte_r / max(self.rte_rot_thresh_deg, 1e-6)))

        # Weighted blend (조정 가능)
        w_ate, w_trans, w_rot = 0.5, 0.35, 0.15
        confidence = float(np.clip(w_ate*ate_term + w_trans*trans_term + w_rot*rot_term, 0.0, 1.0))
        self.pub_conf.publish(Float32(data=confidence))

        if hasattr(self, 'ate_window') and len(self.ate_window) > 0:
            t = self.ate_window[-1][0]
            ate = self.ate_window[-1][1]
            rte_t = self._rte_trans_window[-1][1] if len(self._rte_trans_window) > 0 else None
            rte_r = self._rte_rotdeg_window[-1][1] if len(self._rte_rotdeg_window) > 0 else None
            conf = self.pub_conf.get_subscription_count()  # dummy check
            # 실제 confidence는 self.pub_conf.publish(Float32(data=confidence)) 근처에 있음
            conf_val = getattr(self, "last_confidence", 0.0)
            self.log_ate.append((t, ate))
            self.log_rte_trans.append((t, rte_t))
            self.log_rte_rot.append((t, rte_r))
            self.log_conf.append((t, conf_val))
    
    def save_plots(self):
        if not self.log_ate:
            self.get_logger().warn("No log data to save.")
            return

        csv_path = os.path.join(self.log_dir, "odom_eval_log.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "ATE[m]", "RTE_trans[m]", "RTE_rot[deg]", "Confidence"])
            for i in range(len(self.log_ate)):
                t = self.log_ate[i][0]
                ate = self.log_ate[i][1]
                rtet = self.log_rte_trans[i][1] if i < len(self.log_rte_trans) else None
                rter = self.log_rte_rot[i][1] if i < len(self.log_rte_rot) else None
                conf = self.log_conf[i][1] if i < len(self.log_conf) else None
                writer.writerow([t, ate, rtet, rter, conf])
        self.get_logger().info(f"Saved CSV: {csv_path}")

        # === Plot ===
        fig, ax1 = plt.subplots(figsize=(10,5))
        t_vals = [t for t,_ in self.log_ate]
        ate_vals = [v for _,v in self.log_ate]
        rte_vals = [v if v is not None else np.nan for _,v in self.log_rte_trans]
        conf_vals = [v if v is not None else np.nan for _,v in self.log_conf]

        ax1.plot(t_vals, ate_vals, 'b-', label="ATE [m]")
        ax1.plot(t_vals, rte_vals, 'r--', label="RTE_trans [m]")
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Error [m]")
        ax1.grid(True)
        ax1.legend(loc="upper left")

        ax2 = ax1.twinx()
        ax2.plot(t_vals, conf_vals, 'g-', label="Confidence")
        ax2.set_ylabel("Confidence [0-1]")
        ax2.set_ylim([0,1])
        ax2.legend(loc="upper right")

        plt.title("Odometry Evaluation (ATE, RTE, Confidence)")
        plt.tight_layout()
        fig_path = os.path.join(self.log_dir, "odom_eval_plot.png")
        plt.savefig(fig_path, dpi=150)
        plt.close(fig)
        self.get_logger().info(f"Saved plot: {fig_path}")

    # ------------------ Helpers ------------------
    def prune_buffers(self, t_ref):
        # drop old samples older than max_buffer_sec
        t_min = t_ref - self.max_buffer_sec
        def prune_deque(dq, key_idx=0):
            while len(dq) and dq[0][key_idx] < t_min:
                dq.popleft()

        # gt_traj: (t,p,q), pred_traj: (t,p,q), gt_cumdist aligned by index
        while len(self.gt_traj) and self.gt_traj[0][0] < t_min:
            self.gt_traj.popleft()
            self.gt_cumdist.popleft()
        while len(self.pred_traj) and self.pred_traj[0][0] < t_min:
            self.pred_traj.popleft()

        # pose buffers (store same tuples)
        while len(self.gt_buf.buf) and self.gt_buf.buf[0][0] < t_min:
            self.gt_buf.buf.popleft()
        while len(self.pred_buf.buf) and self.pred_buf.buf[0][0] < t_min:
            self.pred_buf.buf.popleft()

        # ATE window uses (t,val)
        self._prune_time_window(self.ate_window, t_ref, self.time_window)

    @staticmethod
    def _prune_time_window(dq, t_now, W):
        while len(dq) and dq[0][0] < (t_now - W):
            dq.popleft()


def main(args=None):
    rclpy.init(args=args)
    node = OdomErrorEvaluator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
