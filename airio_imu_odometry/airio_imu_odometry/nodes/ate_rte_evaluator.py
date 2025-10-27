#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import math
import csv
import time
from collections import deque
from dataclasses import dataclass
from typing import Tuple, Optional, List

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64


# ========= SE(3)/SE(2) 유틸 =========
def quat_to_rot(qx, qy, qz, qw) -> np.ndarray:
    q = np.array([qw, qx, qy, qz], dtype=float)
    q = q / max(1e-12, np.linalg.norm(q))
    w, x, y, z = q
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)]
    ], dtype=float)
    return R


def pose_to_T(odom: Odometry) -> np.ndarray:
    p = odom.pose.pose.position
    q = odom.pose.pose.orientation
    R = quat_to_rot(q.x, q.y, q.z, q.w)
    t = np.array([p.x, p.y, p.z], dtype=float)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def umeyama_se3(A: np.ndarray, B: np.ndarray, with_scale: bool = False) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    A: Nx3 (est), B: Nx3 (gt). 찾는 변환: B ≈ s * R * A + t
    """
    assert A.shape == B.shape and A.shape[1] == 3
    N = A.shape[0]
    mu_A, mu_B = A.mean(axis=0), B.mean(axis=0)
    Ac, Bc = A - mu_A, B - mu_B
    Sigma = (Bc.T @ Ac) / N
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    if with_scale:
        varA = (Ac * Ac).sum() / N
        s = float(np.trace(np.diag(D) @ S) / max(1e-12, varA))
    else:
        s = 1.0
    t = mu_B - s * (R @ mu_A)
    return R, t, s


def se2_xyyaw_from_T(T: np.ndarray) -> Tuple[float, float, float]:
    x, y = T[0, 3], T[1, 3]
    yaw = math.atan2(T[1, 0], T[0, 0])
    return x, y, yaw


def align_first_frame(T_est0: np.ndarray, T_gt0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    # GT = R * T_est + t (첫 프레임 정렬)
    R0 = T_gt0[:3, :3] @ T_est0[:3, :3].T
    t0 = T_gt0[:3, 3] - R0 @ T_est0[:3, 3]
    return R0, t0, 1.0


def yaw_from_R(R: np.ndarray) -> float:
    # yaw(rad) = atan2(r21, r11)
    return math.atan2(R[1, 0], R[0, 0])


# ========= 데이터 구조 =========
@dataclass
class PoseStamped:
    t: float
    T: np.ndarray  # 4x4


# ========= 노드 =========
class ATERTEEvaluator(Node):
    """
    /odom_airio(추정) vs /odom(GT) 비교
      - ATE: Umeyama/SE2/첫프레임 정렬 후 RMSE
      - RTE(yaw-only): Δt 간격 상대변환의 yaw 오차(deg) RMSE
      파라미터 설명::

    """
    def __init__(self):
        super().__init__('ate_rte_evaluator')

        # ---- 파라미터 ----
        self.declare_parameter('est_topic', '/odom_airio')
        self.declare_parameter('gt_topic', '/odom')
        self.declare_parameter('assoc_max_dt', 0.02)          # s
        self.declare_parameter('window_sec', 20.0)            # s
        self.declare_parameter('align_mode', 'se2_xyyaw')     # 'se3_umeyama'|'se2_xyyaw'|'first'
        self.declare_parameter('align_with_scale', False)     # Umeyama scale 고려 여부
        self.declare_parameter('recompute_alignment_every', 2.0)  # s; 0이면 고정
        self.declare_parameter('rte_delta_sec', 1.0)          # s
        self.declare_parameter('publish_rate', 5.0)           # Hz
        self.declare_parameter('output_csv', './result')        
        self.declare_parameter('save_plots', True)            # 종료 시 matplotlib 저장
        self.declare_parameter('eval_stride', 5)              # 1이면 전체, 5면 1/5 샘플만 평가

        self.est_topic = self.get_parameter('est_topic').value
        self.gt_topic = self.get_parameter('gt_topic').value
        self.assoc_max_dt = float(self.get_parameter('assoc_max_dt').value)
        self.window_sec = float(self.get_parameter('window_sec').value)
        self.align_mode = str(self.get_parameter('align_mode').value)
        self.align_with_scale = bool(self.get_parameter('align_with_scale').value)
        self.recompute_alignment_every = float(self.get_parameter('recompute_alignment_every').value)
        self.rte_delta_sec = float(self.get_parameter('rte_delta_sec').value)
        self.publish_rate = float(self.get_parameter('publish_rate').value)
        self.output_csv = str(self.get_parameter('output_csv').value)
        self.save_plots = bool(self.get_parameter('save_plots').value)
        self.eval_stride = int(self.get_parameter('eval_stride').value)

        # ---- 구독 ----
        self.sub_est = self.create_subscription(Odometry, self.est_topic, self.cb_est, 50)
        self.sub_gt = self.create_subscription(Odometry, self.gt_topic, self.cb_gt, 50)

        # ---- 버퍼 ----
        self.buf_est: deque[PoseStamped] = deque(maxlen=100000)
        self.buf_gt: deque[PoseStamped] = deque(maxlen=100000)
        self.last_align_time: float = -1.0
        self.aligned_R = np.eye(3)
        self.aligned_t = np.zeros(3)
        self.aligned_s = 1.0
        self.have_first = False
        self.first_T_est = None
        self.first_T_gt = None
        self.last_align_time: float = -1.0 

        # ---- 퍼블리셔 ----
        self.pub_ate = self.create_publisher(Float64, '/metrics/ate_rmse', 10)
        self.pub_rte_yaw = self.create_publisher(Float64, '/metrics/rte_yaw_deg_rmse', 10)

        # # ---- 타이머 ----
        # self.timer_pub = self.create_timer(1.0 / max(1e-6, self.publish_rate),
        #                                    self.timer_compute_and_publish)

        # ---- CSV ----
        self.csv_f = None
        self.csv_w = None
        if self.output_csv:
            os.makedirs(os.path.dirname(self.output_csv) or ".", exist_ok=True)
            self.csv_f = open(self.output_csv, 'w', newline='')
            self.csv_w = csv.writer(self.csv_f)
            self.csv_w.writerow(['t', 'ATE_RMSE', 'RTE_yaw_deg_RMSE'])

        self.get_logger().info(f"[ATE/RTE] est={self.est_topic}, gt={self.gt_topic}, "
                               f"align={self.align_mode}, Δt={self.rte_delta_sec}s")

    # ---------- 콜백 ----------
    def cb_est(self, msg: Odometry):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.buf_est.append(PoseStamped(t, pose_to_T(msg)))

    def cb_gt(self, msg: Odometry):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.buf_gt.append(PoseStamped(t, pose_to_T(msg)))
        self.timer_compute_and_publish()

    # ---------- 내부 함수 ----------
    def _purge_old(self):
        # 최신 스탬프(둘 중 큰 값)를 기준
        latest_est = self.buf_est[-1].t if self.buf_est else None
        latest_gt  = self.buf_gt[-1].t if self.buf_gt else None
        if latest_est is None and latest_gt is None:
            return
        now_t = max(latest_est or -1e30, latest_gt or -1e30)
        lim = now_t - self.window_sec
        while self.buf_est and self.buf_est[0].t < lim:
            self.buf_est.popleft()
        while self.buf_gt and self.buf_gt[0].t < lim:
            self.buf_gt.popleft()


    def _associate(self) -> List[Tuple[PoseStamped, PoseStamped]]:
        if not self.buf_est or not self.buf_gt:
            return []
        gt_ts = np.fromiter((p.t for p in self.buf_gt), dtype=float)
        pairs = []
        for e in self.buf_est:
            i = np.searchsorted(gt_ts, e.t)
            cand = []
            if 0 <= i-1 < len(gt_ts): cand.append((abs(gt_ts[i-1]-e.t), i-1))
            if 0 <= i   < len(gt_ts): cand.append((abs(gt_ts[i]  -e.t), i))
            if not cand: continue
            dt, j = min(cand, key=lambda x: x[0])
            if dt <= self.assoc_max_dt:
                pairs.append((e, self.buf_gt[j]))
        return pairs


    def _compute_alignment(self, pairs: List[Tuple[PoseStamped, PoseStamped]]):
        if not pairs:
            return
        if self.align_mode == 'first':
            if not self.have_first:
                self.first_T_est = pairs[0][0].T
                self.first_T_gt = pairs[0][1].T
                self.have_first = True
            R, t, s = align_first_frame(self.first_T_est, self.first_T_gt)
        elif self.align_mode == 'se2_xyyaw':
            # XY + Yaw 정렬(첫 프레임 기준 yaw/xy 오프셋 보정)
            if not self.have_first:
                self.first_T_est = pairs[0][0].T
                self.first_T_gt = pairs[0][1].T
                self.have_first = True
            x_e, y_e, yaw_e = se2_xyyaw_from_T(self.first_T_est)
            x_g, y_g, yaw_g = se2_xyyaw_from_T(self.first_T_gt)
            c, s_ = math.cos(yaw_g - yaw_e), math.sin(yaw_g - yaw_e)
            R = np.array([[c, -s_, 0],
                          [s_,  c, 0],
                          [0,   0, 1]], dtype=float)
            t = np.array([x_g, y_g, 0.0]) - R @ np.array([x_e, y_e, 0.0])
            s = 1.0
        else:  # 'se3_umeyama'
            A = np.array([p[0].T[:3, 3] for p in pairs])
            B = np.array([p[1].T[:3, 3] for p in pairs])
            R, t, s = umeyama_se3(A, B, with_scale=bool(self.align_with_scale))

        self.aligned_R, self.aligned_t, self.aligned_s = R, t, s

    def _apply_alignment(self, T_est: np.ndarray) -> np.ndarray:
        T = T_est.copy()
        T[:3, 3] = self.aligned_s * (self.aligned_R @ T[:3, 3]) + self.aligned_t
        T[:3, :3] = self.aligned_R @ T[:3, :3]
        return T

    def _compute_ate_rmse(self, pairs) -> Optional[float]:
        if len(pairs) < 3:
            return None
        errs = []
        for e, g in pairs:
            Te = self._apply_alignment(e.T)
            pe = Te[:3, 3]
            pg = g.T[:3, 3]
            errs.append(np.linalg.norm(pg - pe))
        if not errs:
            return None
        return float(math.sqrt(np.mean(np.square(errs))))

    def _compute_rte_yaw_deg_rmse(self) -> Optional[float]:
        if not self.buf_est or not self.buf_gt:
            return None

        est_ts = np.fromiter((p.t for p in self.buf_est), dtype=float)
        gt_ts  = np.fromiter((p.t for p in self.buf_gt), dtype=float)
        if len(est_ts) < 2 or len(gt_ts) < 2:
            return None

        # 다운샘플
        est_idx_iter = range(0, len(est_ts), max(1, self.eval_stride))

        yaw_errs_deg: List[float] = []
        for i in est_idx_iter:
            t1 = est_ts[i]
            t2 = t1 + self.rte_delta_sec

            # est에서 t1, t2의 최근접
            ie2 = np.searchsorted(est_ts, t2)
            ie1 = i
            cand_e2 = []
            if 0 <= ie2-1 < len(est_ts): cand_e2.append((abs(est_ts[ie2-1]-t2), ie2-1))
            if 0 <= ie2   < len(est_ts): cand_e2.append((abs(est_ts[ie2]  -t2), ie2))
            if not cand_e2: continue
            de2, j_e2 = min(cand_e2, key=lambda x: x[0])
            de1 = 0.0  # ie1은 정확히 인덱스 i

            if de1 > self.assoc_max_dt or de2 > self.assoc_max_dt:
                continue

            # gt에서 t1, t2의 최근접
            ig1 = np.searchsorted(gt_ts, t1)
            ig2 = np.searchsorted(gt_ts, t2)
            cand_g1 = []
            cand_g2 = []
            if 0 <= ig1-1 < len(gt_ts): cand_g1.append((abs(gt_ts[ig1-1]-t1), ig1-1))
            if 0 <= ig1   < len(gt_ts): cand_g1.append((abs(gt_ts[ig1]  -t1), ig1))
            if 0 <= ig2-1 < len(gt_ts): cand_g2.append((abs(gt_ts[ig2-1]-t2), ig2-1))
            if 0 <= ig2   < len(gt_ts): cand_g2.append((abs(gt_ts[ig2]  -t2), ig2))
            if not cand_g1 or not cand_g2: continue
            dg1, j_g1 = min(cand_g1, key=lambda x: x[0])
            dg2, j_g2 = min(cand_g2, key=lambda x: x[0])
            if dg1 > self.assoc_max_dt or dg2 > self.assoc_max_dt:
                continue

            # 행렬들 구성
            Te1 = self._apply_alignment(self.buf_est[ie1].T)
            Te2 = self._apply_alignment(self.buf_est[j_e2].T)
            Tg1 = self.buf_gt[j_g1].T
            Tg2 = self.buf_gt[j_g2].T

            E_est = np.linalg.inv(Te1) @ Te2
            E_gt  = np.linalg.inv(Tg1) @ Tg2
            E_err = np.linalg.inv(E_gt) @ E_est

            yaw_err_rad = yaw_from_R(E_err[:3, :3])
            yaw_errs_deg.append(math.degrees(yaw_err_rad))

        if not yaw_errs_deg:
            return None
        return float(math.sqrt(np.mean(np.square(yaw_errs_deg))))

    def _latest_stamp(self) -> Optional[float]:
        t_est = self.buf_est[-1].t if self.buf_est else None
        t_gt  = self.buf_gt[-1].t  if self.buf_gt  else None
        if t_est is None and t_gt is None:
            return None
        if t_est is None: return t_gt
        if t_gt  is None: return t_est
        return max(t_est, t_gt)

    # ---------- 주기 계산/퍼블리시 ----------
    def timer_compute_and_publish(self):
        self._purge_old()

        pairs = self._associate()
        if not pairs:
            return

        # 최신 시각(시뮬레이션/백 기준 스탬프)
        now = self._latest_stamp()
        if now is None:
            return

        # 정렬 갱신
        if self.align_mode in ('se3_umeyama', 'se2_xyyaw'):
            need_realign = (
                self.last_align_time < 0.0 or
                (self.recompute_alignment_every > 0.0 and
                 (now - self.last_align_time) >= self.recompute_alignment_every)
            )
            if need_realign:
                self._compute_alignment(pairs)
                self.last_align_time = now
        elif self.align_mode == 'first' and not self.have_first:
            self._compute_alignment(pairs)
            # first 모드에선 last_align_time은 굳이 안 씀

        ate = self._compute_ate_rmse(pairs)
        yaw_rmse_deg = self._compute_rte_yaw_deg_rmse()

        if ate is not None:
            self.pub_ate.publish(Float64(data=ate))
        if yaw_rmse_deg is not None:
            self.pub_rte_yaw.publish(Float64(data=yaw_rmse_deg))

        if self.csv_w is not None and ate is not None and yaw_rmse_deg is not None:
            self.csv_w.writerow([now, ate, yaw_rmse_deg])
            self.csv_f.flush()


    # ---------- 종료 처리 ----------
    def destroy_node(self):
        if self.csv_f:
            try:
                self.csv_f.close()
            except:
                pass

        if self.save_plots and self.output_csv:
            try:
                import matplotlib.pyplot as plt
                import pandas as pd
                base = os.path.splitext(self.output_csv)[0]

                # ===== 1) 시간 추이 RMSE 플롯 (기존) =====
                df = pd.read_csv(self.output_csv)
                # ATE
                if 'ATE_RMSE' in df.columns and len(df) > 0:
                    plt.figure(figsize=(7, 3.5))
                    t0 = float(df['t'].iloc[0])
                    plt.plot(df['t'] - t0, df['ATE_RMSE'], label='ATE RMSE [m]')
                    plt.xlabel('time [s]'); plt.ylabel('ATE RMSE [m]'); plt.title('ATE RMSE')
                    plt.grid(True); plt.legend(loc='best'); plt.tight_layout()
                    plt.savefig(base + "_ATE.png", dpi=140); plt.close()
                # yaw-only RTE
                if 'RTE_yaw_deg_RMSE' in df.columns and len(df) > 0:
                    plt.figure(figsize=(7, 3.5))
                    t0 = float(df['t'].iloc[0])
                    plt.plot(df['t'] - t0, df['RTE_yaw_deg_RMSE'], label='Yaw-only RTE RMSE [deg]')
                    plt.xlabel('time [s]'); plt.ylabel('Yaw RTE RMSE [deg]'); plt.title('Yaw-only RTE RMSE')
                    plt.grid(True); plt.legend(loc='best'); plt.tight_layout()
                    plt.savefig(base + "_RTE_yaw.png", dpi=140); plt.close()

                # ===== 2) 궤적 XY 플롯 (신규) =====
                # 버퍼에서 현재 윈도우 구간 궤적을 가져와 그림
                if len(self.buf_gt) >= 2 and len(self.buf_est) >= 2:
                    # 정렬 파라미터 최신화가 안 되어 있을 수 있으므로, 가능한 경우 한번 더 정렬 갱신
                    pairs = self._associate()
                    if pairs:
                        # align_mode가 se3_umeyama/se2_xyyaw면 주기적 재정렬 로직과 동일하게 갱신
                        # first 모드는 최초 한 번만 정렬
                        if self.align_mode in ('se3_umeyama', 'se2_xyyaw'):
                            self._compute_alignment(pairs)
                        elif self.align_mode == 'first' and not self.have_first:
                            self._compute_alignment(pairs)

                    # GT 궤적
                    gt_xy = np.array([[p.T[0, 3], p.T[1, 3]] for p in self.buf_gt], dtype=float)
                    # EST 궤적 (정렬 적용)
                    est_xy = np.array([[self._apply_alignment(p.T)[0, 3],
                                        self._apply_alignment(p.T)[1, 3]] for p in self.buf_est], dtype=float)

                    plt.figure(figsize=(6.5, 6.0))
                    if len(gt_xy) > 0:
                        plt.plot(gt_xy[:, 0], gt_xy[:, 1], '-', label='GT (/odom)', linewidth=2.0, color='tab:blue')
                    if len(est_xy) > 0:
                        plt.plot(est_xy[:, 0], est_xy[:, 1], '-', label='EST aligned (/odom_airio)', linewidth=2.0, color='tab:orange')

                    # 시작점/끝점 마커
                    if len(gt_xy) > 0:
                        plt.scatter(gt_xy[0, 0], gt_xy[0, 1], marker='o', s=40, color='tab:blue', label='GT start')
                        plt.scatter(gt_xy[-1, 0], gt_xy[-1, 1], marker='x', s=60, color='tab:blue', label='GT end')
                    if len(est_xy) > 0:
                        plt.scatter(est_xy[0, 0], est_xy[0, 1], marker='o', s=40, color='tab:orange', label='EST start')
                        plt.scatter(est_xy[-1, 0], est_xy[-1, 1], marker='x', s=60, color='tab:orange', label='EST end')

                    plt.gca().set_aspect('equal', adjustable='box')
                    plt.xlabel('X [m]'); plt.ylabel('Y [m]')
                    plt.title('Trajectory (XY) — GT vs EST (aligned)')
                    plt.grid(True); plt.legend(loc='best'); plt.tight_layout()
                    plt.savefig(base + "_TRAJ_XY.png", dpi=160); plt.close()

            except Exception as e:
                self.get_logger().warn(f"plot save failed: {e}")

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ATERTEEvaluator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except:
            pass


if __name__ == '__main__':
    main()
