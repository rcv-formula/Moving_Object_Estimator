from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import sys
import numpy as np

def quat_conj(q: np.ndarray) -> np.ndarray:
    x,y,z,w = q
    return np.array([-x,-y,-z, w], dtype=float)

def quat_log(q: np.ndarray) -> np.ndarray:
    """
    Log: SO(3) quaternion -> so(3) vector (axis*angle)
    q = [vx, vy, vz, w], unit quaternion
    return phi in R^3 (rotation vector)
    """
    x, y, z, w = q
    w = np.clip(w, -1.0, 1.0)
    s2 = x*x + y*y + z*z
    if s2 < 1e-16:
        return np.zeros(3, dtype=float)
    s = np.sqrt(s2)
    angle = 2.0 * np.arctan2(s, w)  # in [0, 2pi)
    axis = np.array([x, y, z], dtype=float) / s
    return axis * angle

def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return q / n

def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    return np.array([x, y, z, w], dtype=float)

def axis_angle_to_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    n = np.linalg.norm(a)
    if n < 1e-12 or abs(angle) < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    a = a / n
    s = np.sin(angle / 2.0)
    c = np.cos(angle / 2.0)
    return np.array([a[0]*s, a[1]*s, a[2]*s, c], dtype=float)

def so3_exp(w: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    a = w / theta
    s = np.sin(theta/2.0)
    c = np.cos(theta/2.0)
    return np.array([a[0]*s, a[1]*s, a[2]*s, c], dtype=float)

def R_from_quat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    R = np.array([
        [1 - 2*(yy+zz), 2*(xy - wz),     2*(xz + wy)    ],
        [2*(xy + wz),     1 - 2*(xx+zz), 2*(yz - wx)    ],
        [2*(xz - wy),     2*(yz + wx),   1 - 2*(xx+yy)  ]
    ], dtype=float)
    return R

@dataclass
class ImuSample:
    wx: float; wy: float; wz: float
    ax: float; ay: float; az: float
    stamp: float

@dataclass
class EkfParams:
    gyro_noise: float = 0.02       # [rad/s]   (기본 표준편차)
    acc_noise: float  = 0.20       # [m/s^2]   (기본 표준편차)
    gyro_bias_rw: float = 0.0    # [rad/s^2]^0.5  (랜덤워크 표준편차)
    acc_bias_rw:  float = 0.0     # [m/s^3]^0.5    (랜덤워크 표준편차)
    gravity: float = 9.81007

class AirIOEKFWrapper:
    """EKF wrapper.
    Tries to load Air-IO repo EKF if available; otherwise uses a light 15-state EKF.
    State: [p(3), v(3), q(xyzw), bg(3), ba(3)], covariance on [p v theta bg ba] (15x15).

    add_imu(...)가 per-sample gyro/acc 분산(variance)을 입력으로 받아
    전파 잡음 공분산 Q에 반영하도록 확장.
    """
    def __init__(self, airio_root: str, use_repo: bool = True, params: Optional[EkfParams] = None):
        self.params = params or EkfParams()
        self.use_repo = False
        self.repo_ekf = None
        self.repo = None

        self.p = np.zeros(3, dtype=float)
        self.v = np.zeros(3, dtype=float)
        self.q = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        self.bg = np.zeros(3, dtype=float)
        self.ba = np.zeros(3, dtype=float)
        self.P = np.eye(15, dtype=float) * 1e-3

        self.last_t: Optional[float] = None

        if use_repo and airio_root:
            try:
                root = Path(airio_root)
                if root.exists():
                    if str(root) not in sys.path:
                        sys.path.append(str(root))
                    candidates = [
                        ("EKF", "ekf"), ("EKF", "inekf"),
                        ("EKF/src", "ekf"), ("EKF/src", "inekf"),
                        ("EKF", None), ("EKF/src", None)
                    ]
                    for subdir, modname in candidates:
                        mod_path = root / subdir if subdir is not None else root
                        if mod_path.exists():
                            if str(mod_path) not in sys.path:
                                sys.path.append(str(mod_path))
                            try:
                                import importlib
                                m = importlib.import_module(modname if modname else "EKF")
                                for cname in dir(m):
                                    if "EKF" in cname.upper() and callable(getattr(m, cname)):
                                        self.repo_ekf = getattr(m, cname)
                                        self.use_repo = True
                                        break
                                if self.use_repo:
                                    break
                            except Exception:
                                continue
            except Exception:
                pass

    def set_init_state(self, state: Dict[str, Any]):
        self.p = np.array(state["pos"], dtype=float).reshape(3)
        self.v = np.array(state["vel"], dtype=float).reshape(3)
        self.q = quat_normalize(np.array(state["rot"], dtype=float).reshape(4))
        self.last_t = float(state.get("stamp", 0.0))

        self.P = np.eye(15, dtype=float) * 1e-3
        self.P[0:3,0:3] *= 1e-4
        self.P[3:6,3:6] *= 1e-3
        self.P[6:9,6:9] *= 1e-3
        self.P[9:12,9:12] *= 1e-5
        self.P[12:15,12:15] *= 1e-4

        if self.use_repo and callable(self.repo_ekf):
            try:
                self.repo = self.repo_ekf()
                for name in ["set_state", "initialize", "reset"]:
                    if hasattr(self.repo, name):
                        try:
                            getattr(self.repo, name)(
                                p=self.p.copy(), v=self.v.copy(), q=self.q.copy(),
                                bg=self.bg.copy(), ba=self.ba.copy(), t=self.last_t
                            )
                            break
                        except Exception:
                            continue
            except Exception:
                self.use_repo = False
                self.repo = None

    # --------- Internal propagate with optional per-sample variances ---------
    def _propagate_internal(self, imu: ImuSample,
                            gyro_var: Optional[Tuple[float,float,float]] = None,
                            acc_var:  Optional[Tuple[float,float,float]]  = None):
        if self.last_t is None:
            self.last_t = imu.stamp
            return
        dt = float(imu.stamp - self.last_t)
        if dt <= 0.0 or dt > 0.2:
            self.last_t = imu.stamp
            return

        g = np.array([0.0, 0.0, -self.params.gravity], dtype=float)
        w = np.array([imu.wx, imu.wy, imu.wz], dtype=float) - self.bg
        a = np.array([imu.ax, imu.ay, imu.az], dtype=float) - self.ba

        # Orientation integration
        theta = np.linalg.norm(w * dt)
        if theta > 0.0:
            axis = (w * dt) / theta
            s = np.sin(theta/2.0); c = np.cos(theta/2.0)
            dq = np.array([axis[0]*s, axis[1]*s, axis[2]*s, c], dtype=float)
        else:
            dq = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        self.q = quat_normalize(quat_mul(self.q, dq))

        # Position/Velocity integration
        Rwb = R_from_quat(self.q)
        acc_world = Rwb @ a + g
        self.p = self.p + self.v * dt + 0.5 * acc_world * dt * dt
        self.v = self.v + acc_world * dt

        # --------- Process noise (use per-sample variances if provided) ---------
        if gyro_var is not None and len(gyro_var) == 3:
            q_gyro = np.diag(np.asarray(gyro_var, dtype=float))  # already variance
        else:
            q_gyro = (self.params.gyro_noise**2) * np.eye(3)
        if acc_var is not None and len(acc_var) == 3:
            q_acc = np.diag(np.asarray(acc_var, dtype=float))    # already variance
        else:
            q_acc  = (self.params.acc_noise**2)  * np.eye(3)

        q_bg_rw = (self.params.gyro_bias_rw**2) * np.eye(3)
        q_ba_rw = (self.params.acc_bias_rw**2)  * np.eye(3)

        Qd = np.zeros((12,12), dtype=float)
        Qd[0:3,0:3]     = q_gyro * dt
        Qd[3:6,3:6]     = q_acc  * dt
        Qd[6:9,6:9]     = q_bg_rw * dt
        Qd[9:12,9:12]   = q_ba_rw * dt

        # --------- Linearized dynamics matrices ---------
        F = np.eye(15, dtype=float)
        F[0:3,3:6] = np.eye(3) * dt

        ax, ay, az = (Rwb @ a).tolist()
        Gv_theta = np.array([[0,  az, -ay],
                             [-az, 0,  ax],
                             [ay, -ax, 0]], dtype=float)
        F[3:6,6:9] = -Gv_theta * dt
        F[3:6,12:15] = -Rwb * dt

        G = np.zeros((15,12), dtype=float)
        G[6:9,0:3]    = -np.eye(3)    # gyro noise to theta
        G[3:6,3:6]    = Rwb           # acc noise to v
        G[9:12,6:9]   = np.eye(3)     # gyro bias RW
        G[12:15,9:12] = np.eye(3)     # acc bias RW

        self.P = F @ self.P @ F.T + G @ Qd @ G.T
        self.last_t = imu.stamp

    # --------- Public API ---------
    def add_imu(self, imu: ImuSample,
                gyro_var: Optional[Tuple[float,float,float]] = None,
                acc_var:  Optional[Tuple[float,float,float]]  = None):
        """EKF propagation with optional per-sample gyro/acc variances."""
        # Try repo EKF first (if it supports extra args)
        if self.use_repo and (self.repo is not None):
            # Best-effort: try a few common method names with optional kwargs
            for name in ["propagate", "predict", "step_imu", "update_imu", "process_imu"]:
                if hasattr(self.repo, name):
                    try:
                        getattr(self.repo, name)(
                            wx=float(imu.wx), wy=float(imu.wy), wz=float(imu.wz),
                            ax=float(imu.ax), ay=float(imu.ay), az=float(imu.az),
                            stamp=float(imu.stamp),
                            gyro_var=gyro_var, acc_var=acc_var
                        )
                        return
                    except TypeError:
                        # Method exists but doesn't accept variances -> try without
                        try:
                            getattr(self.repo, name)(
                                wx=float(imu.wx), wy=float(imu.wy), wz=float(imu.wz),
                                ax=float(imu.ax), ay=float(imu.ay), az=float(imu.az),
                                stamp=float(imu.stamp)
                            )
                            return
                        except Exception:
                            continue
                    except Exception:
                        continue
        # Internal fallback
        self._propagate_internal(imu, gyro_var=gyro_var, acc_var=acc_var)

    def update_velocity_body(self, vel_b: Tuple[float,float,float], R_meas: Optional[np.ndarray] = None):
        if R_meas is None:
            R_meas = np.eye(3, dtype=float) * 0.05**2

        Rwb = R_from_quat(self.q)
        z_w = Rwb @ np.asarray(vel_b, dtype=float).reshape(3)
        H = np.zeros((3,15), dtype=float)
        H[:,3:6] = np.eye(3)
        y = z_w - self.v
        S = H @ self.P @ H.T + R_meas
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ y
        self.P = (np.eye(15) - K @ H) @ self.P

        self.p += dx[0:3]
        self.v += dx[3:6]
        dtheta = dx[6:9]
        self.bg += dx[9:12]
        self.ba += dx[12:15]

        theta = np.linalg.norm(dtheta)
        if theta > 0.0:
            axis = dtheta / theta
            s = np.sin(theta/2.0); c = np.cos(theta/2.0)
            dq = np.array([axis[0]*s, axis[1]*s, axis[2]*s, c], dtype=float)
        else:
            dq = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        self.q = quat_normalize(quat_mul(self.q, dq))

    def get_state(self) -> Dict[str, Any]:
        return {
            "pos": self.p.copy(),
            "vel": self.v.copy(),
            "rot": self.q.copy(),
            "bg": self.bg.copy(),
            "ba": self.ba.copy(),
            "P": self.P.copy(),
            "stamp": float(self.last_t if self.last_t is not None else 0.0)
        }
    # --- AirIOEKFWrapper 클래스 내부에 추가 ---
    def update_pose_world_adaptive(self,
                               p_meas: np.ndarray,
                               q_meas: np.ndarray,
                               R_p: np.ndarray,
                               R_theta: np.ndarray,
                               chi2_thresh: float = 22.46,
                               alpha_max: float = 50.0,
                               skip_thresh: Optional[float] = None) -> float:
        """
        월드 좌표계 포즈 측정으로 EKF 업데이트(위치+자세, 6D).
        - 혁신 r = [p_meas - p;  Log(q_meas * conj(q_pred))]  (자세는 회전벡터 residual)
        - 게이팅: lam = rᵀ S⁻¹ r  (dof=6 가정). lam > chi2_thresh면 R를 인플레이트.
          skip_thresh가 주어지고 lam > skip_thresh면 업데이트를 스킵.
        - 반환: lam (마할라노비스 거리)
        """
        # 상태/공분산
        p = self.p
        q = self.q

        # 6x15 H 구성: [ I3 0 I3 0 0 ] on [p v theta bg ba] 순서
        H = np.zeros((6, 15), dtype=float)
        H[0:3, 0:3] = np.eye(3)  # position wrt p
        H[3:6, 6:9] = np.eye(3)  # orientation (small-angle) wrt theta

        # 측정 잔차 r
        rp = np.asarray(p_meas, float).reshape(3) - p.reshape(3)
        # 회전 잔차: q_err = q_meas * conj(q_pred)
        q_err = quat_mul(np.asarray(q_meas, float).reshape(4), quat_conj(q))
        rth = quat_log(q_err)  # so(3) residual (3)

        r = np.hstack([rp, rth])  # (6,)

        # 측정 공분산 R (6x6)
        R = np.zeros((6, 6), dtype=float)
        R[0:3, 0:3] = np.asarray(R_p, float)
        R[3:6, 3:6] = np.asarray(R_theta, float)

        # 혁신 공분산 S, 마할라노비스 lam
        S = H @ self.P @ H.T + R
        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # 수치 문제 시 작은 eye 더해 안정화
            Sinv = np.linalg.inv(S + 1e-9 * np.eye(6))
        lam = float(r.T @ Sinv @ r)

        # 게이팅/인플레이트 or 스킵
        if skip_thresh is not None and lam > float(skip_thresh):
            # 스킵: 업데이트 없이 lam만 반환
            return lam

        if lam > float(chi2_thresh):
            alpha = min(alpha_max, lam / float(chi2_thresh))
            R = R * alpha
            S = H @ self.P @ H.T + R
            try:
                Sinv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                Sinv = np.linalg.inv(S + 1e-9 * np.eye(6))

        # 칼만 이득/업데이트
        K = self.P @ H.T @ Sinv
        dx = K @ r  # [dp, dv, dtheta, dbg, dba]
        I_KH = np.eye(15) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T  # Joseph form이 수치적으로 더 안전

        # 상태 보정
        self.p += dx[0:3]
        self.v += dx[3:6]
        dtheta = dx[6:9]
        self.bg += dx[9:12]
        self.ba += dx[12:15]

        # 쿼터니언 보정
        theta = np.linalg.norm(dtheta)
        if theta > 0.0:
            axis = dtheta / theta
            s = np.sin(theta/2.0); c = np.cos(theta/2.0)
            dq = np.array([axis[0]*s, axis[1]*s, axis[2]*s, c], dtype=float)
        else:
            dq = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        self.q = quat_normalize(quat_mul(self.q, dq))

        return lam