from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import sys
import numpy as np
import torch
from pyhocon import ConfigFactory, ConfigTree

# =====================
# Data Structures
# =====================

@dataclass
class ImuData:
    # Angular velocity [rad/s]
    wx: float; wy: float; wz: float
    # Linear acceleration [m/s^2]
    ax: float; ay: float; az: float
    # Orientation (pass-through; not used by AIR-IMU)
    qx: float; qy: float; qz: float; qw: float
    # Timestamp [sec]
    stamp: float


@dataclass
class CorrectedImu:
    # corrected signals (body frame)
    wx: float; wy: float; wz: float
    ax: float; ay: float; az: float
    # orientation passthrough
    qx: float; qy: float; qz: float; qw: float
    stamp: float
    # uncertainties (per-axis variances; always filled)
    gyro_var: Tuple[float, float, float]
    acc_var: Tuple[float, float, float]
    # optional bias estimates (if provided by model)
    gyro_bias: Optional[Tuple[float, float, float]] = None
    acc_bias: Optional[Tuple[float, float, float]] = None


# =====================
# AirIMU Wrapper (NDArray Ring-Buffer)
# =====================

class AirIMUCorrector:
    """
    AIR-IMU 래퍼 (ndarray 고정 크기 원형 버퍼):
      - 입력: add_sample(ImuData) 가 즉시 numpy 배열에 기록
      - 추론: 시퀀스(seqlen) 및 interval 충족 + stride 조건에서만 실행
      - 출력: 보정된 IMU + 분산(gyro_var/acc_var) + (가능시) 바이어스 추정치
      - 실패/미탑재 시: pass-through + 기본 분산(항상 채움)
    """

    def __init__(
        self,
        airimu_root: str,
        ckpt_path: Optional[str] = None,
        conf_path: Optional[str] = None,
        device: str = "cuda",
        seqlen: int = 100,
        stride: int = 5,
        use_pass_through_if_fail: bool = True,
    ):
        self.device = device
        self.seqlen = int(seqlen)
        self.stride = max(1, int(stride))

        # runtime
        self.ready = False
        self.model = None
        self.conf: Optional[ConfigTree] = None
        self.interval = 9  # repo 기본값
        self._step = 0

        # ---------- ndarray ring buffers ----------
        L = self.seqlen
        self._gyr  = np.zeros((L, 3), dtype=np.float32)
        self._acc  = np.zeros((L, 3), dtype=np.float32)
        self._quat = np.zeros((L, 4), dtype=np.float32)
        self._t    = np.zeros((L,),   dtype=np.float32)
        self._idx  = 0
        self._size = 0

        self.last_quat  = (0.0, 0.0, 0.0, 1.0)
        self.last_stamp = 0.0

        # latest result cache
        self.last_output: Optional[CorrectedImu] = None

        # repo path
        if airimu_root and Path(airimu_root).exists():
            if str(airimu_root) not in sys.path:
                sys.path.append(str(airimu_root))

        try:
            torch.set_num_threads(1)
            from model.code import CodeNet  # AirIMU repo

            # conf
            self.conf = ConfigFactory.parse_file(str(conf_path)) if (conf_path and Path(conf_path).exists()) else None
            try:
                if isinstance(self.conf, ConfigTree) and self.conf.get("interval", None) is not None:
                    self.interval = int(self.conf.get("interval"))
            except Exception:
                pass

            # model
            self.model = CodeNet(self.conf.train).to(device) if self.conf is not None else CodeNet().to(device)

            # load ckpt
            if ckpt_path and Path(ckpt_path).exists():
                try:
                    state = torch.load(ckpt_path, map_location=device, weights_only=True)  # torch>=2.0
                except TypeError:
                    state = torch.load(ckpt_path, map_location=device)
                sd = state.get("state_dict", state)
                self.model.load_state_dict(sd, strict=False)

            self.model.eval()
            self.ready = True
            print(f"[AirIMU] ready | device={self.device}, seqlen={self.seqlen}, interval={self.interval}, stride={self.stride}")

        except Exception:
            if not use_pass_through_if_fail:
                raise
            import traceback
            print("[AirIMU] load failed, pass-through mode:")
            traceback.print_exc()
            self.ready = False

        # 기본 fallback 보정 파라미터 (없으면 0/1)
        self.bias_g  = (0.0, 0.0, 0.0)
        self.bias_a  = (0.0, 0.0, 0.0)
        self.scale_g = (1.0, 1.0, 1.0)
        self.scale_a = (1.0, 1.0, 1.0)

        # 기본 분산 (항상 사용 가능)
        self.default_gyro_var = (0.02**2, 0.02**2, 0.02**2)  # (rad/s)^2
        self.default_acc_var  = (0.20**2, 0.20**2, 0.20**2)  # (m/s^2)^2

    # ----------------- utils -----------------
    @staticmethod
    def _clamp_var3(v: Tuple[float, float, float], eps: float = 1e-10) -> Tuple[float, float, float]:
        a, b, c = v
        def safe(x): 
            if not np.isfinite(x): return eps
            return max(float(x), eps)
        return (safe(a), safe(b), safe(c))

    def fallback_correct_vals(self, x: ImuData) -> Tuple[float, float, float, float, float, float]:
        wx = (x.wx - self.bias_g[0]) * self.scale_g[0]
        wy = (x.wy - self.bias_g[1]) * self.scale_g[1]
        wz = (x.wz - self.bias_g[2]) * self.scale_g[2]
        ax = (x.ax - self.bias_a[0]) * self.scale_a[0]
        ay = (x.ay - self.bias_a[1]) * self.scale_a[1]
        az = (x.az - self.bias_a[2]) * self.scale_a[2]
        return wx, wy, wz, ax, ay, az

    # ----------------- ring buffer helpers -----------------
    def _push(self, wxa: Tuple[float, float, float, float, float, float], stamp: float, quat: Tuple[float,float,float,float]):
        i = self._idx
        wx, wy, wz, ax, ay, az = wxa
        self._gyr[i, :]  = (wx, wy, wz)
        self._acc[i, :]  = (ax, ay, az)
        self._quat[i, :] = quat
        self._t[i]       = stamp
        self._idx  = (i + 1) % self.seqlen
        self._size = min(self._size + 1, self.seqlen)
        self.last_quat  = quat
        self.last_stamp = stamp

    def _chron_view(self):
        T = self._size
        if T <= 0:
            return self._gyr[:0], self._acc[:0], self._quat[:0], self._t[:0], 0
        end = self._idx
        start = (self._idx - T) % self.seqlen
        if start < end:
            return (self._gyr[start:end], self._acc[start:end], self._quat[start:end], self._t[start:end], T)
        # wrapped
        gyr  = np.concatenate((self._gyr[start:],  self._gyr[:end]),  axis=0)
        acc  = np.concatenate((self._acc[start:],  self._acc[:end]),  axis=0)
        quat = np.concatenate((self._quat[start:], self._quat[:end]), axis=0)
        tt   = np.concatenate((self._t[start:],    self._t[:end]),    axis=0)
        return gyr, acc, quat, tt, T

    # ----------------- public API -----------------

    def add_sample(self, x: ImuData) -> None:
        """IMU 콜백에서 호출: ndarray ring-buffer에 직접 누적"""
        corr_vals = self.fallback_correct_vals(x)  # 기본은 항등(바이어스/스케일=0/1)
        self._push(corr_vals, x.stamp, (x.qx, x.qy, x.qz, x.qw))
        self._step += 1

    def _make_tensors_infer_only(self) -> Optional[Dict[str, Any]]:
        """CodeNet.inference는 acc/gyro 시퀀스만 필요. interval 정렬은 모델 내부 처리."""
        gyr, acc, _, _, T = self._chron_view()
        if T < self.seqlen or T < (self.interval + 1):
            return None
        gyro = torch.from_numpy(gyr).unsqueeze(0).to(self.device)  # (1,T,3)
        acc_t = torch.from_numpy(acc).unsqueeze(0).to(self.device) # (1,T,3)
        return {"acc": acc_t, "gyro": gyro}

    def _infer_once(self) -> Optional[CorrectedImu]:
        data = self._make_tensors_infer_only()
        if data is None:
            return None

        try:
            with torch.inference_mode():
                inf = self.model.inference(data)  
            # corrected = raw[:, interval:, :] + correction
            corr_acc  = inf['correction_acc']          # (1, T-interval, 3)
            corr_gyro = inf['correction_gyro']         # (1, T-interval, 3)
            acc_tail  = data['acc'][:,  self.interval:, :]   # (1, T-interval, 3)
            gyr_tail  = data['gyro'][:, self.interval:, :]   # (1, T-interval, 3)

            corrected_acc  = acc_tail  + corr_acc
            corrected_gyro = gyr_tail  + corr_gyro

            # 마지막 프레임 추출
            ax, ay, az = corrected_acc[:,  -1, :].float().squeeze(0).tolist()
            wx, wy, wz = corrected_gyro[:, -1, :].float().squeeze(0).tolist()

            # 분산(가능 시) 추출
            gvar = None
            avar = None
            cov_state = inf.get('cov_state', None)
            if isinstance(cov_state, dict):
                gcv = cov_state.get('gyro_cov', None)
                acv = cov_state.get('acc_cov',  None)
                if torch.is_tensor(gcv) and gcv.numel() >= 3:
                    g = gcv[:, -1, :].reshape(-1, 3)[-1].float().tolist()
                    gvar = (float(g[0]), float(g[1]), float(g[2]))
                if torch.is_tensor(acv) and acv.numel() >= 3:
                    a = acv[:, -1, :].reshape(-1, 3)[-1].float().tolist()
                    avar = (float(a[0]), float(a[1]), float(a[2]))

            if gvar is None: gvar = self.default_gyro_var
            if avar is None: avar = self.default_acc_var
            gvar = self._clamp_var3(gvar)
            avar = self._clamp_var3(avar)

            return CorrectedImu(
                wx, wy, wz, ax, ay, az,
                self.last_quat[0], self.last_quat[1], self.last_quat[2], self.last_quat[3],
                float(self.last_stamp),
                gyro_var=gvar, acc_var=avar,
                gyro_bias=None, acc_bias=None
            )
        except Exception as e:
            print(f"[AirIMU] inference-only failed, fallback used: {e}")
            return None

    def correct_latest(self) -> Optional[CorrectedImu]:
        """
        inference만 수행하여 보정된 IMU 한 프레임을 반환.
        길이/stride 미충족 시에는 캐시 또는 pass-through 반환.
        항상 gyro_var/acc_var가 채워진 CorrectedImu를 돌려준다(가능 시).
        """
        do_infer = (
            self.ready and (self.model is not None) and
            (self._size >= self.seqlen) and
            (self._size >= self.interval + 1) and
            (self._step % self.stride == 0)
        )

        if do_infer:
            out = self._infer_once()
            if out is not None:
                self.last_output = out

        # 캐시 있으면 반환
        if self.last_output is not None:
            return self.last_output

        # 최소 pass-through
        if self._size == 0:
            return None
        i = (self._idx - 1) % self.seqlen
        wx, wy, wz = self._gyr[i, :].tolist()
        ax, ay, az = self._acc[i, :].tolist()
        qx, qy, qz, qw = self.last_quat
        return CorrectedImu(
            wx, wy, wz, ax, ay, az,
            qx, qy, qz, qw, float(self.last_stamp),
            gyro_var=self._clamp_var3(self.default_gyro_var),
            acc_var=self._clamp_var3(self.default_acc_var)
        )
