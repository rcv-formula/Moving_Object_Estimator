from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Deque, Tuple, Dict, Any
from collections import deque
from pathlib import Path
import sys
import numpy as np
import torch
from pyhocon import ConfigFactory, ConfigTree


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
    # uncertainties (per-axis variances; None if unknown)
    gyro_var: Optional[Tuple[float, float, float]] = None
    acc_var: Optional[Tuple[float, float, float]] = None
    # optional bias estimates (if provided by model)
    gyro_bias: Optional[Tuple[float, float, float]] = None
    acc_bias: Optional[Tuple[float, float, float]] = None


class AirIMUCorrector:
    """
    AIR-IMU 래퍼 (확장판):
      - 입력: 단일 IMU 샘플들을 버퍼에 누적(add_sample)
      - 추론: 시퀀스가 충분하고 stride가 도달했을 때만 실행
      - 출력: 보정된 IMU + 공분산(분산) + (가능시) 바이어스 추정치
      - 실패/미탑재 시: pass-through + 합리적 기본 분산
    """

    def __init__(
        self,
        airimu_root: str,
        ckpt_path: Optional[str] = None,
        conf_path: Optional[str] = None,
        device: str = "cuda",
        seqlen: int = 10,
        stride: int = 1,
        use_pass_through_if_fail: bool = True,
    ):
        self.device = device
        self.seqlen = int(seqlen)
        self.stride = max(1, int(stride))

        self._init_state_t = None
        # runtime
        self.ready = False
        self.model = None
        self.conf: Optional[ConfigTree] = None
        self.interval = 9  # default in repo configs
        self._step = 0

        # buffers
        self.buf_gyro: Deque[Tuple[float, float, float]] = deque(maxlen=self.seqlen)
        self.buf_acc:  Deque[Tuple[float, float, float]] = deque(maxlen=self.seqlen)
        self.buf_t:    Deque[float] = deque(maxlen=self.seqlen)
        self.buf_quat: Deque[Tuple[float,float,float,float]] = deque(maxlen=self.seqlen)
        self.last_quat = (0.0, 0.0, 0.0, 1.0)
        self.last_stamp = 0.0

        # latest result cache
        self.last_output: Optional[CorrectedImu] = None

        # add repo to path
        if airimu_root and Path(airimu_root).exists():
            if str(airimu_root) not in sys.path:
                sys.path.append(str(airimu_root))

        try:
            torch.set_num_threads(1)  # 안정성용
            from model.code import CodeNet  # AirIMU repo

            # conf
            self.conf = ConfigFactory.parse_file(str(conf_path)) if (conf_path and Path(conf_path).exists()) else None
            try:
                if isinstance(self.conf, ConfigTree) and self.conf.get("interval", None) is not None:
                    self.interval = int(self.conf.get("interval"))
            except Exception:
                pass

            # model
            self.model = CodeNet(self.conf.train).to(device)

            # load ckpt
            if ckpt_path and Path(ckpt_path).exists():
                state = torch.load(ckpt_path, map_location=device)
                if isinstance(state, dict) and "state_dict" in state:
                    self.model.load_state_dict(state["state_dict"], strict=False)
                else:
                    self.model.load_state_dict(state, strict=False)

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
        self.bias_g = (0.0, 0.0, 0.0)
        self.bias_a = (0.0, 0.0, 0.0)
        self.scale_g = (1.0, 1.0, 1.0)
        self.scale_a = (1.0, 1.0, 1.0)

        # 기본 분산 (fallback): 시작값은 경험치 수준
        self.default_gyro_var = (0.02**2, 0.02**2, 0.02**2)  # (rad/s)^2
        self.default_acc_var  = (0.20**2, 0.20**2, 0.20**2)  # (m/s^2)^2

    # ----------------- utils -----------------
    def set_init_state(self, state: Dict[str, Any]):
        """
        state = {"pos":[x,y,z], "rot":[qx,qy,qz,qw], "vel":[vx,vy,vz], "stamp": float}
        """
        def to1(x): return torch.tensor([x], dtype=torch.float32, device=self.device)

        pos = to1(state["pos"])    # (1,3)
        rot = to1(state["rot"])    # (1,4)
        vel = to1(state["vel"])    # (1,3)
        self._init_state_t = {"pos": pos, "rot": rot, "vel": vel}

    def fallback_correct_vals(self, x: ImuData) -> Tuple[float, float, float, float, float, float]:
        wx = (x.wx - self.bias_g[0]) * self.scale_g[0]
        wy = (x.wy - self.bias_g[1]) * self.scale_g[1]
        wz = (x.wz - self.bias_g[2]) * self.scale_g[2]
        ax = (x.ax - self.bias_a[0]) * self.scale_a[0]
        ay = (x.ay - self.bias_a[1]) * self.scale_a[1]
        az = (x.az - self.bias_a[2]) * self.scale_a[2]
        return wx, wy, wz, ax, ay, az

    def push_buffer(self, wxa: Tuple[float, float, float, float, float, float], stamp: float, quat: Tuple[float,float,float,float]):
        wx, wy, wz, ax, ay, az = wxa
        self.buf_t.append(stamp)
        self.buf_gyro.append((wx, wy, wz))
        self.buf_acc.append((ax, ay, az))
        self.buf_quat.append(quat)
        self.last_quat = quat
        self.last_stamp = stamp

    def make_tensors_infer_only(self) -> Optional[Dict[str, Any]]:
        T = len(self.buf_t)
        # CodeNet.inference는 acc/gyro 시퀀스만 있으면 됨. interval 정렬은 forward에서 처리.
        if T < self.seqlen or T < self.interval + 1:
            return None
        gyro_np = np.asarray(self.buf_gyro, dtype=np.float32)  # (T,3)
        acc_np  = np.asarray(self.buf_acc,  dtype=np.float32)  # (T,3)

        gyro = torch.from_numpy(gyro_np)[None].to(self.device)  # (1,T,3)
        acc  = torch.from_numpy(acc_np )[None].to(self.device)  # (1,T,3)
        return {"acc": acc, "gyro": gyro}

    # ----------------- public API -----------------

    def add_sample(self, x: ImuData) -> None:
        """IMU 콜백에서 호출: 버퍼에 누적만."""
        corr_vals = self.fallback_correct_vals(x)
        self.push_buffer(corr_vals, x.stamp, (x.qx, x.qy, x.qz, x.qw))
        self._step += 1

    def correct_latest(self) -> Optional[CorrectedImu]:
        """
        integrate 없이 inference만 수행하여 보정된 IMU 한 프레임을 반환.
        길이/stride 미충족 시에는 캐시 또는 pass-through 반환.
        """
        do_infer = (
            self.ready and self.model is not None and
            len(self.buf_t) >= self.seqlen and
            len(self.buf_t) >= self.interval + 1 and
            (self._step % self.stride == 0)
        )

        if do_infer:
            data = self.make_tensors_infer_only()
            if data is not None:
                try:
                    with torch.inference_mode():
                        inf = self.model.inference(data)
                    # corrected = raw[:, interval:, :] + correction
                    corr_acc  = inf['correction_acc']          # (1, T-interval, 3)
                    corr_gyro = inf['correction_gyro']         # (1, T-interval, 3)
                    acc_tail  = data['acc'][:, self.interval:, :]   # (1, T-interval, 3)
                    gyr_tail  = data['gyro'][:, self.interval:, :]  # (1, T-interval, 3)

                    corrected_acc  = acc_tail  + corr_acc
                    corrected_gyro = gyr_tail  + corr_gyro

                    # (마지막 프레임)만 추출
                    ax, ay, az = corrected_acc[:, -1, :].float().squeeze(0).tolist()
                    wx, wy, wz = corrected_gyro[:, -1, :].float().squeeze(0).tolist()

                    # 분산(있으면 사용, 없으면 기본값)
                    gvar = None
                    avar = None
                    cov_state = inf.get('cov_state', None)
                    if isinstance(cov_state, dict):
                        if torch.is_tensor(cov_state.get('gyro_cov', None)) and cov_state['gyro_cov'].numel() > 0:
                            g = cov_state['gyro_cov'][:, -1, :].reshape(-1,3)[-1].float().tolist()
                            gvar = (float(g[0]), float(g[1]), float(g[2]))
                        if torch.is_tensor(cov_state.get('acc_cov', None)) and cov_state['acc_cov'].numel() > 0:
                            a = cov_state['acc_cov'][:, -1, :].reshape(-1,3)[-1].float().tolist()
                            avar = (float(a[0]), float(a[1]), float(a[2]))
                    if gvar is None: gvar = self.default_gyro_var
                    if avar is None: avar = self.default_acc_var

                    self.last_output = CorrectedImu(
                        wx, wy, wz, ax, ay, az,
                        self.last_quat[0], self.last_quat[1], self.last_quat[2], self.last_quat[3],
                        self.last_stamp,
                        gyro_var=gvar, acc_var=avar,
                        gyro_bias=None, acc_bias=None
                    )
                except Exception as e:
                    print(f"[AirIMU] inference-only failed, fallback used: {e}")

        # 캐시 있으면 반환
        if self.last_output is not None:
            return self.last_output

        # 최소 pass-through
        if len(self.buf_t) == 0:
            return None
        wx, wy, wz = self.buf_gyro[-1]
        ax, ay, az = self.buf_acc[-1]
        qx, qy, qz, qw = self.last_quat
        return CorrectedImu(
            wx, wy, wz, ax, ay, az,
            qx, qy, qz, qw, self.last_stamp,
            gyro_var=self.default_gyro_var, acc_var=self.default_acc_var
        )
