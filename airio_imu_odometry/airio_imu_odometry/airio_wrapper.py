from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import sys
import numpy as np
import torch
import traceback
from pyhocon import ConfigFactory

from airio_imu_odometry.tools import import_pkg_as_alias, _quat_xyzw_to_rotvec

@dataclass
class RawImuSample:
    wx: float; wy: float; wz: float      # rad/s
    ax: float; ay: float; az: float      # m/s^2
    qx: float; qy: float; qz: float; qw: float
    stamp: float

class AirIOWrapper:
    """
    AIR-IO 속도 예측 래퍼 (ndarray ring-buffer)
    - 입력: raw IMU 시퀀스(acc, gyro, dt)
    - 회전 특징: 호출 시점에 전달되는 cur_rot(= PyPose integrator 결과)만 사용
    - 출력: {"vel": (vx,vy,vz), "eta_v": (σx,σy,σz)}
    - 실패/미탑재: 기본값 반환 + 원인 로그
    """
    def __init__(
        self,
        airio_root: str,
        ckpt_path: Optional[str] = None,
        conf_path: Optional[str] = None,
        device: str = "cuda",
        seqlen: int = 30,
        interval: int = 9,
        use_downsample: bool = False,          # (보존: 필요 시 내부에서 활용)
        airio_model: str = "CodeNetMotionwithRot",
    ):
        self.device = device
        self.seqlen = int(seqlen)
        self.interval = int(interval)
        self.airio_model = airio_model

        # ===== ndarray ring buffers =====
        L = self.seqlen
        self._t    = np.zeros((L,),   dtype=np.float32)
        self._gyr  = np.zeros((L, 3), dtype=np.float32)
        self._acc  = np.zeros((L, 3), dtype=np.float32)
        self._quat = np.zeros((L, 4), dtype=np.float32)  # (실시간 회전입력만 쓰더라도 수집은 유지)
        self._idx  = 0
        self._size = 0

        # 모델
        self.model = None
        self.ready = False
        self.model_class_name = None
        self.ckpt_loaded = False
        # 모델이 불확실도를 내지 않을 때 사용할 기본 표준편차(σ)
        self._default_eta = (0.05, 0.05, 0.05)

        # 경로 추가
        try:
            if airio_root and Path(airio_root).exists():
                if str(airio_root) not in sys.path:
                    sys.path.append(str(airio_root))
            else:
                print(f"[AirIO] WARN: airio_root not exists: {airio_root}")
        except Exception:
            print("[AirIO] failed to append airio_root to sys.path")
            traceback.print_exc()

        try:
            torch.set_num_threads(1)

            # 1) Config 로드 (pyhocon)
            conf_obj = None
            if conf_path and Path(conf_path).exists():
                try:
                    conf_obj = ConfigFactory.parse_file(str(conf_path))
                except Exception as e:
                    print(f"[AirIO] WARN: failed to parse conf {conf_path}: {e}")
                    conf_obj = None
            if conf_obj is None:
                raise RuntimeError("[AirIO] ERROR: conf is required and failed to load.")

            # device 설정
            conf_obj.train.device = self.device
            conf_obj['device'] = self.device

            # 2) 네트워크 생성
            airio_root = Path(airio_root).resolve()
            airio_model = import_pkg_as_alias("airio_model", airio_root / "model")
            net_dict = airio_model.net_dict
            self.model = net_dict[conf_obj.train.network](conf_obj.train).to(self.device).float()
            self.model_class_name = f"net_dict['{conf_obj.train.network}']"

            # 3) 체크포인트 로드
            if ckpt_path and Path(ckpt_path).exists():
                try:
                    state = torch.load(ckpt_path, map_location=torch.device(self.device), weights_only=True)
                except TypeError:
                    state = torch.load(ckpt_path, map_location=torch.device(self.device))

                if isinstance(state, dict) and "model_state_dict" in state:
                    sd = state["model_state_dict"]; ep = state.get("epoch", -1)
                elif isinstance(state, dict) and "state_dict" in state:
                    sd = state["state_dict"]; ep = state.get("epoch", -1)
                else:
                    sd = state; ep = -1  # epoch 정보 없음

                print(f"[AirIO] loaded state dict {ckpt_path} in epoch {ep}")
                self.model.load_state_dict(sd, strict=False)
                self.ckpt_loaded = True
            else:
                raise KeyError(f"No model loaded {ckpt_path}")

            self.model.eval()
            self.ready = True
            print(f"[AirIO] ready | cls={self.model_class_name}, device={self.device}, "
                  f"seqlen={self.seqlen}, interval={self.interval}, ckpt_loaded={self.ckpt_loaded}")

        except Exception as e:
            print(f"[AirIO] load failed -> pass-through defaults: {e}")
            traceback.print_exc()
            self.ready = False

    # ========= ring buffer helpers =========
    def _push(self, s: RawImuSample):
        i = self._idx
        self._t[i]    = float(s.stamp)
        self._gyr[i]  = (s.wx, s.wy, s.wz)
        self._acc[i]  = (s.ax, s.ay, s.az)
        self._quat[i] = (s.qx, s.qy, s.qz, s.qw)   # 수집만; 실시간 특성은 predict에서 주입
        self._idx  = (i + 1) % self.seqlen
        self._size = min(self._size + 1, self.seqlen)

    def _chron_view(self):
        T = self._size
        if T <= 0:
            return self._gyr[:0], self._acc[:0], self._quat[:0], self._t[:0], 0
        end = self._idx
        start = (self._idx - T) % self.seqlen
        if start < end:
            return self._gyr[start:end], self._acc[start:end], self._quat[start:end], self._t[start:end], T
        # wrapped
        gyr = np.concatenate((self._gyr[start:], self._gyr[:end]), axis=0)
        acc = np.concatenate((self._acc[start:], self._acc[:end]), axis=0)
        quat = np.concatenate((self._quat[start:], self._quat[:end]), axis=0)
        tt = np.concatenate((self._t[start:], self._t[:end]), axis=0)
        return gyr, acc, quat, tt, T

    # ========= public API =========
    def add_sample(self, s: RawImuSample):
        """리스트/데크 사용 없이, 즉시 ndarray ring-buffer에 기록"""
        self._push(s)

    def make_tensors(self) -> Optional[Dict[str, torch.Tensor]]:
        """
        반환:
          - data dict: {"acc": (1,T,3), "gyro": (1,T,3), "dt": (1,T-interval,1)}
        """
        gyr_np, acc_np, _, tt_np, T = self._chron_view()
        if T < self.seqlen or T < self.interval + 1:
            return None

        acc = torch.from_numpy(acc_np).unsqueeze(0).to(self.device)   # (1,T,3)
        gyr = torch.from_numpy(gyr_np).unsqueeze(0).to(self.device)   # (1,T,3)

        diffs = np.diff(tt_np).astype(np.float32)
        if diffs.shape[0] < self.interval:
            return None
        dt = torch.from_numpy(diffs[self.interval-1:]).unsqueeze(0).unsqueeze(-1).to(self.device)  # (1,T-interval,1)

        return {"acc": acc, "gyro": gyr, "dt": dt}

    def predict_velocity(self, cur_rot):
        """cur_rot: (qx, qy, qz, qw) → {"vel": (vx,vy,vz), "eta_v": (σx,σy,σz)}"""
        data = self.make_tensors()
        if data is None or (not self.ready) or (self.model is None):
            return {"vel": (0.0, 0.0, 0.0), "eta_v": self._default_eta}

        try:
            dev = next(t.device for t in data.values() if torch.is_tensor(t))
            dty = next(t.dtype  for t in data.values() if torch.is_tensor(t))
            Tlen = int(data['acc'].shape[1])

            # 현재 자세 → so(3) 로트벡 변환 → 윈도 길이에 맞춰 반복
            qx, qy, qz, qw = [float(v) for v in cur_rot]
            rvec = _quat_xyzw_to_rotvec(qx, qy, qz, qw, device=dev, dtype=dty)  # (3,)
            rot_feat = rvec.view(1,1,3).repeat(1, Tlen, 1)                      # (1, T, 3)

            with torch.inference_mode():
                out = self.model(data, rot=rot_feat)

            # --- 출력 파싱 (vel 우선, 그다음 eta_v/uncertainty 선택적으로) ---
            vel = None
            eta = None
            if isinstance(out, dict):
                for k in ["net_vel", "velocity", "vel", "y", "out"]:
                    v = out.get(k, None)
                    if torch.is_tensor(v) and v.dim() == 3 and v.size(-1) >= 3:
                        vv = v[:, -1, :3].float().squeeze(0).tolist()
                        vel = (float(vv[0]), float(vv[1]), float(vv[2]))
                        break
                if vel is None:
                    for v in out.values():
                        if torch.is_tensor(v) and v.dim() == 3 and v.size(-1) >= 3:
                            vv = v[:, -1, :3].float().squeeze(0).tolist()
                            vel = (float(vv[0]), float(vv[1]), float(vv[2]))
                            break
                # 불확실도 키(있으면 사용)
                for k in ["eta_v", "sigma_v", "std_v", "uncert"]:
                    sv = out.get(k, None)
                    if torch.is_tensor(sv):
                        if sv.dim() == 3 and sv.size(-1) >= 3:
                            ss = sv[:, -1, :3].float().squeeze(0).abs().tolist()
                            eta = (float(ss[0]), float(ss[1]), float(ss[2]))
                            break
            elif torch.is_tensor(out):
                if out.dim() == 3 and out.size(-1) >= 3:
                    v = out[:, -1, :3].float().squeeze(0).tolist()
                    vel = (float(v[0]), float(v[1]), float(v[2]))
                elif out.dim() == 2 and out.size(-1) >= 3:
                    v = out[-1, :3].float().tolist()
                    vel = (float(v[0]), float(v[1]), float(v[2]))

            if vel is None:
                vel = (0.0, 0.0, 0.0)
            if eta is None:
                eta = self._default_eta
            return {"vel": vel, "eta_v": eta}

        except Exception as e:
            print(f"[AirIO] inference failed -> defaults: {e}")
            traceback.print_exc()
            return {"vel": (0.0, 0.0, 0.0), "eta_v": self._default_eta}
