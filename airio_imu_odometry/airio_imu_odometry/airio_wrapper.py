from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Deque, Tuple, Dict, Any
from collections import deque
from pathlib import Path
import sys
import numpy as np
import torch
import traceback
import importlib
import pickle
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
    AIR-IO 속도 예측 래퍼
    - 입력: raw IMU 시퀀스(acc, gyro, dt) + rot(orientation)
    - 출력: body-frame velocity (vx, vy, vz)
    - 실패/미탑재: (0,0,0) 반환 + 원인 로그
    """
    def __init__(
        self,
        airio_root: str,
        ckpt_path: Optional[str] = None,
        conf_path: Optional[str] = None,
        device: str = "cuda",
        seqlen: int = 10,
        interval: int = 5,                 # conf의 sampling과 매칭
        use_downsample: bool = False,
        airio_model: str = "CodeNetMotionwithRot",       
        # rot 설정
        rot_type: str = "",             
        rot_path: Optional[str] = None,     # orientation_output.pickle 경로:: "/root/moving_object_estimator_ws/airio_imu_odometry/config/airio/orientation_output.pickle"
    ):
        self.device = device
        self.seqlen = int(seqlen)
        self.interval = int(interval)
        self.airio_model = airio_model

        # 버퍼
        self.buf_t: Deque[float] = deque(maxlen=self.seqlen)
        self.buf_gyr: Deque[Tuple[float,float,float]] = deque(maxlen=self.seqlen)
        self.buf_acc: Deque[Tuple[float,float,float]] = deque(maxlen=self.seqlen)
        self.buf_quat: Deque[Tuple[float,float,float,float]] = deque(maxlen=self.seqlen)

    
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

        # rot 설정
        self.rot_type = (rot_type or "none").lower()
        self.rot_path = rot_path
        self.rot_seq: Optional[torch.Tensor] = None  # (1, T, 4)

        # 모델
        self.model = None
        self.ready = False
        self.model_class_name = None
        self.ckpt_loaded = False

        try:
            torch.set_num_threads(1)

            # ----------------------------
            # 1) Config 로드 (pyhocon)
            # ----------------------------
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
            # 외부에서 self.device(str) 를 받는다고 가정. 예: "cuda:0" 또는 "cpu"
            conf_obj.train.device = self.device
            conf_obj['device'] = self.device

            # ----------------------------
            # 2) 네트워크 생성 (메인 스크립트 방식과 동일)
            # ----------------------------
            airio_root = Path(airio_root).resolve() 
            airio_model = import_pkg_as_alias("airio_model", airio_root / "model")
            net_dict = airio_model.net_dict

            self.model = net_dict[conf_obj.train.network](conf_obj.train).to(self.device).float()
            self.model_class_name = f"net_dict['{conf_obj.train.network}']"

            # ----------------------------
            # 3) 체크포인트 로드 (epoch, model_state_dict 기대)
            # ----------------------------
            if ckpt_path and Path(ckpt_path).exists():
                try:
                    state = torch.load(ckpt_path, map_location=torch.device(self.device), weights_only=True)
                except TypeError:
                    state = torch.load(ckpt_path, map_location=torch.device(self.device))

                if isinstance(state, dict) and "model_state_dict" in state:
                    sd = state["model_state_dict"]
                    ep = state.get("epoch", -1)
                elif isinstance(state, dict) and "state_dict" in state:
                    sd = state["state_dict"]
                    ep = state.get("epoch", -1)
                else:
                    sd = state
                    ep = -1  # epoch 정보 없음

                print(f"[AirIO] loaded state dict {ckpt_path} in epoch {ep}")
                self.model.load_state_dict(sd, strict=False)
                self.ckpt_loaded = True
            else:
                raise KeyError(f"No model loaded {ckpt_path}")

            # rot_type=airimu면 pickle 로드
            if self.rot_type == "airimu":
                if self.rot_path and Path(self.rot_path).exists():
                    try:
                        with open(self.rot_path, "rb") as f:
                            d = pickle.load(f)
                            print (d)
                        rot_arr = None
                        rot_arr = d["imu_4"]["airimu_rot"]
                        rot_arr = np.asarray(rot_arr, dtype=np.float32)
                        print(rot_arr)

                        if rot_arr.ndim == 2 and rot_arr.shape[-1] == 4:
                            rot_arr = rot_arr[None, ...]      # (1, T, 4)
                        if rot_arr.ndim != 3 or rot_arr.shape[-1] != 4:
                            raise ValueError(f"rot shape must be (1,T,4) or (T,4), got {rot_arr.shape}")
                        self.rot_seq = torch.from_numpy(rot_arr).to(self.device)
                        print(f"[AirIO] loaded rot (airimu) from {self.rot_path}, shape={self.rot_seq.shape}")
                    except Exception as e:
                        print(f"[AirIO] ERROR: failed to load orientation pickle: {e}")
                        traceback.print_exc()
                        self.rot_seq = None
                else:
                    print(f"[AirIO] WARN: rot_type=airimu but invalid rot_path: {self.rot_path}")

            self.model.eval()
            self.ready = True
            print(f"[AirIO] ready | cls={self.model_class_name}, device={self.device}, "
                  f"seqlen={self.seqlen}, interval={self.interval}, "
                  f"ckpt_loaded={self.ckpt_loaded}, rot_type={self.rot_type}, rot_loaded={self.rot_seq is not None}")

        except Exception as e:
            print(f"[AirIO] load failed -> pass-through velocity (0,0,0): {e}")
            traceback.print_exc()
            self.ready = False

    
    def add_sample(self, s: RawImuSample):

        self.buf_t.append(s.stamp)
        self.buf_gyr.append((s.wx, s.wy, s.wz))
        self.buf_acc.append((s.ax, s.ay, s.az))
        self.buf_quat.append((s.qx, s.qy, s.qz, s.qw))

    def make_tensors(self) -> Optional[Dict[str, torch.Tensor]]:
        """
        반환:
          - data dict: {"acc": (1,T,3), "gyro": (1,T,3), "dt": (1,T-interval,1)}
          - rot tensor: (1,T,4)
        """
        T = len(self.buf_t)
        if T < self.seqlen or T < self.interval + 1:
            return None
        acc = torch.from_numpy(np.asarray(self.buf_acc, dtype=np.float32))[None].to(self.device)
        gyr = torch.from_numpy(np.asarray(self.buf_gyr, dtype=np.float32))[None].to(self.device)

        # dt
        t_np = np.asarray(self.buf_t, dtype=np.float32)
        diffs = np.diff(t_np).astype(np.float32)
        if diffs.shape[0] < self.interval:
            return None
        dt = torch.from_numpy(diffs[self.interval-1:]).unsqueeze(0).unsqueeze(-1).to(self.device)

        data = {"acc": acc, "gyro": gyr, "dt": dt}
        return data

    def predict_velocity(self, cur_rot) -> Tuple[float,float,float]:
        data = self.make_tensors()
        if data is None:
            return (0.0, 0.0, 0.0)
        if not self.ready or self.model is None:
            return (0.0, 0.0, 0.0)

        try:
            dev   = next(t.device for t in data.values() if torch.is_tensor(t))
            dty   = next(t.dtype  for t in data.values() if torch.is_tensor(t))
            T     = data['acc'].shape[1]  
            qx, qy, qz, qw = [float(v) for v in cur_rot]
            rvec = _quat_xyzw_to_rotvec(qx, qy, qz, qw, device=dev, dtype=dty)  # (1, T, 3)
            rot_feat = rvec.view(1,1,3).repeat(1,self.seqlen,1)

            with torch.inference_mode():
                # Air-IO: forward(self, data, rot=None) → 'net_vel' 우선
                out = self.model(data, rot=rot_feat)
                
            if isinstance(out, dict):
                for k in ["net_vel", "velocity", "vel", "y", "out"]:
                    if k in out and torch.is_tensor(out[k]) and out[k].dim()==3 and out[k].size(-1)>=3:
                        v = out[k][:, -1, :3].float().squeeze(0).tolist()
                        return (float(v[0]), float(v[1]), float(v[2]))
                for v in out.values():
                    if torch.is_tensor(v) and v.dim()==3 and v.size(-1)>=3:
                        vv = v[:, -1, :3].float().squeeze(0).tolist()
                        return (float(vv[0]), float(vv[1]), float(vv[2]))
            elif torch.is_tensor(out):
                if out.dim()==3 and out.size(-1)>=3:
                    v = out[:, -1, :3].float().squeeze(0).tolist()
                    return (float(v[0]), float(v[1]), float(v[2]))
                if out.dim()==2 and out.size(-1)>=3:
                    v = out[-1, :3].float().tolist()
                    return (float(v[0]), float(v[1]), float(v[2]))
        except Exception as e:
            print(f"[AirIO] inference failed -> (0,0,0): {e}")
            traceback.print_exc()

        return (0.0, 0.0, 0.0)
