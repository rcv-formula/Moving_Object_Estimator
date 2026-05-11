import torch
from torch import nn
import math
from airio_imu_odometry.tools import _quat_xyzw_to_R, _ensure_bt1

class VelocityIntegrator(nn.Module):
    """
    네트워크 속도(v)와 dt로 위치 적분.
    - frame='world'  : v가 세계좌표(odom) 기준 → 그대로 적분
    - frame='body'   : v가 바디(base_link) 기준 → orient(쿼터니언/행렬)로 월드 변환 후 적분
    - method='euler' 또는 'trapezoid' (사다리꼴)
    실시간 step() / 배치 forward() 모두 지원.
    """
    def __init__(self, init_pos=None, frame='world', method='euler', dtype=torch.float64, device='cpu'):
        super().__init__()
        self.frame = frame  # 'world' or 'body'
        self.method = method  # 'euler' or 'trapezoid'
        self.dtype = dtype
        self.device = torch.device(device)
        if init_pos is None:
            init_pos = torch.zeros(3, dtype=dtype, device=self.device)
        init_pos = _ensure_bt1(torch.as_tensor(init_pos, dtype=dtype, device=self.device))
        # 내부 상태: (1,1,3)
        self.register_buffer('pos', init_pos.clone(), persistent=False)
        # 이전 스텝 속도 보관(사다리꼴용)
        self.register_buffer('prev_vw', None, persistent=False)

    def reset(self, pos=None):
        if pos is None:
            pos = torch.zeros(3, dtype=self.dtype, device=self.device)
        self.pos = _ensure_bt1(torch.as_tensor(pos, dtype=self.dtype, device=self.device))
        self.prev_vw = None

    def _to_world_vel(self, vel, orient=None):
        """
        vel: (B,T,3) 바디 or 월드
        orient:
          - frame='world'  : 무시
          - frame='body'   : 필요. (B,T,4)[xyzw] 또는 (B,T,3,3) 회전.
        반환: v_world (B,T,3)
        """
        if self.frame == 'world':
            return vel
        # body → world
        if orient is None:
            raise ValueError("frame='body'이면 orient(쿼터니언(xyzw) 또는 회전행렬)가 필요합니다.")
        if orient.dim() == 3 and orient.shape[-1] == 4:
            R = _quat_xyzw_to_R(orient)  # (B,T,3,3)
        elif orient.dim() == 4 and orient.shape[-2:] == (3,3):
            R = orient
        else:
            raise ValueError(f"orient shape invalid: {orient.shape}. Expect (B,T,4) or (B,T,3,3)")
        v_world = torch.einsum('btij,btj->bti', R, vel)
        return v_world

    def forward(self, dt, vel, init_state=None, orient=None):
        """
        배치 적분:
          dt:  (B,T,1) 또는 브로드캐스팅 가능한 형태
          vel: (B,T,3)  (frame 기준은 self.frame)
          orient: (B,T,4[xyzw]) 또는 (B,T,3,3)  (frame='body'일 때 필요)
          init_state: {'pos': (B,1,3)} 없으면 내부 self.pos 사용
        반환: {'pos': (B,T,3)}
        """
        dt  = _ensure_bt1(torch.as_tensor(dt,  dtype=self.dtype, device=self.device))
        vel = _ensure_bt1(torch.as_tensor(vel, dtype=self.dtype, device=self.device))
        B, T = dt.shape[0], dt.shape[1]
        if init_state is None or 'pos' not in init_state:
            pos0 = self.pos.expand(B,1,3)  # 내부 상태에서 시작
        else:
            pos0 = _ensure_bt1(torch.as_tensor(init_state['pos'], dtype=self.dtype, device=self.device))

        # 프레임 변환
        if self.frame == 'body':
            if orient is None:
                raise ValueError("orient is required when frame='body'")
            orient = _ensure_bt1(torch.as_tensor(orient, dtype=self.dtype, device=self.device))
            v_w = self._to_world_vel(vel, orient)
        else:
            v_w = vel

        # 적분
        if self.method == 'trapezoid' and T >= 2:
            # v0, v1, ... v_{T-1}
            v_prev = torch.cat([v_w[:, :1, :], v_w[:, :-1, :]], dim=1)
            inc = 0.5 * (v_prev + v_w) * dt  # (B,T,3)
        else:
            inc = v_w * dt

        pos = pos0 + torch.cumsum(inc, dim=1)  # (B,T,3)
        # 내부 상태 최신화
        self.pos = pos[..., -1:, :]  # (B,1,3)
        return {'pos': pos}

    @torch.no_grad()
    def step(self, dt, vel, orient=None):
        """
        실시간 1 스텝 적분:
          dt: scalar or (1,1,1)
          vel: (3,) 또는 (1,1,3)
          orient: (4,) or (3,3) or (1,1,4)/(1,1,3,3) — frame='body'일 때 필요
        반환: (pos_now):(3,) (world)
        """
        dt  = _ensure_bt1(torch.as_tensor(dt,  dtype=self.dtype, device=self.device))
        vel = _ensure_bt1(torch.as_tensor(vel, dtype=self.dtype, device=self.device))
        if self.frame == 'body':
            if orient is None:
                raise ValueError("orient is required when frame='body'")
            if isinstance(orient, torch.Tensor) and orient.dim()==1 and orient.numel()==4:
                orient = orient.view(1,1,4)
            elif isinstance(orient, torch.Tensor) and orient.dim()==2 and orient.shape==(3,3):
                orient = orient.view(1,1,3,3)
            else:
                orient = _ensure_bt1(torch.as_tensor(orient, dtype=self.dtype, device=self.device))
            v_w = self._to_world_vel(vel, orient)  # (1,1,3)
        else:
            v_w = vel

        if self.method == 'trapezoid' and self.prev_vw is not None:
            inc = 0.5 * (self.prev_vw + v_w) * dt
        else:
            inc = v_w * dt
        self.pos = self.pos + inc  # (1,1,3)
        self.prev_vw = v_w.clone()
        return self.pos.view(3)
