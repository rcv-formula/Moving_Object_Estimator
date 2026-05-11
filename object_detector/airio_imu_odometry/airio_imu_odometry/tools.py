import importlib.util
import sys
from pathlib import Path
import torch
import pypose as pp

def import_pkg_as_alias(alias: str, pkg_dir: Path):
    """
    pkg_dir: .../model  (안에 __init__.py 가 있어야 함)
    alias:   예) 'airio_model'  -> 이후 'airio_model.code' 처럼 서브모듈도 사용 가능
    """
    init_file = pkg_dir / "__init__.py"
    if not init_file.exists():
        raise FileNotFoundError(f"__init__.py not found in {pkg_dir}")

    # pkg로 인식시키려면 submodule_search_locations 지정
    spec = importlib.util.spec_from_file_location(
        alias, str(init_file),
        submodule_search_locations=[str(pkg_dir)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

def _quat_xyzw_to_rotmat( qx, qy, qz, qw, device, dtype=torch.float64):
    # ROS 쿼터니언 [x,y,z,w] → 회전행렬 (3x3)
    w = torch.tensor(qw, dtype=dtype, device=device)
    x = torch.tensor(qx, dtype=dtype, device=device)
    y = torch.tensor(qy, dtype=dtype, device=device)
    z = torch.tensor(qz, dtype=dtype, device=device)
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    one = torch.ones((), dtype=dtype, device=device)
    two = 2.0 * one
    R = torch.stack([
        torch.stack([one-2*(yy+zz),     two*(xy-wz),     two*(xz+wy)], dim=-1),
        torch.stack([    two*(xy+wz), one-2*(xx+zz),     two*(yz-wx)], dim=-1),
        torch.stack([    two*(xz-wy),     two*(yz+wx), one-2*(xx+yy)], dim=-1),
    ], dim=-2)  # (3,3)
    return R

def _so3_from_xyzw( qx, qy, qz, qw, device, dtype=torch.float64, batch_shape=None):
    R = _quat_xyzw_to_rotmat(qx, qy, qz, qw, device, dtype)
    try:
        return pp.SO3(R)  
    except Exception:
        if hasattr(pp, 'mat2SO3'):
            return pp.mat2SO3(R)
        raise TypeError("Unable to construct pp.SO3 from quaternion on this PyPose version.")

def _quat_xyzw_to_rotvec(qx, qy, qz, qw, device, dtype):
    """
    입력: ROS 쿼터니언 [x,y,z,w]
    출력: torch.tensor(3,) on device,dtype  (so(3) 회전벡터: theta * axis)
    안정성:
      - q와 -q는 동일 → w<0이면 부호 뒤집어 연속성 유지
      - 소각 근사: |v|≈0이면 rotvec ≈ 2*v
    """
    # normalize (안전)
    v = torch.tensor([qx, qy, qz], dtype=dtype, device=device)
    w = torch.tensor(qw, dtype=dtype, device=device)
    norm = torch.linalg.norm(torch.cat([v, w.view(1)]))
    if norm > 0:
        v = v / norm
        w = w / norm

    # q와 -q 동일 → w<0이면 뒤집어서 theta 연속성 확보
    if (w < 0).item():
        v = -v
        w = -w

    s = torch.linalg.norm(v)              # = sin(theta/2)
    eps = torch.tensor(1e-8, dtype=dtype, device=device)

    # 소각 근사: s ~ 0 → rotvec ≈ 2*v
    if s < eps:
        return 2.0 * v

    # 일반: theta = 2*atan2(s, w), rotvec = theta * (v/s)
    theta = 2.0 * torch.atan2(s, w.clamp(min=-1.0, max=1.0))
    axis  = v / s
    return theta * axis

def _ensure_bt1(x):
    """
    입력 x를 (B,T,*) 꼴로 맞춤.
    - 스칼라/1D → (1,1,*)
    - 2D → (1,T,*) 로 가정
    """
    if x is None:
        return None
    if isinstance(x, (float, int)):
        x = torch.tensor(x, dtype=torch.float64)
    if x.dim() == 0:
        return x.view(1,1,1)
    if x.dim() == 1:
        # (3,) 같은 경우는 (1,1,3)
        return x.view(1,1,-1)
    if x.dim() == 2:
        # (T,3) → (1,T,3), (T,1) → (1,T,1)
        return x.unsqueeze(0)
    return x  # 이미 (B,T,*)

def _quat_xyzw_to_R(q):
    """
    q: (B,T,4) or (1,1,4) ... torch (float64 권장)
    반환: R (B,T,3,3)
    """
    B, T, _ = q.shape
    x, y, z, w = q[...,0], q[...,1], q[...,2], q[...,3]
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    one = torch.ones_like(x)
    two = 2.0*one
    R00 = one - two*(yy+zz)
    R01 = two*(xy - wz)
    R02 = two*(xz + wy)
    R10 = two*(xy + wz)
    R11 = one - two*(xx+zz)
    R12 = two*(yz - wx)
    R20 = two*(xz - wy)
    R21 = two*(yz + wx)
    R22 = one - two*(xx+yy)
    R = torch.stack([
        torch.stack([R00,R01,R02], dim=-1),
        torch.stack([R10,R11,R12], dim=-1),
        torch.stack([R20,R21,R22], dim=-1),
    ], dim=-2)  # (B,T,3,3)
    return R