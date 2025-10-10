#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# === 스레드/BLAS 세팅: torch import 전에 환경변수/스레드 제한 ===
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import qos_profile_sensor_data
from collections import deque

from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry

# 순서 주의: 위에서 env 세팅 후 numpy/torch import
import numpy as np
import threading
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.set_grad_enabled(False)  # 전역 autograd off
import time
import pypose as pp

from airio_imu_odometry.airimu_wrapper import AirIMUCorrector, ImuData
from airio_imu_odometry.airio_wrapper import AirIOWrapper
from airio_imu_odometry.velocity_integrator import VelocityIntegrator
from airio_imu_odometry.tools import _so3_from_xyzw


class AirIoImuOdomNode(Node):
    def __init__(self):
        super().__init__('airio_imu_odometry')

        # === 콜백 그룹 ===
        self.cbgroup_imu   = MutuallyExclusiveCallbackGroup()
        self.cbgroup_timer = MutuallyExclusiveCallbackGroup()  # (남겨두되 사용하지 않음)

        # --- Parameters ---
        self.declare_parameter("airimu_root", "")
        self.declare_parameter("airimu_ckpt", "")
        self.declare_parameter("airimu_conf", "")
        self.declare_parameter("airio_root", "")
        self.declare_parameter("airio_ckpt", "")
        self.declare_parameter("airio_conf", "")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("airimu_seqlen", 10)
        self.declare_parameter("timming_logging_mode", False)
        self.declare_parameter("timming_logging_outputpath", ".")

        airimu_root = self.get_parameter("airimu_root").get_parameter_value().string_value
        airimu_ckpt = self.get_parameter("airimu_ckpt").get_parameter_value().string_value
        airimu_conf = self.get_parameter("airimu_conf").get_parameter_value().string_value
        airio_root  = self.get_parameter("airio_root").get_parameter_value().string_value
        airio_ckpt  = self.get_parameter("airio_ckpt").get_parameter_value().string_value
        airio_conf  = self.get_parameter("airio_conf").get_parameter_value().string_value
        device_str  = self.get_parameter("device").get_parameter_value().string_value
        self.seqlen = int(self.get_parameter("airimu_seqlen").get_parameter_value().integer_value)
        self.TL_out_path = self.get_parameter("timming_logging_outputpath").get_parameter_value().string_value
        self.TL_mode     = bool(self.get_parameter("timming_logging_mode").get_parameter_value().bool_value)

        # --- Init gating ---
        self.initialized = False
        self.init_lock   = threading.Lock()
        self.sample_lock = threading.Lock()  # add_sample 보호
        self.prev_odom   = None
        self.init_state  = {"pos": None, "rot": None, "vel": None, "stamp": None}

        # === Cartographer 기반 재초기화 요청 버퍼 ===
        self._realign_lock = threading.Lock()
        self._realign_req = None  # dict | None  {pos, rot, vel, stamp}

        # --- Modules ---
        self.corrector = AirIMUCorrector(
            airimu_root=airimu_root, ckpt_path=airimu_ckpt, conf_path=airimu_conf,
            device=device_str, seqlen=self.seqlen
        )
        self.airio = AirIOWrapper(
            airio_root=airio_root, ckpt_path=airio_ckpt, conf_path=airio_conf, device=device_str
        )

        # --- Pypose INTEGRATOR ---
        self.pp_integrator = None
        self.last_integrated_stamp = None
        self.gravity = 9.81007
        self.pp_dev = torch.device(device_str)

        # 사전할당 버퍼 (float32, device 상주)
        self._pp_dt  = None  # shape [1,1,1], float32
        self._pp_gyr = None  # shape [1,1,3], float32
        self._pp_acc = None  # shape [1,1,3], float32

        # --- Subs & Pubs ---
        self.create_subscription(
            Imu, '/imu/data_raw', self.imu_callback,
            qos_profile_sensor_data, callback_group=self.cbgroup_imu
        )
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        self.odom_pub     = self.create_publisher(Odometry, '/odom_airio', qos_profile_sensor_data)
        self.filtered_pub = self.create_publisher(Imu,      '/airimu_imu_data', qos_profile_sensor_data)
        self.imu_sec = None
        self.imu_nanosec = None

        # === Timer 삭제 ===
        # 타이머는 사용하지 않고, IMU 콜백에서 seqlen 샘플이 모이면 처리 트리거
        # self.timer = self.create_timer(...)

        # === IMU 콜백에서 처리 트리거 관련 상태 ===
        self.samples_since_proc = 0
        self.proc_lock = threading.Lock()  # 재진입 방지

        if self.corrector.ready:
            self.get_logger().info("AIR-IMU ready.")
        else:
            self.get_logger().warn("AIR-IMU in pass-through mode.")

        if self.airio.ready:
            self.get_logger().info("AIR-IO ready (velocity net).")
        else:
            self.get_logger().warn("AIR-IO in pass-through mode (velocity=0).")

        self.get_logger().info("Waiting for /odom to initialize...")

        # --- Velocity_Integrator ---
        self.net_vel_is_body = True
        self.vel_integ = None  # (CPU/double 유지)

        # --- Timming_Logging ---
        self.airimu_step_t_deque = deque(maxlen=5000)
        self.airimu_rot_step_t_deque = deque(maxlen=5000)
        self.airio_network_step_t_deque = deque(maxlen=5000)
        self.velocity_integrator_step_t_deque = deque(maxlen=5000)
        self.total_t_deque = deque(maxlen=5000)

        # --- ZUPT/드리프트 방지용 상태 ---
        self.zupt_win_sec = 0.3
        self.gyro_thr     = 0.02
        self.acc_thr      = 0.15
        self.deadband_ms  = 5.0
        self.max_dt       = 0.2

        self._imu_hist = deque(maxlen=2000)
        self._last_ego_pos = np.zeros(3, dtype=float)

    # --- 내부 유틸 ---
    def _prepare_pp_buffers(self):
        """IMUPreintegrator 입력 버퍼를 device/float32로 사전할당."""
        self._pp_dt  = torch.zeros((1, 1, 1), dtype=torch.float32, device=self.pp_dev)
        self._pp_gyr = torch.zeros((1, 1, 3), dtype=torch.float32, device=self.pp_dev)
        self._pp_acc = torch.zeros((1, 1, 3), dtype=torch.float32, device=self.pp_dev)

    def _diff_velocity(self, prev: Odometry, curr: Odometry):
        p0, p1 = prev.pose.pose.position, curr.pose.pose.position
        t0 = prev.header.stamp.sec + prev.header.stamp.nanosec * 1e-9
        t1 = curr.header.stamp.sec + curr.header.stamp.nanosec * 1e-9
        dt = t1 - t0
        if dt <= 0.0:
            raise ValueError(f"non-positive dt: {dt}")
        return [(p1.x - p0.x)/dt, (p1.y - p0.y)/dt, (p1.z - p0.z)/dt], t1

    def _is_stationary(self, now_stamp: float) -> bool:
        if not self._imu_hist:
            return False
        win_start = now_stamp - self.zupt_win_sec
        g_vals, a_vals = [], []
        for t, g, a in self._imu_hist:
            if t >= win_start:
                g_vals.append(g); a_vals.append(a)
        if len(g_vals) < 3:
            return False
        g_mean = float(np.mean(g_vals))
        a_mean = float(np.mean(a_vals))
        return (g_mean < self.gyro_thr) and (a_mean < self.acc_thr)

    # === Callbacks ===
    def odom_callback(self, msg: Odometry):
        # 초기화(첫 2프레임) 로직은 그대로 유지
        if not self.initialized:
            with self.init_lock:
                if self.initialized:
                    return
                if self.prev_odom is None:
                    self.prev_odom = msg
                    self.get_logger().info("First /odom buffered. Waiting next /odom for diff-velocity.")
                    return
                try:
                    _, _ = self._diff_velocity(self.prev_odom, msg)
                except Exception as e:
                    self.prev_odom = msg
                    self.get_logger().warn(f"diff-velocity failed; defer init. reason={e}")
                    return

                p = msg.pose.pose.position
                q = msg.pose.pose.orientation
                v = msg.twist.twist.linear
                self.init_state = {
                    "pos": [p.x, p.y, p.z],
                    "rot": [q.x, q.y, q.z, q.w],
                    "vel": [v.x, v.y, v.z],
                    "stamp": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                }

                try:
                    if hasattr(self.airio, "set_init_state"):
                        self.airio.set_init_state(self.init_state)
                    if hasattr(self.corrector, "set_init_state"):
                        self.corrector.set_init_state(self.init_state)
                except Exception as e:
                    self.get_logger().warn(f"set_init_state hook failed: {e}")

                # === pp_integrator(float32) 생성 + 버퍼 사전할당 ===
                try:
                    self.pp_dev = torch.device(self.get_parameter("device").get_parameter_value().string_value)
                    pos0 = torch.tensor(self.init_state["pos"], dtype=torch.float32, device=self.pp_dev)
                    vel0 = torch.tensor(self.init_state["vel"], dtype=torch.float32, device=self.pp_dev)
                    qx, qy, qz, qw = self.init_state["rot"]
                    rot0 = _so3_from_xyzw(qx, qy, qz, qw, device=self.pp_dev).float()

                    self.pp_integrator = pp.module.IMUPreintegrator(
                        pos0, rot0, vel0, gravity=float(self.gravity), reset=False
                    ).to(self.pp_dev).float()

                    self._prepare_pp_buffers()
                    self.last_integrated_stamp = None
                except Exception as e:
                    self.get_logger().error(f"IMUPreintegrator init failed: {e}")
                    return

                try:
                    init_pos = torch.tensor(self.init_state["pos"], dtype=torch.float64)  # VelocityIntegrator는 기존대로
                    self.vel_integ = VelocityIntegrator(
                        init_pos, frame=('body' if self.net_vel_is_body else 'world'),
                        method='trapezoid', device='cpu'
                    ).double()
                    self._last_ego_pos = init_pos.numpy().astype(float)
                    self.last_integrated_stamp = None
                except Exception as e:
                    self.get_logger().error(f"VelocityIntegrator init failed: {e}")
                    return

                self.initialized = True
                self.get_logger().info(f"/odom init pos={self.init_state['pos']} vel={self.init_state['vel']}")
                return

        # 초기화 이후에는 재정렬 요청만 큐잉
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        v = msg.twist.twist.linear
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        req = {
            "pos": np.array([p.x, p.y, p.z], dtype=float),
            "rot": np.array([q.x, q.y, q.z, q.w], dtype=float),
            "vel": np.array([v.x, v.y, v.z], dtype=float),
            "stamp": stamp
        }
        with self._realign_lock:
            self._realign_req = req

    def imu_callback(self, msg: Imu):
        if not self.initialized:
            return
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.imu_sec = msg.header.stamp.sec
        self.imu_nanosec = msg.header.stamp.nanosec

        imu_in = ImuData(
            wx=msg.angular_velocity.x, wy=msg.angular_velocity.y, wz=msg.angular_velocity.z,
            ax=msg.linear_acceleration.x, ay=msg.linear_acceleration.y, az=msg.linear_acceleration.z,
            qx=msg.orientation.x, qy=msg.orientation.y, qz=msg.orientation.z, qw=msg.orientation.w,
            stamp=stamp
        )
        # add_sample 보호 (최소 락 구간)
        with self.sample_lock:
            self.corrector.add_sample(imu_in)
            self.airio.add_sample(imu_in)

        # 정지 판정용 히스토리 기록(가벼운 계산)
        gx, gy, gz = msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z
        ax, ay, az = msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z
        gyro_norm = float(np.linalg.norm([gx, gy, gz]))
        acc_norm  = float(np.linalg.norm([ax, ay, az]) - self.gravity)
        self._imu_hist.append((stamp, gyro_norm, abs(acc_norm)))

        # ====== seqlen 샘플마다 처리 트리거 ======
        self.samples_since_proc += 1
        if self.samples_since_proc >= max(1, self.seqlen):
            # 재진입 방지: 바쁘면 스킵하고 다음 기회에 처리
            if self.proc_lock.acquire(False):
                try:
                    self._process_once()
                finally:
                    self.samples_since_proc = 0
                    self.proc_lock.release()

    # === 타이머 대신 호출되는 처리 본문 ===
    def _process_once(self):
        if not self.initialized:
            return

        t0 = time.time()

        # 최신 보정값
        imu_out = self.corrector.correct_latest()
        if imu_out is None:
            return
        airimu_step_t = time.time() - t0

        # === 재정렬 요청 반영 ===
        with self._realign_lock:
            req = self._realign_req
            self._realign_req = None
        if req is not None:
            try:
                self.pp_dev = torch.device(self.get_parameter("device").get_parameter_value().string_value)

                pos0 = torch.tensor(req["pos"], dtype=torch.float32, device=self.pp_dev)
                vel0 = torch.tensor(req["vel"], dtype=torch.float32, device=self.pp_dev)
                qx, qy, qz, qw = req["rot"].tolist()
                rot0 = _so3_from_xyzw(qx, qy, qz, qw, device=self.pp_dev).float()

                self.pp_integrator = pp.module.IMUPreintegrator(
                    pos0, rot0, vel0, gravity=float(self.gravity), reset=False
                ).to(self.pp_dev).float()

                self._prepare_pp_buffers()

                self.vel_integ = VelocityIntegrator(
                    pos0.double(), frame=('body' if self.net_vel_is_body else 'world'),
                    method='trapezoid', device='cpu'
                ).double()
                self._last_ego_pos = pos0.detach().cpu().numpy().astype(float)

                self.last_integrated_stamp = req["stamp"]

                self.init_state = {
                    "pos": req["pos"].tolist(),
                    "rot": req["rot"].tolist(),
                    "vel": req["vel"].tolist(),
                    "stamp": req["stamp"]
                }
                try:
                    if hasattr(self.airio, "set_init_state"):
                        self.airio.set_init_state(self.init_state)
                    if hasattr(self.corrector, "set_init_state"):
                        self.corrector.set_init_state(self.init_state)
                except Exception as e:
                    self.get_logger().warn(f"set_init_state hook failed during realign: {e}")
            except Exception as e:
                self.get_logger().error(f"Realign failed: {e}")

        # dt 가드
        if self.last_integrated_stamp is not None and imu_out.stamp <= self.last_integrated_stamp + 1e-12:
            return
        if self.last_integrated_stamp is None:
            self.last_integrated_stamp = imu_out.stamp
            return
        dt = max(1e-6, imu_out.stamp - self.last_integrated_stamp)
        if dt > self.max_dt:
            self.last_integrated_stamp = imu_out.stamp
            return
        self.last_integrated_stamp = imu_out.stamp

        # === PyPose 한 스텝 (float32 + 사전할당 버퍼 + inference_mode) ===
        t1 = time.time()
        try:
            with torch.inference_mode():
                # 새 텐서 생성 없이 값만 갱신
                self._pp_dt[0, 0, 0]  = float(dt)
                self._pp_gyr[0, 0, 0] = float(imu_out.wx)
                self._pp_gyr[0, 0, 1] = float(imu_out.wy)
                self._pp_gyr[0, 0, 2] = float(imu_out.wz)
                self._pp_acc[0, 0, 0] = float(imu_out.ax)
                self._pp_acc[0, 0, 1] = float(imu_out.ay)
                self._pp_acc[0, 0, 2] = float(imu_out.az)

                state = self.pp_integrator(
                    init_state=None, dt=self._pp_dt, gyro=self._pp_gyr, acc=self._pp_acc, rot=None
                )

                cur_pos_t = state['pos'][..., -1, :]   # [1,1,3], float32
                cur_vel_t = state['vel'][..., -1, :]   # [1,1,3], float32
                cur_rot_t = state['rot'][..., -1, :]   # [1,1,4], float32

                # 쿼터니언 정규화 (디바이스에서)
                nrm = torch.linalg.norm(cur_rot_t, dim=-1, keepdim=True).clamp_min(1e-9)
                cur_rot_t = cur_rot_t / nrm

                # 퍼블리시 직전에만 CPU로 꺼냄
                cur_pos = cur_pos_t.squeeze(0).squeeze(0).cpu().numpy().astype(float)
                cur_vel = cur_vel_t.squeeze(0).squeeze(0).cpu().numpy().astype(float)
                cur_rot = cur_rot_t.squeeze(0).squeeze(0).cpu().numpy().astype(float)
        except Exception as e:
            self.get_logger().warn(f"IMUPreintegrator step failed: {e}")
            return
        airimu_rot_step_t = time.time() - t1

        # ZUPT: 정지면 속도=0, 적분 스킵 준비
        stationary = self._is_stationary(imu_out.stamp)

        # AIR-IO 속도 예측 (정지면 0)
        t2 = time.time()
        if stationary:
            net_vel = np.array([0.0, 0.0, 0.0], dtype=float)
        else:
            net_vel = np.asarray(self.airio.predict_velocity(cur_rot), dtype=float)
        airio_network_step_t = time.time() - t2

        # 데드밴드
        if np.linalg.norm(net_vel) * 1000.0 < self.deadband_ms:
            net_vel[:] = 0.0

        # 속도 적분 → 위치
        t3 = time.time()
        try:
            if stationary or np.allclose(net_vel, 0.0, atol=1e-9):
                ego_pos = self._last_ego_pos.copy()
            else:
                ego_pos = self.vel_integ.step(dt, net_vel, orient=cur_rot).detach().cpu().numpy()
                self._last_ego_pos = ego_pos.copy()
        except Exception as e:
            self.get_logger().warn(f"Velocity integration failed: {e}")
            return
        velocity_integrator_step_t = time.time() - t3
        total_t = time.time() - t0

        if self.TL_mode and total_t <= 0.1:
            self.airimu_step_t_deque.append(airimu_step_t)
            self.airimu_rot_step_t_deque.append(airimu_rot_step_t)
            self.airio_network_step_t_deque.append(airio_network_step_t)
            self.velocity_integrator_step_t_deque.append(velocity_integrator_step_t)
            self.total_t_deque.append(total_t)

        # republish IMU (보정된 최신 샘플)
        imu_msg = Imu()
        imu_msg.header.stamp.sec = self.imu_sec
        imu_msg.header.stamp.nanosec = self.imu_nanosec
        imu_msg.header.frame_id = "base_link"
        imu_msg.angular_velocity.x = float(imu_out.wx)
        imu_msg.angular_velocity.y = float(imu_out.wy)
        imu_msg.angular_velocity.z = float(imu_out.wz)
        imu_msg.linear_acceleration.x = float(imu_out.ax)
        imu_msg.linear_acceleration.y = float(imu_out.ay)
        imu_msg.linear_acceleration.z = float(imu_out.az)
        imu_msg.orientation.x = float(imu_out.qx)
        imu_msg.orientation.y = float(imu_out.qy)
        imu_msg.orientation.z = float(imu_out.qz)
        imu_msg.orientation.w = float(imu_out.qw)
        self.filtered_pub.publish(imu_msg)

        # publish odom (frame=map)
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = "map"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = float(ego_pos[0])
        odom.pose.pose.position.y = float(ego_pos[1])
        odom.pose.pose.position.z = float(ego_pos[2])
        odom.pose.pose.orientation.x = float(cur_rot[0])
        odom.pose.pose.orientation.y = float(cur_rot[1])
        odom.pose.pose.orientation.z = float(cur_rot[2])
        odom.pose.pose.orientation.w = float(cur_rot[3])
        odom.twist.twist.linear.x = float(net_vel[0])
        odom.twist.twist.linear.y = float(net_vel[1])
        odom.twist.twist.linear.z = float(net_vel[2])
        self.odom_pub.publish(odom)

    # (옵션) 타이밍 저장: 필요 시만 사용
    def save_timings(self):
        import matplotlib.pyplot as plt  # 필요 시에만 임포트(지연 로딩)
        timings = {
            "AIR-IMU": list(self.airimu_step_t_deque),
            "AIR-IMU RotStep": list(self.airimu_rot_step_t_deque),
            "AIR-IO Network": list(self.airio_network_step_t_deque),
            "VelocityIntegrator": list(self.velocity_integrator_step_t_deque),
            "Total": list(self.total_t_deque),
        }
        outdir = os.path.dirname(self.TL_out_path) or "."
        prefix = os.path.splitext(os.path.basename(self.TL_out_path))[0] or "timings"
        os.makedirs(outdir, exist_ok=True)
        for name, values in timings.items():
            plt.figure(figsize=(8, 4))
            ms = [v * 1000 for v in values]
            plt.plot(ms, marker='o', markersize=2, linewidth=0.7, label=name)
            if values:
                avg = sum(values) / len(values) * 1000.0
                plt.axhline(avg, linestyle='--', label=f"avg={avg:.3f}ms")
                plt.axhline(min(values) * 1000.0, linestyle=':', label=f"min={min(values)*1000:.3f}ms")
                plt.axhline(max(values) * 1000.0, linestyle=':', label=f"max={max(values)*1000:.3f}ms")
            plt.ylabel("millisecond"); plt.xlabel("#"); plt.title(name)
            plt.legend(loc="upper right"); plt.grid(True)
            safe = name.replace(" ", "_").replace("/", "_")
            plt.tight_layout(); plt.savefig(os.path.join(outdir, f"{prefix}_{safe}.png"), dpi=130); plt.close()


def main(args=None):
    rclpy.init(args=args)
    node = AirIoImuOdomNode()
    try:
        rclpy.spin(node)  # 싱글 스레드
    except KeyboardInterrupt:
        if node.TL_mode:
            node.save_timings()
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
