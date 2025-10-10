import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from collections import deque

from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry

from airio_imu_odometry.airimu_wrapper import AirIMUCorrector, ImuData
from airio_imu_odometry.airio_wrapper import AirIOWrapper
from airio_imu_odometry.velocity_integrator import VelocityIntegrator
from airio_imu_odometry.tools import _so3_from_xyzw

import numpy as np
import threading
import torch
import time
import os
import matplotlib.pyplot as plt
import pypose as pp


class AirIoImuOdomNode(Node):
    def __init__(self):
        super().__init__('airio_imu_odometry')

        # === 콜백 그룹 ===
        self.cbgroup_imu   = MutuallyExclusiveCallbackGroup()
        self.cbgroup_timer = MutuallyExclusiveCallbackGroup()

        # --- Parameters ---
        self.declare_parameter("airimu_root", "")
        self.declare_parameter("airimu_ckpt", "")
        self.declare_parameter("airimu_conf", "")
        self.declare_parameter("airio_root", "")
        self.declare_parameter("airio_ckpt", "")
        self.declare_parameter("airio_conf", "")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("airimu_seqlen", 200)
        self.declare_parameter("publish_rate", 30.0)
        self.declare_parameter("timming_logging_mode", False)
        self.declare_parameter("timming_logging_outputpath", ".")

        airimu_root = self.get_parameter("airimu_root").get_parameter_value().string_value
        airimu_ckpt = self.get_parameter("airimu_ckpt").get_parameter_value().string_value
        airimu_conf = self.get_parameter("airimu_conf").get_parameter_value().string_value
        airio_root  = self.get_parameter("airio_root").get_parameter_value().string_value
        airio_ckpt  = self.get_parameter("airio_ckpt").get_parameter_value().string_value
        airio_conf  = self.get_parameter("airio_conf").get_parameter_value().string_value
        device      = self.get_parameter("device").get_parameter_value().string_value
        seqlen      = int(self.get_parameter("airimu_seqlen").get_parameter_value().integer_value)
        self.TL_out_path = self.get_parameter("timming_logging_outputpath").get_parameter_value().string_value
        self.TL_mode     = bool(self.get_parameter("timming_logging_mode").get_parameter_value().bool_value)
        self.pub_hz      = float(self.get_parameter("publish_rate").get_parameter_value().double_value)

        # --- Init gating ---
        self.initialized = False
        self.init_lock   = threading.Lock()
        self.sample_lock = threading.Lock()  # add_sample 보호
        self.prev_odom   = None
        self.init_state  = {"pos": None, "rot": None, "vel": None, "stamp": None}

        # --- Modules ---
        self.corrector = AirIMUCorrector(
            airimu_root=airimu_root, ckpt_path=airimu_ckpt, conf_path=airimu_conf,
            device=device, seqlen=seqlen
        )
        self.airio = AirIOWrapper(
            airio_root=airio_root, ckpt_path=airio_ckpt, conf_path=airio_conf, device=device
        )

        # --- Pypose INTEGRATOR ---
        self.pp_integrator = None
        self.last_integrated_stamp = None
        self.gravity = 9.81007
        self.pp_dev = None

        # --- Subs & Pubs ---
        self.create_subscription(Imu, '/imu/data_raw', self.imu_callback, 1000,
                                 callback_group=self.cbgroup_imu)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        self.odom_pub     = self.create_publisher(Odometry, '/odom_airio', 10)
        self.filtered_pub = self.create_publisher(Imu, '/airimu_imu_data', 10)
        self.imu_sec = None
        self.imu_nanosec = None

        # --- Timer ---
        period = 1.0 / max(1.0, self.pub_hz)
        self.timer = self.create_timer(period, self.on_timer, callback_group=self.cbgroup_timer)

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
        self.vel_integ = None

        # --- Timming_Logging ---
        self.airimu_step_t_deque = deque(maxlen=5000)
        self.airimu_rot_step_t_deque = deque(maxlen=5000)
        self.airio_network_step_t_deque = deque(maxlen=5000)
        self.velocity_integrator_step_t_deque = deque(maxlen=5000)
        self.total_t_deque = deque(maxlen=5000)

        # --- ZUPT/드리프트 방지용 상태 ---
        self.zupt_win_sec = 0.3     # 정지 판정 윈도우 길이 [s]
        self.gyro_thr     = 0.02    # [rad/s] 평균 절댓값 임계
        self.acc_thr      = 0.15    # [m/s^2] (|a|-g) 평균 임계
        self.deadband_ms  = 5.0     # [mm/s] 속도 데드밴드
        self.max_dt       = 0.2     # [s] 비정상 큰 dt 스킵

        self._imu_hist = deque(maxlen=2000)           # (stamp, |gyro|, |acc|-g)
        self._last_ego_pos = np.zeros(3, dtype=float) # 정지 시 적분 스킵용 백업

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
                g_vals.append(g)
                a_vals.append(a)
        if len(g_vals) < 3:
            return False
        g_mean = float(np.mean(g_vals))
        a_mean = float(np.mean(a_vals))
        return (g_mean < self.gyro_thr) and (a_mean < self.acc_thr)

    # === Callbacks ===
    def odom_callback(self, msg: Odometry):
        if self.initialized:
            return
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

            try:
                device_str = self.get_parameter("device").get_parameter_value().string_value
                self.pp_dev = torch.device(device_str)
                pos0 = torch.tensor(self.init_state["pos"], dtype=torch.float64)
                vel0 = torch.tensor(self.init_state["vel"], dtype=torch.float64)
                qx, qy, qz, qw = self.init_state["rot"]
                rot0 = _so3_from_xyzw(qx, qy, qz, qw, device=self.pp_dev)
                self.pp_integrator = pp.module.IMUPreintegrator(
                    pos0, rot0, vel0, gravity=self.gravity, reset=False
                ).to(self.pp_dev).double()
                self.last_integrated_stamp = None
            except Exception as e:
                self.get_logger().error(f"IMUPreintegrator init failed: {e}")
                return

            try:
                init_pos = torch.tensor(self.init_state["pos"], dtype=torch.float64)
                self.vel_integ = VelocityIntegrator(
                    init_pos, frame=('body' if self.net_vel_is_body else 'world'),
                    method='trapezoid', device='cpu'
                ).double()
                self._last_ego_pos = init_pos.numpy().astype(float)  # 초기 위치 백업
                self.last_integrated_stamp = None
            except Exception as e:
                self.get_logger().error(f"VelocityIntegrator init failed: {e}")
                return

            self.initialized = True
            self.get_logger().info(f"/odom init pos={self.init_state['pos']} vel={self.init_state['vel']}")

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
        # add_sample 보호
        with self.sample_lock:
            self.corrector.add_sample(imu_in)
            self.airio.add_sample(imu_in)

        # 정지 판정용 히스토리 기록
        gx, gy, gz = msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z
        ax, ay, az = msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z
        gyro_norm = float(np.linalg.norm([gx, gy, gz]))
        acc_norm  = float(np.linalg.norm([ax, ay, az]) - self.gravity)
        self._imu_hist.append((stamp, gyro_norm, abs(acc_norm)))

    def on_timer(self):
        if not self.initialized:
            return

        t0 = time.time()

        # (중요) 매 틱마다 init_state를 재주입하지 않음 → 드리프트 축적 완화
        # self.corrector.set_init_state(self.init_state)  # 제거

        imu_out = self.corrector.correct_latest()
        if imu_out is None:
            return
        airimu_step_t = time.time() - t0

        # dt 계산 및 가드
        if self.last_integrated_stamp is not None and imu_out.stamp <= self.last_integrated_stamp + 1e-12:
            return
        if self.last_integrated_stamp is None:
            self.last_integrated_stamp = imu_out.stamp
            return
        dt = max(1e-6, imu_out.stamp - self.last_integrated_stamp)
        if dt > self.max_dt:
            self.get_logger().warn(f"Abnormal dt={dt:.3f}s skipped.")
            self.last_integrated_stamp = imu_out.stamp
            return
        self.last_integrated_stamp = imu_out.stamp

        # IMU 적분 → pose, vel, rot
        t1 = time.time()
        try:
            dev  = self.pp_dev
            dt_t = torch.tensor([[[dt]]], dtype=torch.float64, device=dev)
            gyr  = torch.tensor([[[imu_out.wx, imu_out.wy, imu_out.wz]]], dtype=torch.float64, device=dev)
            acc  = torch.tensor([[[imu_out.ax, imu_out.ay, imu_out.az]]], dtype=torch.float64, device=dev)
            state = self.pp_integrator(init_state=None, dt=dt_t, gyro=gyr, acc=acc, rot=None)

            cur_pos = state['pos'][..., -1, :].detach().cpu().numpy().ravel()
            cur_vel = state['vel'][..., -1, :].detach().cpu().numpy().ravel()
            cur_rot = state['rot'][..., -1, :].detach().cpu().numpy().ravel()

            # 쿼터니언 정규화
            nrm = np.linalg.norm(cur_rot)
            if nrm > 1e-9:
                cur_rot = (cur_rot / nrm).astype(float)
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
        print(net_vel)  
        airio_network_step_t = time.time() - t2

        # 데드밴드(아주 작은 속도는 0으로)
        if np.linalg.norm(net_vel) * 1000.0 < self.deadband_ms:
            net_vel[:] = 0.0

        # 속도 적분 → 위치
        t3 = time.time()
        try:
            if stationary or np.allclose(net_vel, 0.0, atol=1e-9):
                ego_pos = self._last_ego_pos.copy()  # 정지 시 적분 스킵
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
        imu_msg.angular_velocity.x = imu_out.wx
        imu_msg.angular_velocity.y = imu_out.wy
        imu_msg.angular_velocity.z = imu_out.wz
        imu_msg.linear_acceleration.x = imu_out.ax
        imu_msg.linear_acceleration.y = imu_out.ay
        imu_msg.linear_acceleration.z = imu_out.az
        imu_msg.orientation.x = imu_out.qx
        imu_msg.orientation.y = imu_out.qy
        imu_msg.orientation.z = imu_out.qz
        imu_msg.orientation.w = imu_out.qw
        self.filtered_pub.publish(imu_msg)

        # publish odom
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

    def save_timings(self):
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
    executor = MultiThreadedExecutor(num_threads=2)  # imu와 timer 병렬 처리
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        if node.TL_mode:
            node.save_timings()
    finally:
        executor.shutdown()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass
