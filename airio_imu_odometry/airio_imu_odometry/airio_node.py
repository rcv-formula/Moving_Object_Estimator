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

from airio_imu_odometry.airimu_wrapper import AirIMUCorrector, ImuData
from airio_imu_odometry.airio_wrapper import AirIOWrapper
from airio_imu_odometry.ekf_wrapper import AirIOEKFWrapper, ImuSample, EkfParams

class AirIoImuOdomNode(Node):
    def __init__(self):
        super().__init__('airio_imu_odometry')

        # === 콜백 그룹 ===
        self.cbgroup_imu   = MutuallyExclusiveCallbackGroup()

        # --- Parameters ---
        self.declare_parameter("airimu_root", "")
        self.declare_parameter("airimu_ckpt", "")
        self.declare_parameter("airimu_conf", "")
        self.declare_parameter("airio_root", "")
        self.declare_parameter("airio_ckpt", "")
        self.declare_parameter("airio_conf", "")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("airimu_seqlen", 100)
        # 퍼블리시는 단순화(권장: 1), airio_every만 유지
        self.declare_parameter("airio_every", 1)
        self.declare_parameter("timming_logging_mode", False)
        self.declare_parameter("timming_logging_outputpath", ".")
        self.declare_parameter("odom_pub_rate", 50.0)  # 추가: /odom_airio 퍼블리시 주기(Hz)

        airimu_root = self.get_parameter("airimu_root").get_parameter_value().string_value
        airimu_ckpt = self.get_parameter("airimu_ckpt").get_parameter_value().string_value
        airimu_conf = self.get_parameter("airimu_conf").get_parameter_value().string_value
        airio_root  = self.get_parameter("airio_root").get_parameter_value().string_value
        airio_ckpt  = self.get_parameter("airio_ckpt").get_parameter_value().string_value
        airio_conf  = self.get_parameter("airio_conf").get_parameter_value().string_value
        device_str  = self.get_parameter("device").get_parameter_value().string_value
        self.seqlen = int(self.get_parameter("airimu_seqlen").get_parameter_value().integer_value)
        self.airio_every = max(1, int(self.get_parameter("airio_every").get_parameter_value().integer_value))
        self.TL_out_path = self.get_parameter("timming_logging_outputpath").get_parameter_value().string_value
        self.TL_mode     = bool(self.get_parameter("timming_logging_mode").get_parameter_value().bool_value)
        self.odom_pub_rate = float(self.get_parameter("odom_pub_rate").get_parameter_value().double_value)
        self._pub_period = 1.0 / max(1e-6, self.odom_pub_rate)
        self._last_pub_ts = 0.0  # 초 단위(ROS time)
        
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
            airimu_root=airimu_root,
            ckpt_path=airimu_ckpt, 
            conf_path=airimu_conf,
            device=device_str, 
            seqlen=self.seqlen
        )
        self.airio = AirIOWrapper(
            airio_root=airio_root,
            ckpt_path=airio_ckpt,
            conf_path=airio_conf,
            device=device_str
        )

        self.ekf = AirIOEKFWrapper(
            airio_root=airio_root,
            use_repo=True,
            params=EkfParams(gyro_noise=0.02, acc_noise=0.20)
        )

        # --- 상수/상태 ---
        self.gravity = 9.81007

        # --- Subs & Pubs ---
        self._wait_for_sim_time(timeout_sec=5.0)
        self.create_subscription(
            Imu, '/imu/data_raw', self.imu_callback,
            qos_profile_sensor_data, callback_group=self.cbgroup_imu
        )
        self.odom_pub     = self.create_publisher(Odometry, '/odom_airio', 10)
        self.filtered_pub = self.create_publisher(Imu,      '/airimu_imu_data', 10)
        self.imu_sec = None
        self.imu_nanosec = None

        # === IMU 콜백에서 처리 트리거 관련 상태 ===
        self.samples_since_proc = 0
        self.proc_lock = threading.Lock()  # 재진입 방지
        self.proc_count = 0                # 퍼블리시/에어아이오 디커플링용 카운터

        # AirIO 예측/불확실도 최근값 (airio_every>1일 때 재사용)
        self.last_net_vel = np.zeros(3, dtype=float)
        self.last_eta_v   = np.array([0.05, 0.05, 0.05], dtype=float)

        if self.corrector.ready:
            self.get_logger().info("AIR-IMU ready.")
        else:
            self.get_logger().warn("AIR-IMU in pass-through mode.")

        if self.airio.ready:
            self.get_logger().info("AIR-IO ready (velocity net).")
        else:
            self.get_logger().warn("AIR-IO in pass-through mode (velocity=0).")

        self.get_logger().info("Waiting for /odom to initialize...")

        try:
            if getattr(self.ekf, "use_repo", False) and getattr(self.ekf, "repo_ekf", None) is not None:
                self.get_logger().info(f"EKF backend: Air-IO repo -> {self.ekf.repo_ekf.__name__}")
            else:
                self.get_logger().warn("EKF backend: internal fallback (built-in 15-state EKF)")
        except Exception as e:
            self.get_logger().warn(f"EKF backend status check failed: {e}")

        # --- Timming_Logging ---
        self.airimu_step_t_deque = deque(maxlen=5000)
        self.airio_network_step_t_deque = deque(maxlen=5000)
        self.total_t_deque = deque(maxlen=5000)

        # --- ZUPT/드리프트 방지용 상태 ---
        # self.zupt_win_sec = 0.3
        # self.gyro_thr     = 0.02
        # self.acc_thr      = 0.15
        self.deadband_ms  = 5.0
        self.max_dt       = 0.2

        self._imu_hist = deque(maxlen=2000)

    # --- 내부 유틸 ---
    def _wait_for_sim_time(self, timeout_sec: float = 5.0):
        """
        use_sim_time=True 가정. /clock 기반 시간이 유효해질 때까지 대기.
        timeout_sec 내에 유효해지지 않으면 경고 로그만 남기고 진행한다.
        """
        start = time.time()
        # now()==0 인 동안 대기
        while rclpy.ok() and self.get_clock().now().nanoseconds == 0:
            # 50ms 단위로 spin_once (콜백/파라미터 이벤트 처리)
            rclpy.spin_once(self, timeout_sec=0.05)
            if timeout_sec is not None and (time.time() - start) > timeout_sec:
                self.get_logger().warn(
                    f"Sim time(/clock) not available after {timeout_sec:.1f}s. "
                    "Continuing anyway — check 'ros2 topic echo /clock' and rosbag '--clock'."
                )
                break

    def _diff_velocity(self, prev: Odometry, curr: Odometry):
        p0, p1 = prev.pose.pose.position, curr.pose.pose.position
        t0 = prev.header.stamp.sec + prev.header.stamp.nanosec * 1e-9
        t1 = curr.header.stamp.sec + curr.header.stamp.nanosec * 1e-9
        dt = t1 - t0
        if dt <= 0.0:
            raise ValueError(f"non-positive dt: {dt}")
        return [(p1.x - p0.x)/dt, (p1.y - p0.y)/dt, (p1.z - p0.z)/dt], t1

    def imu_callback(self, msg: Imu):
        # if not self.initialized:
        #     return
        
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.imu_sec = msg.header.stamp.sec
        self.imu_nanosec = msg.header.stamp.nanosec

        imu_in = ImuData(
            wx=msg.angular_velocity.x, wy=msg.angular_velocity.y, wz=msg.angular_velocity.z,
            ax=msg.linear_acceleration.x, ay=msg.linear_acceleration.y, az=msg.linear_acceleration.z,
            qx=msg.orientation.x, qy=msg.orientation.y, qz=msg.orientation.z, qw=msg.orientation.w,
            stamp=stamp
        )

        self.corrector.add_sample(imu_in)
        self.airio.add_sample(imu_in)

        # ====== seqlen 샘플마다 처리 트리거 ======
        if self.samples_since_proc >= max(1, self.seqlen):
            # 재진입 방지: 바쁘면 스킵하고 다음 기회에 처리
            if self.proc_lock.acquire(False):
                try:
                    self._process_once()
                finally:
                    self.samples_since_proc = 0
                    self.proc_lock.release()
    
    def _process_once(self):
        # if not self.initialized:
        #     return
        
        self.proc_count += 1  # 디커플링 카운터
        t0 = time.time()

        # 최신 보정값
        imu_out = self.corrector.correct_latest()
        if imu_out is None:
            return
        airimu_step_t = time.time() - t0

        # === EKF PROPAGATION (보정 IMU 사용) ===
        try:
            self.ekf.add_imu(ImuSample(
                wx=float(imu_out.wx), wy=float(imu_out.wy), wz=float(imu_out.wz),
                ax=float(imu_out.ax), ay=float(imu_out.ay), az=float(imu_out.az),
                stamp=float(imu_out.stamp)
                ),
                imu_out.gyro_var,
                imu_out.acc_var
            )
        except Exception as e:
            self.get_logger().warn(f"EKF propagate failed: {e}")

        # === AirIO 속도 예측 ===
        t2 = time.time()
        run_airio = (self.proc_count % self.airio_every == 0)
        if run_airio:
            # EKF 추정 자세 사용
            cur_rot_ekf = np.asarray(self.ekf.get_state()["rot"], dtype=float)
            airio_out = self.airio.predict_velocity(cur_rot_ekf)

            # vel + eta_v 파싱 (dict/tuple 호환)
            if isinstance(airio_out, dict):
                net_vel = np.asarray(airio_out.get("vel", (0.0, 0.0, 0.0)), dtype=float)
                eta_v   = np.asarray(airio_out.get("eta_v", (0.05, 0.05, 0.05)), dtype=float)
            else:
                net_vel = np.asarray(airio_out, dtype=float)
                eta_v   = np.asarray((0.05, 0.05, 0.05), dtype=float)

            self.last_net_vel = net_vel
            self.last_eta_v   = eta_v
        else:
            net_vel = self.last_net_vel
        airio_network_step_t = time.time() - t2

        # 데드밴드
        if np.linalg.norm(net_vel) * 1000.0 < self.deadband_ms:
            net_vel = np.zeros(3, dtype=float)
            self.last_net_vel = net_vel

        # === EKF UPDATE (바디 프레임 속도 + R_meas=diag(eta_v^2)) — 정지 조건 제거 ===
        # if run_airio and not stationary:
        if run_airio:
            try:
                eta_v = getattr(self, "last_eta_v", np.array([0.05, 0.05, 0.05], dtype=float))
                R_meas = np.diag((eta_v ** 2.0).tolist())
                self.ekf.update_velocity_body(tuple(net_vel.tolist()), R_meas)
            except Exception as e:
                self.get_logger().warn(f"EKF update (velocity) failed: {e}")

        total_t = time.time() - t0

        if self.TL_mode and total_t <= 0.1:
            self.airimu_step_t_deque.append(airimu_step_t)
            self.airio_network_step_t_deque.append(airio_network_step_t)
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
        # 공분산 관련 update
        gyro_var = imu_out.gyro_var
        acc_var = imu_out.acc_var
        imu_msg.linear_acceleration_covariance = [
            float(acc_var[0]), 0.0, 0.0,
            0.0, float(acc_var[1]), 0.0,
            0.0, 0.0, float(acc_var[2])
        ]
        imu_msg.angular_velocity_covariance = [
        float(gyro_var[0]), 0.0, 0.0,
        0.0, float(gyro_var[1]), 0.0,
        0.0, 0.0, float(gyro_var[2])
        ]
        self.filtered_pub.publish(imu_msg)
        
        # publish odom (frame=map) — EKF 상태 직결
        self._publish_ekf_state_if_due()

    def _now_ros_time_sec(self) -> float:
        # ROS 시뮬레이션 시간(/clock) 사용 시에도 안전하게 초 단위 반환
        nsec = self.get_clock().now().nanoseconds
        return float(nsec) * 1e-9
    
    def _publish_ekf_state_if_due(self):
        now = self._now_ros_time_sec()
        if (now - self._last_pub_ts) < self._pub_period:
            return
        self._last_pub_ts = now
        self._publish_ekf_state()

    def _publish_ekf_state(self):
        s = self.ekf.get_state()
        pos = np.asarray(s["pos"], dtype=float)
        rot = np.asarray(s["rot"], dtype=float)
        vel = np.asarray(s["vel"], dtype=float)
    
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = "map"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = float(pos[0])
        odom.pose.pose.position.y = float(pos[1])
        odom.pose.pose.position.z = float(pos[2])
        odom.pose.pose.orientation.x = float(rot[0])
        odom.pose.pose.orientation.y = float(rot[1])
        odom.pose.pose.orientation.z = float(rot[2])
        odom.pose.pose.orientation.w = float(rot[3])
        odom.twist.twist.linear.x = float(vel[0])
        odom.twist.twist.linear.y = float(vel[1])
        odom.twist.twist.linear.z = float(vel[2])
        self.odom_pub.publish(odom)
        
    # (옵션) 타이밍 저장
    def save_timings(self):
        import matplotlib.pyplot as plt  # 필요 시에만 임포트(지연 로딩)
        timings = {
            "AIR-IMU": list(self.airimu_step_t_deque),
            "AIR-IO Network": list(self.airio_network_step_t_deque),
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