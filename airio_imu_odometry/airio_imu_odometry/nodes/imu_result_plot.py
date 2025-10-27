#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Imu

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
import os
import argparse
from threading import Lock

class ImuPlotter(Node):
    def __init__(self,
                 imu_topic:str="/imu/data_raw",
                 max_len:int=1500,
                 gravity:float=9.81007,
                 save_dir:str=".",
                 save_prefix:str="imu",
                 autosave_sec:float=0.0):
        super().__init__('imu_plotter')

        self.gravity = gravity
        self.save_dir = save_dir
        self.save_prefix = save_prefix
        self.autosave_sec = float(autosave_sec)
        os.makedirs(self.save_dir, exist_ok=True)

        # --- QoS (센서 데이터용) ---
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=100
        )

        # --- Subscription ---
        self.create_subscription(Imu, imu_topic, self.imu_callback, qos_sensor)

        # --- Buffers ---
        self.max_len = max_len
        self.lock = Lock()
        self.t0_imu  = None
        self.t_imu   = deque(maxlen=self.max_len)
        self.gyro_n  = deque(maxlen=self.max_len)
        self.acc_mg  = deque(maxlen=self.max_len)

        # --- Matplotlib ---
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(11, 6))
        (self.line_gyro,) = self.ax.plot([], [], label="IMU Gyro Norm [rad/s]", linewidth=1.2)
        (self.line_acc,)  = self.ax.plot([], [], label="IMU (|acc|-g) [m/s²]", linewidth=1.2)

        self.ax.set_title("IMU Stationary Check")
        self.ax.set_xlabel("Time [s]")
        self.ax.set_ylabel("Value")
        self.ax.grid(True)
        self.ax.legend(loc="upper right")

        self.last_draw = time.time()
        self.last_save = time.time()
        self.timer = self.create_timer(0.05, self.update_plot)

        self.get_logger().info(f"Subscribed: imu={imu_topic}")
        self.get_logger().info(f"Autosave every {self.autosave_sec:.1f}s (0=off). Save dir: {self.save_dir}")

    # ---------- Callbacks ----------
    def imu_callback(self, msg: Imu):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.t0_imu is None:
            self.t0_imu = stamp
        t = stamp - self.t0_imu

        gx, gy, gz = msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z
        ax, ay, az = msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z

        gyro_norm = float(np.linalg.norm([gx, gy, gz]))
        acc_norm_minus_g = float(np.linalg.norm([ax, ay, az]) - self.gravity)
        acc_norm_minus_g = float(abs(acc_norm_minus_g))

        with self.lock:
            self.t_imu.append(t)
            self.gyro_n.append(gyro_norm)
            self.acc_mg.append(acc_norm_minus_g)

    # ---------- Plot updater ----------
    def update_plot(self):
        if time.time() - self.last_draw < 0.03:
            return
        with self.lock:
            self.line_gyro.set_data(self.t_imu, self.gyro_n)
            self.line_acc.set_data(self.t_imu, self.acc_mg)
            self.ax.relim()
            self.ax.autoscale_view()

        plt.pause(0.001)
        self.last_draw = time.time()

        if self.autosave_sec > 0.0 and (time.time() - self.last_save) >= self.autosave_sec:
            self.savefig(tag="latest")  
            self.last_save = time.time()

    # ---------- Save helpers ----------
    def _outfile(self, tag:str):
        ts = time.strftime("%Y%m%d_%H%M%S")
        base = f"{self.save_prefix}_{tag}_{ts}.png" if tag else f"{self.save_prefix}_{ts}.png"
        return os.path.join(self.save_dir, base)

    def savefig(self, tag:str=""):
        path = self._outfile(tag)
        self.fig.savefig(path, dpi=160, bbox_inches="tight")
        self.get_logger().info(f"Saved figure: {path}")

def main():
    parser = argparse.ArgumentParser(description="Plot only /imu/data_raw (Gyro norm & |Acc|-g). Save PNG on exit.")
    parser.add_argument("--imu_topic", default="/imu/data_raw")
    parser.add_argument("--max_len", type=int, default=1500)
    parser.add_argument("--gravity", type=float, default=9.81007)
    parser.add_argument("--save_dir", default=".")
    parser.add_argument("--save_prefix", default="imu")
    parser.add_argument("--autosave_sec", type=float, default=0.0,
                        help="Autosave interval in seconds (0=disable)")
    args = parser.parse_args()

    rclpy.init()
    node = ImuPlotter(
        imu_topic=args.imu_topic,
        max_len=args.max_len,
        gravity=args.gravity,
        save_dir=args.save_dir,
        save_prefix=args.save_prefix,
        autosave_sec=args.autosave_sec
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt: saving final figure...")
        node.savefig(tag="final")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
