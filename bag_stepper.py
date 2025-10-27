#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BagStepper: 스페이스바를 누를 때마다 /scan 한 프레임과
그에 가장 가까운 /odom 프레임을 publish한다.

사용 예:
  python3 bag_stepper.py /path/to/rosbag2_dir \
    --scan /scan --odom /odom \
    --pub-scan /step_scan --pub-odom /step_odom \
    --scan-reliable --odom-reliable

- /scan: sensor_msgs/LaserScan
- /odom: nav_msgs/Odometry
- QoS는 --scan-reliable / --odom-reliable 플래그로 RELIABLE/BEST_EFFORT 선택 가능
"""

import os
import sys
import tty
import termios
import select
import argparse
import bisect
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from builtin_interfaces.msg import Time as TimeMsg
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

import rosbag2_py


# ========= 유틸 =========
def time_to_ns(t: TimeMsg) -> int:
    return int(t.sec) * 1_000_000_000 + int(t.nanosec)


def nearest_index(sorted_ns_list: List[int], target_ns: int) -> int:
    """정렬된 타임스탬프 배열에서 target과 가장 가까운 인덱스를 찾는다."""
    pos = bisect.bisect_left(sorted_ns_list, target_ns)
    if pos == 0:
        return 0
    if pos == len(sorted_ns_list):
        return len(sorted_ns_list) - 1
    before = sorted_ns_list[pos - 1]
    after = sorted_ns_list[pos]
    return pos - 1 if (target_ns - before) <= (after - target_ns) else pos


class KeyPoll:
    """터미널에서 비차단 방식으로 키 입력 감지."""
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def poll(self) -> str:
        """입력이 있으면 한 글자 반환, 없으면 ''."""
        dr, _, _ = select.select([sys.stdin], [], [], 0.0)
        if dr:
            ch = sys.stdin.read(1)
            return ch
        return ''


def make_qos(depth: int, reliable: bool) -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth
    )


def load_bag(bag_path: str,
             scan_topic: str,
             odom_topic: str) -> Tuple[List[LaserScan], List[int], List[Odometry], List[int]]:
    """
    rosbag2 디렉터리에서 /scan(LaserScan), /odom(Odometry)을 모두 읽어
    시간 정렬된 리스트로 반환한다.
    """
    if not os.path.isdir(bag_path):
        raise FileNotFoundError(f"존재하지 않는 디렉터리: {bag_path}")

    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    # 토픽 타입 맵 구성
    topics_and_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topics_and_types}

    scan_type = type_map.get(scan_topic, None)
    odom_type = type_map.get(odom_topic, None)

    if scan_type != 'sensor_msgs/msg/LaserScan':
        raise RuntimeError(f"{scan_topic} 타입이 LaserScan이 아님: {scan_type}")
    if odom_type != 'nav_msgs/msg/Odometry':
        raise RuntimeError(f"{odom_topic} 타입이 Odometry가 아님: {odom_type}")

    scan_msgs: List[LaserScan] = []
    scan_times_ns: List[int] = []
    odom_msgs: List[Odometry] = []
    odom_times_ns: List[int] = []

    # 메시지 순회
    while reader.has_next():
        (topic, data, _) = reader.read_next()
        if topic == scan_topic:
            msg = deserialize_message(data, LaserScan)
            if hasattr(msg, 'header'):
                tns = time_to_ns(msg.header.stamp)
                scan_msgs.append(msg)
                scan_times_ns.append(tns)
        elif topic == odom_topic:
            msg = deserialize_message(data, Odometry)
            if hasattr(msg, 'header'):
                tns = time_to_ns(msg.header.stamp)
                odom_msgs.append(msg)
                odom_times_ns.append(tns)

    # 시간 기준 정렬(안전)
    scan_sorted = sorted(zip(scan_times_ns, scan_msgs), key=lambda x: x[0])
    odom_sorted = sorted(zip(odom_times_ns, odom_msgs), key=lambda x: x[0])

    scan_times_ns = [t for t, _ in scan_sorted]
    scan_msgs = [m for _, m in scan_sorted]
    odom_times_ns = [t for t, _ in odom_sorted]
    odom_msgs = [m for _, m in odom_sorted]

    if not scan_msgs:
        raise RuntimeError(f"bag에서 {scan_topic} 메시지를 찾지 못함")
    if not odom_msgs:
        raise RuntimeError(f"bag에서 {odom_topic} 메시지를 찾지 못함")

    return scan_msgs, scan_times_ns, odom_msgs, odom_times_ns


# ========= 노드 =========
class BagStepper(Node):
    def __init__(self,
                 bag_path: str,
                 scan_topic: str = '/scan',
                 odom_topic: str = '/odom',
                 pub_scan_topic: str = '/scan',
                 pub_odom_topic: str = '/odom',
                 scan_reliable: bool = False,
                 odom_reliable: bool = False,
                 scan_depth: int = 10,
                 odom_depth: int = 10):
        super().__init__('bag_stepper')

        self.get_logger().info(f"bag 로드 시작: {bag_path}")
        (self.scan_msgs,
         self.scan_times_ns,
         self.odom_msgs,
         self.odom_times_ns) = load_bag(bag_path, scan_topic, odom_topic)

        self.get_logger().info(f"로드 완료: scan {len(self.scan_msgs)}개, odom {len(self.odom_msgs)}개")

        # 퍼블리셔 QoS 구성
        scan_qos = make_qos(depth=scan_depth, reliable=scan_reliable)
        odom_qos = make_qos(depth=odom_depth, reliable=odom_reliable)

        # 퍼블리셔
        self.scan_pub = self.create_publisher(LaserScan, pub_scan_topic, scan_qos)
        self.odom_pub = self.create_publisher(Odometry, pub_odom_topic, odom_qos)

        # 진행 인덱스
        self.idx = 0
        self.total = len(self.scan_msgs)

        # 키 폴링 타이머
        self.timer = self.create_timer(0.02, self.on_timer)  # 50Hz 폴링
        self.keypoll = KeyPoll().__enter__()
        self._closed = False

        self.get_logger().info("스페이스바: 다음 프레임 publish /  q: 종료")

    def on_timer(self):
        if self._closed:
            return
        ch = self.keypoll.poll()
        if ch == ' ':
            self.publish_next()
        elif ch in ('q', 'Q', '\x03', '\x04'):  # Ctrl-C/D
            self.shutdown()

    def publish_next(self):
        if self.idx >= self.total:
            self.get_logger().info("마지막 프레임까지 모두 publish함.")
            return

        scan = self.scan_msgs[self.idx]
        tns = self.scan_times_ns[self.idx]

        # 가장 가까운 odom 찾기
        j = nearest_index(self.odom_times_ns, tns)
        odom = self.odom_msgs[j]
        diff_ns = abs(self.odom_times_ns[j] - tns)

        # publish
        self.scan_pub.publish(scan)
        self.odom_pub.publish(odom)

        # 로그
        s = scan.header.stamp
        o = odom.header.stamp
        self.get_logger().info(
            f"[{self.idx+1}/{self.total}] publish "
            f"scan@{s.sec}.{s.nanosec:09d}  "
            f"odom@{o.sec}.{o.nanosec:09d}  "
            f"(Δ={diff_ns/1e6:.3f} ms)"
        )

        self.idx += 1

    def shutdown(self):
        if not self._closed:
            self._closed = True
            try:
                self.keypoll.__exit__(None, None, None)
            except Exception:
                pass
            self.get_logger().info("종료합니다.")
            rclpy.shutdown()


# ========= 엔트리포인트 =========
def main():
    parser = argparse.ArgumentParser(description="스페이스바로 bag 재생(step) 퍼블리셔")
    parser.add_argument('bag', help='rosbag2 디렉터리 경로')
    parser.add_argument('--scan', default='/scan', help='bag의 scan 토픽명 (기본: /scan)')
    parser.add_argument('--odom', default='/odom', help='bag의 odom 토픽명 (기본: /odom)')
    parser.add_argument('--pub-scan', default='/scan', help='publish할 scan 토픽명 (기본: /scan)')
    parser.add_argument('--pub-odom', default='/odom', help='publish할 odom 토픽명 (기본: /odom)')

    # QoS 옵션
    parser.add_argument('--scan-reliable', action='store_true',
                        help='scan 퍼블리셔를 RELIABLE로 설정 (기본: BEST_EFFORT)')
    parser.add_argument('--odom-reliable', action='store_true',
                        help='odom 퍼블리셔를 RELIABLE로 설정 (기본: BEST_EFFORT)')
    parser.add_argument('--scan-depth', type=int, default=10, help='scan QoS depth (기본: 10)')
    parser.add_argument('--odom-depth', type=int, default=10, help='odom QoS depth (기본: 10)')

    args = parser.parse_args()

    rclpy.init(args=None)
    node = BagStepper(
        bag_path=args.bag,
        scan_topic=args.scan,
        odom_topic=args.odom,
        pub_scan_topic=args.pub_scan,
        pub_odom_topic=args.pub_odom,
        scan_reliable=args.scan_reliable,
        odom_reliable=args.odom_reliable,
        scan_depth=args.scan_depth,
        odom_depth=args.odom_depth,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.shutdown()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
