#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import threading
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

# === 글로벌 변수들 ===
last_received_data = None
ppcmd_active = False
ppcmd_timer = 0.0
target_pos = None
target_ori = None
ros_node = None

# === ppcmd 콜백 함수 ===


def ppcmd_callback(msg):
    global last_received_data, ppcmd_active, ppcmd_timer
    data = msg.data.strip()
    try:
        # 메시지 내용은 14개의 float 값이어야 함
        floats = list(map(float, data.split()))
        if len(floats) != 14:
            print("Received ppcmd data does not have 14 floats.")
            return
        last_received_data = floats
        ppcmd_active = True
        ppcmd_timer = time.time()
        print("Pose command received. Activating ppcmd for 1 second.")
    except Exception as e:
        print(f"Error parsing ppcmd message: {e}")


# === follow_target 콜백 함수 ===


def follow_target_callback(msg):
    global target_pos, target_ori
    data = msg.data.strip()
    try:
        # 메시지 내용은 6개의 float (x,y,z,roll,pitch,yaw)여야 함
        floats = list(map(float, data.split()))
        if len(floats) != 6:
            print("follow_target data does not have 6 floats.")
            return
        x, y, z, roll, pitch, yaw = floats
        # roll += 180.0  # roll을 180도 보정
        # Euler (degree) → Quaternion (xyzw 순서)
        r = R.from_euler("xyz", [roll, pitch, yaw], degrees=True)
        quat = r.as_quat()  # [x, y, z, w]
        target_ori = np.array(quat)
        target_pos = np.array([x, y, z])
        print("Updated target pos and orientation (xyzw, degrees input).")
    except Exception as e:
        print(f"Error parsing follow_target message: {e}")


# === ROS Node 정의 ===


class CommandSubscriber(Node):
    def __init__(self):
        super().__init__("command_listener")
        # "ppcmd" 토픽 구독
        self.create_subscription(String, "ppcmd", ppcmd_callback, 10)
        # "follow_target" 토픽 구독
        self.create_subscription(String, "follow_target", follow_target_callback, 10)
        self.get_logger().info(
            "Command subscriber initialized (ppcmd & follow_target)."
        )


# === OmniGraph ScriptNode 필수 함수 ===


def setup(db):
    global ros_node
    rclpy.init()
    ros_node = CommandSubscriber()
    threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True).start()
    print("[ROS2] Command subscriber (ppcmd & follow_target) initialized.")


def compute(db):
    global last_received_data, ppcmd_active, ppcmd_timer
    global target_pos, target_ori

    if last_received_data:
        # 예시: grasp와 place 명령을 각각 7개의 float로 나눠서 출력
        db.outputs.grasp_point_ori = np.array(last_received_data[0:4])
        db.outputs.grasp_point_pos = np.array(last_received_data[4:7])
        db.outputs.place_point_ori = np.array(last_received_data[7:11])
        db.outputs.place_point_pos = np.array(last_received_data[11:14])

    if ppcmd_active and (time.time() - ppcmd_timer <= 1.0):
        db.outputs.ppcmd = True
    else:
        db.outputs.ppcmd = False
        ppcmd_active = False

    if target_pos is not None and target_ori is not None:
        db.outputs.target_pos = target_pos
        db.outputs.target_ori = target_ori

    return True


def cleanup(db):
    global ros_node
    if ros_node is not None:
        ros_node.destroy_node()
        rclpy.shutdown()
    print("[ROS2] Command subscriber cleanup done.")
