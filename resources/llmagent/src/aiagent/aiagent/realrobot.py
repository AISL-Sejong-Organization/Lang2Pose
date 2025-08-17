#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import re
import speech_recognition as sr
import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from speech_recognition import WaitTimeoutError

# 환경변수 로드 및 API KEY 체크
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set!")

# Prompt 템플릿들
agent_prompt = PromptTemplate(
    input_variables=["user_command", "current_pose"],
    template="""
다음은 로봇의 현재 End-Effector Pose와 사용자의 명령이다.
Pose는 x, y, z, r(roll), p(pitch), y(yaw) 순서로 주어진다.

현재 Pose: {current_pose}
사용자 명령: "{user_command}"

사용자의 의도에 따라 '새로운' pose (x, y, z, r, p, y)를 결정하거나,
'증분 이동'을 적용해야 할 수 있다.
아래 형식으로만 결과를 1줄에 작성하라.
예시: "x=0.3, y=-0.1, z=0.4, r=1.57, p=0.0, y=1.57"

주의사항:
- "준비 자세" 라고 하면 x=0.2, y=0.0, z=0.6, r=2.4, p=0.0, y=1.57 로 맞춘다.
- "왼쪽으로 조금만 더" 는 y를 -0.1m 만큼 이동한다(현재 위치 기준).
- "오른쪽으로 0.2m 정도" 라면 y를 +0.2m 이동한다.
- "앞으로 0.2m" 는 x를 +0.2m 이동.
- "뒤로 0.3m" 는 x를 -0.3m 이동.
- "위로 0.1m" 는 z를 +0.1m 이동.
- "아래로 0.05m" 는 z를 -0.05m 이동.
- "롤값 2.4로 해" 라고 하면 r=2.4 로 변경한다(절대값).
- 정면 주시는 롤 값이 1.57이며 하방 주시는 롤 값이 3.14이며 천장 주시는 롤 값이 0임
- "피치값 1.0으로" 하면 p=1.0, "피치 0.5 더" 면 p += 0.5
- "고개를 숙여"라고 하면 r을 0.3 증가시킨다(최대 3.14)
- "야우값은 항상 1.57" 이 디폴트라면, 변경되지 않는 한 1.57로 고정.

조건에 맞춰 최종 pose를 "x=..., y=..., z=..., r=..., p=..., y=..." 형태로 한 줄에 출력하라.
    """,
)

pick_place_prompt = PromptTemplate(
    input_variables=["objects_list", "user_command"],
    template="""
사용자가 잡고 싶어하는 오브젝트 이름을 찾으려고 한다.
아래는 현재 인식된 오브젝트 이름 목록이다:
{objects_list}

사용자의 명령: "{user_command}"

위 오브젝트 목록 중 어떤 것이 사용자 명령과 가장 관련이 있는지 이름을 정확히 골라내라.
만약 전혀 없으면 "None" 을 출력한다.

오직 이름만 짧게 출력하라(예: "ycb_cracker_box" 또는 "None").
    """,
)


class AIAgentNode(Node):
    def __init__(self):
        node_name = "ai_agent_node"
        super().__init__(node_name)

        self.current_pose = [0.4, 0.0, 0.4, 1.57, 0.0, 1.57]
        self.current_mode = "armcontroller"
        self.objects = {}
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone(device_index=12)

        self.object_marker_sub = self.create_subscription(
            MarkerArray, "/object_marker_array", self.object_marker_callback, 10
        )
        self.command_sub = self.create_subscription(
            String, "/user_command", self.user_command_callback, 10
        )
        self.command_pub = self.create_publisher(String, "/command", 10)

        self.llm = ChatOpenAI(
            model="gpt-4o-mini", temperature=0.2, openai_api_key=OPENAI_API_KEY
        )

        self.get_logger().info(f"{node_name} started.")
        self.input_mode = self.select_input_mode()

        # 음성 입력 모드인 경우 음성 인식 스레드를 시작
        if self.input_mode == "voice":
            self.voice_thread = threading.Thread(target=self.record_voice_loop)
            self.voice_thread.daemon = True
            self.voice_thread.start()
        # 텍스트 입력 모드인 경우 키보드 입력 스레드를 시작
        elif self.input_mode == "text":
            self.text_thread = threading.Thread(target=self.text_input_loop)
            self.text_thread.daemon = True
            self.text_thread.start()

    def select_input_mode(self):
        while True:
            mode = input("입력 모드를 선택하세요 (1: 음성, 2: 텍스트): ").strip()
            if mode == "1":
                return "voice"
            elif mode == "2":
                return "text"
            else:
                print("잘못된 입력입니다. 1 또는 2를 입력하세요.")

    def text_input_loop(self):
        while self.input_mode == "text":
            try:
                command = input("명령어를 입력하세요: ").strip()
                if command:
                    msg = String()
                    msg.data = command
                    self.user_command_callback(msg)
            except KeyboardInterrupt:
                break

    def record_voice_loop(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            while self.input_mode == "voice":
                try:
                    self.get_logger().info("[INFO] 음성을 듣는 중...")
                    audio = self.recognizer.listen(source, timeout=5)
                    text = self.recognizer.recognize_google(audio, language="ko-KR")
                    self.get_logger().info(f"[INFO] 음성 인식 결과: {text}")
                    msg = String()
                    msg.data = text
                    self.user_command_callback(msg)
                except WaitTimeoutError:
                    self.get_logger().warn("[WARN] 음성 시작을 기다리다 타임아웃 발생")
                except sr.UnknownValueError:
                    self.get_logger().warn("[WARN] 음성을 인식하지 못했습니다.")
                except sr.RequestError as e:
                    self.get_logger().error(f"[ERROR] STT 서비스 오류: {e}")

    def object_marker_callback(self, msg: MarkerArray):
        new_objects = {}
        for marker in msg.markers:
            name = marker.text.strip()
            if not name:
                continue
            new_objects[name] = {
                "frame_id": marker.header.frame_id,
                "width": marker.scale.x,
                "depth": marker.scale.y,
                "height": marker.scale.z,
                "x": marker.pose.position.x,
                "y": marker.pose.position.y,
                "z": marker.pose.position.z,
                "qx": marker.pose.orientation.x,
                "qy": marker.pose.orientation.y,
                "qz": marker.pose.orientation.z,
                "qw": marker.pose.orientation.w,
            }
        self.objects = new_objects

    def execute_command(self, command: str):
        try:
            subprocess.run(command, shell=True, check=True)
            self.get_logger().info(f"[INFO] Executed command: {command}")
        except subprocess.CalledProcessError:
            self.get_logger().error(f"[ERROR] Failed to execute command: {command}")

    def user_command_callback(self, msg: String):
        user_cmd = msg.data.strip()
        self.get_logger().info(f"Received command: {user_cmd}")

        if "암컨트롤" in user_cmd:
            self.current_mode = "armcontroller"
            self.execute_command(
                "ros2 param set /kirom_interface control_mode armcontroller"
            )
        elif "플레이스" in user_cmd:
            self.current_mode = "pick_and_place"
            self.execute_command(
                "ros2 param set /kirom_interface control_mode pick_and_place"
            )
        elif self.current_mode == "armcontroller":
            self.handle_armcontroller_command(user_cmd)
        elif self.current_mode == "pick_and_place":
            self.handle_pick_and_place_command(user_cmd)

    def handle_armcontroller_command(self, user_cmd: str):
        prompt_text = agent_prompt.format(
            user_command=user_cmd,
            current_pose=", ".join(
                [
                    f"{n}={v}"
                    for n, v in zip(["x", "y", "z", "r", "p", "y"], self.current_pose)
                ]
            ),
        )
        response = self.llm.invoke(prompt_text).content.strip().replace('"', "")
        self.get_logger().info(f"Generated new pose: {response}")
        try:
            pose_pairs = [pair.split("=") for pair in response.split(", ")]
            new_pose = [float(value.strip()) for _, value in pose_pairs]
            pose_msg = String()
            pose_msg.data = " ".join([f"{v:.2f}" for v in new_pose])
            self.command_pub.publish(pose_msg)
            self.current_pose = new_pose
        except ValueError as e:
            self.get_logger().error(
                f"[ERROR] Failed to parse pose response: {response}. Error: {e}"
            )

    def handle_pick_and_place_command(self, user_cmd: str):
        if not self.objects:
            self.get_logger().error("[ERROR] No objects available.")
            return

        objects_list_str = ", ".join(list(self.objects.keys()))
        prompt_text = pick_place_prompt.format(
            objects_list=objects_list_str, user_command=user_cmd
        )
        identified_obj = self.llm.invoke(prompt_text).content.strip()

        if identified_obj == "None" or identified_obj not in self.objects:
            self.get_logger().info(
                f"[INFO] Cannot find matching object for: {identified_obj}"
            )
            return

        obj = self.objects[identified_obj]
        frame_id, w, d, h = obj["frame_id"], obj["width"], obj["depth"], obj["height"]
        x, y, z = obj["x"], obj["y"], obj["z"]
        qx, qy, qz, qw = obj["qx"], obj["qy"], obj["qz"], obj["qw"]

        tx, ty, tz = x, y, z
        tqx, tqy, tqz, tqw = qx, qy, qz, qw

        for val, attr in zip(
            ["x", "y", "z", "qx", "qy", "qz", "qw"], [tx, ty, tz, tqx, tqy, tqz, tqw]
        ):
            match = re.search(rf"place_{val}\s*=\s*([-\d\.]+)", user_cmd)
            if match:
                locals()[f"t{val}"] = float(match.group(1))

        cmd = f"{frame_id} {w} {d} {h} {x} {y} {z} {qx} {qy} {qz} {qw} {tx} {ty} {tz} {tqx} {tqy} {tqz} {tqw}"
        msg = String()
        msg.data = cmd
        self.command_pub.publish(msg)
        self.get_logger().info(f"[INFO] Published PICK&PLACE command: {cmd}")


def main(args=None):
    rclpy.init(args=args)
    node = AIAgentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
