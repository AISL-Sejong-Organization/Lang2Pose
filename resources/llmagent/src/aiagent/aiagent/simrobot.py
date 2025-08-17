#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import re
import math
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray, Marker
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from speech_recognition import WaitTimeoutError

# tf2 관련 임포트
import tf2_ros
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R

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
예시: "x=0.3, y=-0.1, z=0.4, r=1.57, p=0.0, y=0"

주의사항:
- "준비 자세" 라고 하면 x=0.4, y=0.0, z=0.5, r=0.0, p=3.14, y=0.0 로 맞춘다.
- "왼쪽은 y축의 양수 방향이며 오른쪽은 y축의 음수 방향이다."
- "왼쪽으로 조금만 더" 는 y를 +0.1m 만큼 이동한다(현재 위치 기준).
- "오른쪽으로 0.2m 정도" 라면 y를 -0.2m 이동한다.
- "앞으로 0.2m" 는 x를 +0.2m 이동.
- "뒤로 0.3m" 는 x를 -0.3m 이동.
- "위로 0.1m" 는 z를 +0.1m 이동.
- "아래로 0.05m" 는 z를 -0.05m 이동.
- "롤값 2.4로 해" 라고 하면 r=2.4 로 변경한다(절대값).
- 정면 주시는 p 값이 -90(degree)이며 하방 주시는 p 값이 0이며 천장 주시는 p 값이 90임
- "피치값 1.0으로" 하면 p=1.0, "피치 0.5 더" 면 p += 0.5
- "고개를 숙여"라고 하면 r을 0.3 증가시킨다(최대 3.14)
- "야우값은 항상 0" 이 디폴트라면, 변경되지 않는 한 0로 고정.

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
def my_do_transform_pose(pose_stamped, transform):
    """
    pose_stamped: geometry_msgs.msg.PoseStamped (위치는 pose_stamped.pose.position에 있음)
    transform: tf2_ros.TransformStamped (transform.transform.rotation, transform.transform.translation)
    """
    # 1. 변환 행렬 T (4x4 homogeneous matrix) 구성
    t = transform.transform.translation
    q = transform.transform.rotation
    # scipy를 사용해 변환 회전 행렬을 구함 (형태: [x, y, z, w])
    R_transform = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T = np.eye(4)
    T[0:3, 0:3] = R_transform
    T[0:3, 3] = [t.x, t.y, t.z]

    # 2. pose의 위치를 동차 좌표(homogeneous coordinate)로 변환
    p = pose_stamped.pose.position
    point = np.array([p.x, p.y, p.z, 1.0])
    new_point = T @ point

    # 3. Orientation 변환:
    #    원래 pose의 orientation과 변환의 회전을 결합합니다.
    q_pose = pose_stamped.pose.orientation
    R_pose = R.from_quat([q_pose.x, q_pose.y, q_pose.z, q_pose.w])
    # 변환의 회전(R_transform)을 곱해 새 회전 행렬을 구함
    R_new = R.from_matrix(T[0:3, 0:3]) * R_pose
    new_quat = R_new.as_quat()  # 결과: [x, y, z, w]

    # 4. 새 PoseStamped 생성
    from geometry_msgs.msg import PoseStamped  # 이미 import 되어 있다면 생략 가능

    new_pose = PoseStamped()
    new_pose.header.stamp = transform.header.stamp
    new_pose.header.frame_id = transform.child_frame_id
    new_pose.pose.position.x = new_point[0]
    new_pose.pose.position.y = new_point[1]
    new_pose.pose.position.z = new_point[2]
    new_pose.pose.orientation.x = new_quat[0]
    new_pose.pose.orientation.y = new_quat[1]
    new_pose.pose.orientation.z = new_quat[2]
    new_pose.pose.orientation.w = new_quat[3]
    return new_pose


class AIAgentNode(Node):
    def __init__(self):
        node_name = "ai_agent_node"
        super().__init__(node_name)

        self.current_pose = [0.3, 0.0, 0.6, 0.0, -0.2, 0.0]
        self.current_mode = "armcontroller"
        self.objects = {}
        try:
            import speech_recognition as sr

            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone(device_index=12)
        except Exception:
            self.get_logger().warn("음성 인식 라이브러리 초기화 실패")
            self.recognizer = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.object_marker_sub = self.create_subscription(
            MarkerArray, "/object_marker_array", self.object_marker_callback, 10
        )
        self.command_sub = self.create_subscription(
            String, "/user_command", self.user_command_callback, 10
        )
        self.follow_pub = self.create_publisher(String, "follow_target", 10)
        self.ppcmd_pub = self.create_publisher(String, "ppcmd", 10)

        # 새로운 퍼블리셔: base_link_object_array
        self.base_link_object_array_pub = self.create_publisher(
            MarkerArray, "/base_link_object_array", 10
        )

        self.llm = ChatOpenAI(
            model="gpt-4o-mini", temperature=0.2, openai_api_key=OPENAI_API_KEY
        )

        self.get_logger().info(f"{node_name} started.")
        self.input_mode = self.select_input_mode()

        if self.input_mode == "voice":
            self.voice_thread = threading.Thread(target=self.record_voice_loop)
            self.voice_thread.daemon = True
            self.voice_thread.start()
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
        if self.recognizer is None:
            return
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
                    self.get_logger().warn("[WARN] 음성 시작 타임아웃")
                except Exception as e:
                    self.get_logger().warn(f"[WARN] 음성 인식 오류: {e}")

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

        # base_link_object_array 토픽에 발행 (필요 시 transform_marker_to_base_link를 호출)
        transformed_markers = []
        for marker in msg.markers:
            new_marker = self.transform_marker_to_base_link(marker)
            if new_marker is not None:
                transformed_markers.append(new_marker)
        if transformed_markers:
            new_msg = MarkerArray()
            new_msg.markers = transformed_markers
            self.base_link_object_array_pub.publish(new_msg)
            self.get_logger().info(
                f"[DEBUG] Published {len(transformed_markers)} markers in base_link frame."
            )

    def transform_marker_to_base_link(self, marker: Marker) -> Marker:
        # 새 PoseStamped 객체 생성
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.header.frame_id = marker.header.frame_id  # 예: "Camera"
        pose_stamped.pose.position.x = marker.pose.position.x
        pose_stamped.pose.position.y = marker.pose.position.y
        pose_stamped.pose.position.z = marker.pose.position.z
        pose_stamped.pose.orientation.x = marker.pose.orientation.x
        pose_stamped.pose.orientation.y = marker.pose.orientation.y
        pose_stamped.pose.orientation.z = marker.pose.orientation.z
        pose_stamped.pose.orientation.w = marker.pose.orientation.w

        try:
            transform = self.tf_buffer.lookup_transform(
                "base_link",
                pose_stamped.header.frame_id,
                self.get_clock().now().to_msg(),
                rclpy.duration.Duration(seconds=1.0),
            )
            # 직접 계산: 변환 행렬 생성
            t = np.array(
                [
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z,
                ]
            )
            q = np.array(
                [
                    transform.transform.rotation.x,
                    transform.transform.rotation.y,
                    transform.transform.rotation.z,
                    transform.transform.rotation.w,
                ]
            )
            rot = R.from_quat(q).as_matrix()
            T = np.eye(4)
            T[0:3, 0:3] = rot
            T[0:3, 3] = t

            pos = np.array(
                [
                    marker.pose.position.x,
                    marker.pose.position.y,
                    marker.pose.position.z,
                    1.0,
                ]
            )
            new_pos = T @ pos

            marker_q = np.array(
                [
                    marker.pose.orientation.x,
                    marker.pose.orientation.y,
                    marker.pose.orientation.z,
                    marker.pose.orientation.w,
                ]
            )
            new_q = R.from_quat(q) * R.from_quat(marker_q)
            new_q = new_q.as_quat()  # [x, y, z, w]

            new_marker = Marker()
            new_marker.header.stamp = self.get_clock().now().to_msg()
            new_marker.header.frame_id = "base_link"
            new_marker.ns = marker.ns
            new_marker.id = marker.id
            new_marker.type = marker.type
            new_marker.action = marker.action
            from geometry_msgs.msg import Point, Quaternion

            new_marker.pose.position = Point(
                x=float(new_pos[0]), y=float(new_pos[1]), z=float(new_pos[2])
            )
            new_marker.pose.orientation = Quaternion(
                x=float(new_q[0]),
                y=float(new_q[1]),
                z=float(new_q[2]),
                w=float(new_q[3]),
            )
            new_marker.scale = marker.scale
            new_marker.color = marker.color
            new_marker.lifetime = marker.lifetime
            new_marker.frame_locked = marker.frame_locked
            new_marker.text = marker.text
            new_marker.mesh_resource = marker.mesh_resource
            new_marker.mesh_use_embedded_materials = marker.mesh_use_embedded_materials

            return new_marker
        except Exception as e:
            self.get_logger().error(
                f"[ERROR] TF transform for marker {marker.id} failed: {e}"
            )
            return None

    def reorient_object_rotation(self, object_quat, width, depth):
        """
        object_quat: [qx, qy, qz, qw] 형태의 물체 쿼터니언 (로컬 orientation)
        width, depth: marker.scale.x, marker.scale.y (물체 치수)

        1. 먼저 물체의 로컬 좌표계를 구성하여, 월드 z축과 일치하도록 정렬합니다.
        2. 그 후, 필수적으로 x축 기준 180° 회전을 적용합니다.
        3. 마지막으로, z축 기준 180° 회전을 적용합니다.

        최종 결과는 aligned_rot -> x축 180° -> z축 180° 순으로 회전이 적용된 orientation을 반환합니다.
        """
        # 1. 기존 로컬 좌표계 정렬 (월드 z축과 맞추기)
        r_orig = R.from_quat(object_quat)
        R_matrix = r_orig.as_matrix()  # 각 열: 로컬 x, y, z 축
        local_axes = [R_matrix[:, 0], R_matrix[:, 1], R_matrix[:, 2]]
        base_z = np.array([0, 0, 1])
        # 각 로컬 축과 월드 z축의 내적(절댓값)으로 가장 일치하는 축 선택
        dots = [abs(np.dot(ax, base_z)) for ax in local_axes]
        max_index = np.argmax(dots)
        new_z = local_axes[max_index]

        # 2. 나머지 축 중 하나 선택하여, new_z 성분 제거 후 new_x 계산
        candidate = local_axes[1] if max_index == 0 else local_axes[0]
        new_x = candidate - np.dot(candidate, new_z) * new_z
        new_x = new_x / np.linalg.norm(new_x)

        # 3. new_z와 new_x의 외적으로 new_y 계산 (오른손 좌표계 유지)
        new_y = np.cross(new_z, new_x)
        new_y = new_y / np.linalg.norm(new_y)

        # 4. 정렬된 회전 행렬 구성 (각 열: new_x, new_y, new_z)
        R_aligned = np.column_stack((new_x, new_y, new_z))
        aligned_rot = R.from_matrix(R_aligned)

        # 5. 필수 x축 기준 180° 회전 (회전각 π)
        rot_x_180 = R.from_euler("x", math.pi)

        # 6. 필수 z축 기준 180° 회전 (회전각 π)
        rot_z_180 = R.from_euler("z", math.pi)

        # 최종 회전 순서: 먼저 aligned_rot, 그 후 x축 180° 회전, 마지막에 z축 180° 회전 적용
        final_rot = rot_z_180 * rot_x_180 * aligned_rot
        return final_rot.as_quat()

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
        elif "플레이스" in user_cmd:
            self.current_mode = "pick_and_place"

        if self.current_mode == "armcontroller":
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
            x, y, z = new_pose[0], new_pose[1], new_pose[2]
            r_rad, p_rad, y_rad = new_pose[3], new_pose[4], new_pose[5]
            r_deg = r_rad * 180.0 / math.pi
            p_deg = p_rad * 180.0 / math.pi
            y_deg = y_rad * 180.0 / math.pi
            msg_str = f"{x:.2f} {y:.2f} {z:.2f} {r_deg:.2f} {p_deg:.2f} {y_deg:.2f}"
            msg = String()
            msg.data = msg_str
            self.follow_pub.publish(msg)
            self.get_logger().info(f"[INFO] Published follow_target command: {msg_str}")
            self.current_pose = new_pose
        except ValueError as e:
            self.get_logger().error(
                f"[ERROR] Failed to parse pose response: {response}. Error: {e}"
            )

    def transform_object_pose(self, obj: dict) -> PoseStamped:
        # PoseStamped 생성 (카메라 프레임)
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = obj["frame_id"]
        # 현재 시간으로 stamp 갱신
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.pose.position.x = obj["x"]
        pose_stamped.pose.position.y = obj["y"]
        pose_stamped.pose.position.z = obj["z"]
        pose_stamped.pose.orientation.x = obj["qx"]
        pose_stamped.pose.orientation.y = obj["qy"]
        pose_stamped.pose.orientation.z = obj["qz"]
        pose_stamped.pose.orientation.w = obj["qw"]

        try:
            # base_link로의 변환 (최대 1초 대기)
            transform = self.tf_buffer.lookup_transform(
                "base_link",
                pose_stamped.header.frame_id,
                self.get_clock().now().to_msg(),
                rclpy.duration.Duration(seconds=1.0),
            )
            transformed_pose = my_do_transform_pose(pose_stamped, transform)
            return transformed_pose
        except Exception as e:
            self.get_logger().error(f"[ERROR] TF transform failed: {e}")
            return None


    def handle_pick_and_place_command(self, user_cmd: str):
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
        # TF 변환을 통해 base_link에서의 물체 위치 계산 (기존 방식)
        transformed_pose = self.transform_object_pose(obj)
        if transformed_pose is None:
            self.get_logger().error("[ERROR] TF transformation failed.")
            return

        # 변환된 위치
        obj_x = transformed_pose.pose.position.x
        obj_y = transformed_pose.pose.position.y
        obj_z = transformed_pose.pose.position.z

        self.get_logger().info(
            f"[DEBUG] Object '{identified_obj}' transformed to base_link frame: "
            f"position=({obj_x:.2f}, {obj_y:.2f}, {obj_z:.2f})"
        )

        # 객체 치수 (width, depth, height)
        width = obj["width"]
        depth = obj["depth"]
        height = obj["height"]
        object_quat = [obj["qx"], obj["qy"], obj["qz"], obj["qw"]]

        # 재정의된 orientation 계산 (회전값은 그대로 유지)
        new_quat = self.reorient_object_rotation(object_quat, width, depth)

        # 픽 동작의 z 오프셋: 물체 높이의 절반에서 약간 감소 (0.03 m)
        pick_offset_z = (height / 2) - 0.03
        pick_x = obj_x
        pick_y = obj_y
        pick_z = obj_z + pick_offset_z
        pick_values = list(new_quat) + [pick_x, pick_y, pick_z]

        # "물체를 어디에 놔" 옵션: 사용자 명령에서 보드 색상을 판별하여 좌표 지정
        # x 좌표는 고정 0.2, y 좌표는 색상에 따라 결정
        if ("빨간" in user_cmd) or ("빨강" in user_cmd):
            place_x = 0.2
            place_y = 0.2
        elif ("초록" in user_cmd) or ("초록색" in user_cmd):
            place_x = 0.2
            place_y = 0.0
        elif ("파란" in user_cmd) or ("파랑" in user_cmd):
            place_x = 0.2
            place_y = -0.2
        else:
            # 색상 정보가 없으면 기본값 (초록색 보드)
            place_x = 0.2
            place_y = 0.0

        # 플레이스 동작의 z 좌표는 물체의 중간 높이를 기준으로 설정 (픽 위치와 동일한 오프셋 적용)
        place_z = obj_z + pick_offset_z

        # 플레이스 동작의 orientation은 고정값 [1, 0, 0, 0] 사용 (필요에 따라 조정)
        place_values = [1.0, 0.0, 0.0, 0.0, place_x, place_y, place_z]

        # 최종 명령 결합: 픽 동작 값 + 플레이스 동작 값
        all_values = pick_values + place_values
        values_str = " ".join([f"{v:.2f}" for v in all_values])
        msg_str = f"{values_str}"
        pub_msg = String()
        pub_msg.data = msg_str
        self.ppcmd_pub.publish(pub_msg)
        self.get_logger().info(f"[INFO] Published ppcmd command: {msg_str}")


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
