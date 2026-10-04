"""FoundationPose 기반 6D 자세 추정 노드.

Isaac Sim 카메라 토픽(RGB, depth, semantic segmentation, semantic labels)을 받아
라벨이 붙은 YCB 물체마다 자세를 추정하고 /object_marker_array 로 발행한다.
처음 보이는 물체는 register(전역 추정), 이후에는 track_one(추적)으로 갱신한다.
"""

import json
import os
import re
import sys

import cv2
import numpy as np
import rclpy
import trimesh
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

sys.path.append("/FoundationPose")
from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor  # noqa: E402
import nvdiffrast.torch as dr  # noqa: E402
import torch  # noqa: E402

class PoseEstimator(Node):
    def __init__(self):
        super().__init__("pose_estimator")
        self.mesh_dir = self.declare_parameter(
            "mesh_dir", "/FoundationPose/demo_data/ycb"
        ).value
        self.register_iter = self.declare_parameter("register_iter", 5).value
        self.track_iter = self.declare_parameter("track_iter", 2).value
        # 입력 이미지 축소 비율 (GPU 메모리 절약, 1.0이면 원본)
        self.image_scale = self.declare_parameter("image_scale", 0.5).value
        # 이 픽셀 수(축소 후 기준)보다 작게 보이는 물체는 무시
        self.min_mask_pixels = self.declare_parameter("min_mask_pixels", 50).value

        self.bridge = CvBridge()
        self.camera_info = None
        self.labels = {}  # segmentation id -> class 이름
        self.objects = {}  # class 이름 -> {"est", "to_origin", "extents"}

        # 네트워크/렌더러는 한 번만 만들어 모든 물체가 공유
        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()

        self.create_subscription(CameraInfo, "/camera/camera_info", self.on_camera_info, 10)
        self.create_subscription(String, "/camera/semantic_labels", self.on_labels, 10)
        self.sync = ApproximateTimeSynchronizer(
            [
                Subscriber(self, Image, "/camera/rgb"),
                Subscriber(self, Image, "/camera/depth"),
                Subscriber(self, Image, "/camera/semantic_segmentation"),
            ],
            queue_size=5,
            slop=0.1,
        )
        self.sync.registerCallback(self.on_images)
        self.marker_pub = self.create_publisher(MarkerArray, "/object_marker_array", 10)
        self.get_logger().info(f"pose_estimator started (meshes: {self.mesh_dir})")

    def on_camera_info(self, msg):
        self.camera_info = msg

    def on_labels(self, msg):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f"semantic_labels parse failed: {e}")
            return
        labels = {}
        for key, value in data.items():
            try:
                labels[int(key)] = value.get("class", "")
            except (ValueError, AttributeError):
                continue
        self.labels = labels

    def on_images(self, rgb_msg, depth_msg, seg_msg):
        if self.camera_info is None or not self.labels:
            return

        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough").astype(np.float32)
        depth[(depth < 0.1) | ~np.isfinite(depth)] = 0
        seg = self.bridge.imgmsg_to_cv2(seg_msg, desired_encoding="passthrough")
        K = np.array(self.camera_info.k).reshape(3, 3)
        if self.image_scale != 1.0:
            s = self.image_scale
            size = (int(rgb.shape[1] * s), int(rgb.shape[0] * s))
            rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
            depth = cv2.resize(depth, size, interpolation=cv2.INTER_NEAREST)
            seg = cv2.resize(seg, size, interpolation=cv2.INTER_NEAREST)
            K = K.copy()
            K[:2] *= s

        markers = MarkerArray()
        for label_id, class_name in self.labels.items():
            mask = seg == label_id
            # 화면에 (충분히) 보이지 않는 물체는 발행하지 않음
            if mask.sum() < self.min_mask_pixels:
                continue

            obj = self.objects.get(class_name)
            pose = None
            if obj is None:
                obj = self.load_object(class_name)
                if obj is None:
                    continue
                self.objects[class_name] = obj
            else:
                pose = obj["est"].track_one(rgb=rgb, depth=depth, K=K, iteration=self.track_iter)
                # 손목 카메라가 크게 움직이면 추적을 놓치므로 마스크와 어긋나면 다시 등록
                if not self.projects_into_mask(pose, obj, K, mask):
                    pose = None

            if pose is None:
                pose = obj["est"].register(
                    K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=self.register_iter
                )
                torch.cuda.empty_cache()
                self.get_logger().info(f"registered {class_name}")

            # 메시 원점 -> 바운딩 박스 중심 (카메라 광학 좌표계)
            center = pose @ np.linalg.inv(obj["to_origin"])
            markers.markers.append(self.make_marker(label_id, class_name, center, obj))

        if markers.markers:
            self.marker_pub.publish(markers)

    @staticmethod
    def projects_into_mask(pose, obj, K, mask):
        """물체 중심을 이미지에 투영했을 때 마스크 영역(20% 여유) 안에 들어오는지."""
        c = (pose @ np.linalg.inv(obj["to_origin"]))[:3, 3]
        if c[2] <= 0:
            return False
        u, v = (K @ c)[:2] / c[2]
        ys, xs = np.nonzero(mask)
        mx = 0.2 * (xs.max() - xs.min() + 1)
        my = 0.2 * (ys.max() - ys.min() + 1)
        return xs.min() - mx <= u <= xs.max() + mx and ys.min() - my <= v <= ys.max() + my

    def load_object(self, class_name):
        name = re.sub(r"^ycb_", "", class_name)
        if not name or not os.path.isdir(self.mesh_dir):
            return None
        for d in sorted(os.listdir(self.mesh_dir)):
            path = os.path.join(self.mesh_dir, d, "google_16k", "textured.obj")
            if name in d and os.path.exists(path):
                break
        else:
            return None

        mesh = trimesh.load(path)
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        est = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=self.scorer,
            refiner=self.refiner,
            glctx=self.glctx,
            debug=0,
            debug_dir="/tmp/foundationpose_debug",
        )
        return {"est": est, "to_origin": to_origin, "extents": extents}

    def make_marker(self, label_id, class_name, pose, obj):
        m = Marker()
        m.header.frame_id = self.camera_info.header.frame_id
        m.id = label_id
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.text = class_name
        x, y, z = pose[:3, 3]
        m.pose.position.x, m.pose.position.y, m.pose.position.z = float(x), float(y), float(z)
        qx, qy, qz, qw = R.from_matrix(pose[:3, :3]).as_quat()
        m.pose.orientation.x, m.pose.orientation.y = float(qx), float(qy)
        m.pose.orientation.z, m.pose.orientation.w = float(qz), float(qw)
        # 바운딩 박스 크기 (m)
        m.scale.x, m.scale.y, m.scale.z = (float(v) for v in obj["extents"])
        m.color.r, m.color.a = 1.0, 0.6
        return m


def main(args=None):
    rclpy.init(args=args)
    node = PoseEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
