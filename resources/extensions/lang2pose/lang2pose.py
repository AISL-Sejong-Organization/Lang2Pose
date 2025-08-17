import requests
import json
import aiohttp
from io import BytesIO
import numpy as np
from PIL import Image
from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.core.articulations import Articulation
from omni.isaac.sensor import Camera
import omni.kit.commands
from omni.kit.commands import execute
import omni.graph.core as og
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.physx.scripts import utils
from omni.isaac.core.utils.numpy.rotations import euler_angles_to_quats
from omni.isaac.core.utils.semantics import add_update_semantics
import os
from omni.isaac.core.objects import FixedCuboid
from omni.isaac.core.materials import PhysicsMaterial
import usdrt

from omni.isaac.core.utils import extensions

extensions.enable_extension("aisl.robrain.extension")


class Lang2Pose(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        return

    def setup_scene(self):
        world = self.get_world()

        self.stage = omni.usd.get_context().get_stage()
        self.nucleus_server = get_assets_root_path()
        self.simple_room_usd_path = (
            self.nucleus_server + "/Isaac/Environments/Simple_Room/simple_room.usd"
        )
        self.asset_ycb_path = self.nucleus_server + "/Isaac/Props/YCB/Axis_Aligned"
        self.arm_prim_path = "/World/kinova_robot"
        self.hand_prim_path = "/World/kinova_robot/_f85"
        self.count = 0

        execute(
            "IsaacSimSpawnPrim",
            usd_path="/root/Documents/assets/robot.usd",
            prim_path="/World/kinova_robot",
        )
        execute(
            "IsaacSimSpawnPrim",
            usd_path=self.simple_room_usd_path,
            prim_path="/World/env/simple_room",
            translation=(0.5, 0, 0),
        )

        omni.kit.commands.execute(
            "CreateMdlMaterialPrimCommand",
            mtl_url="http://omniverse-content-production.s3-us-west-2.amazonaws.com/Materials/Base/Metals/Iron.mdl",
            mtl_name="Iron",
            mtl_path="/World/Looks/Iron",
        )

        omni.kit.commands.execute(
            "CreateMdlMaterialPrimCommand",
            mtl_url="http://omniverse-content-production.s3-us-west-2.amazonaws.com/Materials/Base/Textiles/Linen_Beige.mdl",
            mtl_name="Linen_Beige",
            mtl_path="/World/Looks/Linen_Beige",
        )

        omni.kit.commands.execute(
            "CreateMdlMaterialPrimCommand",
            mtl_url="http://omniverse-content-production.s3-us-west-2.amazonaws.com/Materials/Base/Textiles/Linen_Blue.mdl",
            mtl_name="Linen_Blue",
            mtl_path="/World/Looks/Linen_Blue",
        )

        omni.kit.commands.execute(
            "CreateMdlMaterialPrimCommand",
            mtl_url="http://omniverse-content-production.s3-us-west-2.amazonaws.com/Materials/Base/Textiles/Linen_White.mdl",
            mtl_name="Linen_White",
            mtl_path="/World/Looks/Linen_White",
        )

        omni.kit.commands.execute(
            "BindMaterialCommand",
            prim_path="/World/env/simple_room/table_low_327",
            material_path="/World/Looks/Iron",
        )

        omni.kit.commands.execute(
            "SelectPrims",
            old_selected_paths=["/World/env/simple_room/table_low_327"],
            new_selected_paths=["/World/env/simple_room/table_low_327/table_low"],
            expand_in_stage=True,
        )

        omni.kit.commands.execute(
            "BindMaterial",
            material_path="/World/Looks/Iron",
            prim_path=["/World/env/simple_room/table_low_327/table_low"],
            strength=["weakerThanDescendants"],
        )

        material_box1 = PhysicsMaterial(
            prim_path="/World/Looks/Linen_Beige",
            dynamic_friction=0.4,
            static_friction=0.4,
        )
        material_box2 = PhysicsMaterial(
            prim_path="/World/Looks/Linen_Blue",
            dynamic_friction=0.4,
            static_friction=0.4,
        )
        material_box3 = PhysicsMaterial(
            prim_path="/World/Looks/Linen_White",
            dynamic_friction=0.4,
            static_friction=0.4,
        )

        prim = FixedCuboid(
            prim_path="/World/env/box1",
            position=(0.5, 0.0, 0.05),
            scale=(0.3, 0.6, 0.05),
            color=np.array([1.0, 1.0, 1.0]),
            physics_material=material_box1
        )
        prim.set_collision_enabled(True)
        prim.set_collision_approximation("convexDecomposition")

        prim = FixedCuboid(
            prim_path="/World/env/box2",
            position=(0.2, 0.2, 0.05),
            scale=(0.2, 0.2, 0.05),
            color=np.array([1.0, 0.0, 0.0]),
            physics_material=material_box2
        )
        prim.set_collision_enabled(True)
        prim.set_collision_approximation("convexDecomposition")

        prim = FixedCuboid(
            prim_path="/World/env/box3",
            position=(0.2, 0.0, 0.05),
            scale=(0.2, 0.2, 0.05),
            color=np.array([0.0, 0.1, 0.0]),
            physics_material=material_box3
        )
        prim.set_collision_enabled(True)
        prim.set_collision_approximation("convexDecomposition")

        prim = FixedCuboid(
            prim_path="/World/env/box4",
            position=(0.2, -0.2, 0.05),
            scale=(0.2, 0.2, 0.05),
            color=np.array([0.0, 0.0, 1.0]),
            physics_material=material_box3,
        )
        prim.set_collision_enabled(True)
        prim.set_collision_approximation("convexDecomposition")

        omni.kit.commands.execute(
            "BindMaterialCommand",
            prim_path="/World/env/box1",
            material_path="/World/Looks/Linen_Beige",
        )

        self.asset_paths = [
            {
                "name": "gelatin_box",
                "usd_path": self.asset_ycb_path + "/009_gelatin_box.usd",
                "position": (0.5, -0.2, 0.13),
                "orientation": euler_angles_to_quats(
                    np.deg2rad(np.array([0, -90, 0]))
                ),
            },
            {
                "name": "sugar_box",
                "usd_path": self.asset_ycb_path + "/004_sugar_box.usd",
                "position": (0.5, -0.0, 0.20),
                "orientation": euler_angles_to_quats(
                    np.deg2rad(np.array([-90, -7, 10]))
                ),
            },
            {
                "name": "tomato_soup_can",
                "usd_path": self.asset_ycb_path + "/005_tomato_soup_can.usd",
                "position": (0.5, 0.15, 0.14),
                "orientation": euler_angles_to_quats(
                    np.deg2rad(np.array([-90, 0, 0]))
                ),
            },
        ]
        for asset in self.asset_paths:
            execute(
                "IsaacSimSpawnPrim",
                usd_path=asset["usd_path"],
                prim_path=f"/World/env/{asset['name']}",
                translation=asset["position"],
                rotation=(
                    asset["orientation"][3],
                    asset["orientation"][0],
                    asset["orientation"][1],
                    asset["orientation"][2],
                ),
            )
            prim = self.stage.GetPrimAtPath(f"/World/env/{asset['name']}")
            utils.setRigidBody(prim, "convexDecomposition", False)
            label = f"ycb_{asset['name']}"
            add_update_semantics(prim, semantic_label=label, type_label="class")

        self.robot_articulation = Articulation(prim_path=self.arm_prim_path)
        world.scene.add(self.robot_articulation)

        self.camera = Camera(
            prim_path="/World/kinova_robot/bracelet_with_vision_link/Camera",
            name="camera",
        )
        self.camera.initialize()
        self.camera.add_distance_to_image_plane_to_frame()

        og.Controller.edit(
            {"graph_path": "/pick_and_place", "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("on_playback_tick", "omni.graph.action.OnPlaybackTick"),
                    ("gripper_open_value", "omni.graph.nodes.ConstantDouble"),
                    ("gripper_close_value", "omni.graph.nodes.ConstantDouble"),
                    ("gripper_select_if", "omni.graph.nodes.SelectIf"),
                    (
                        "gripper_write_prim_attribute",
                        "omni.graph.nodes.WritePrimAttribute",
                    ),
                    ("gripper_attribute_name", "omni.graph.nodes.ConstantToken"),
                    ("gripper_path", "omni.graph.nodes.ConstantToken"),
                    (
                        "pickandplace_recv_custom_event",
                        "omni.graph.action.OnCustomEvent",
                    ),
                    ("manipulator_path", "omni.graph.nodes.ConstantToken"),
                    ("pickandplace_read_attribute", "omni.graph.core.ReadVariable"),
                    ("pickandplace_write_attribute", "omni.graph.core.WriteVariable"),
                    ("boolean_or", "omni.graph.nodes.BooleanOr"),
                    ("pickandplace_command_not", "omni.graph.nodes.ConstantToken"),
                    ("pickandplace_command", "omni.graph.nodes.ConstantToken"),
                    ("pickandplace_select_if", "omni.graph.nodes.SelectIf"),
                    (
                        "pickandplace_send_custom_event",
                        "omni.graph.action.SendCustomEvent",
                    ),
                    (
                        "pickandplace_recv_custom_event_01",
                        "omni.graph.action.OnCustomEvent",
                    ),
                    ("target_follow_node", "aisl.robrain.extension.targetfollownode"),
                    ("script_node", "omni.graph.scriptnode.ScriptNode"),
                    ("constant_token", "omni.graph.nodes.ConstantToken"),
                    ("constant_token_01", "omni.graph.nodes.ConstantToken"),
                    ("write_prim_attribute", "omni.graph.nodes.WritePrimAttribute"),
                    ("write_prim_attribute_01", "omni.graph.nodes.WritePrimAttribute"),
                    (
                        "pick_and_place_node_01",
                        "aisl.robrain.extension.pickandplacenode",
                    ),
                ],
                og.Controller.Keys.CONNECT: [
                    (
                        "pick_and_place_node_01.outputs:gripper_grasp_command",
                        "gripper_select_if.inputs:condition",
                    ),
                    (
                        "gripper_close_value.inputs:value",
                        "gripper_select_if.inputs:ifFalse",
                    ),
                    (
                        "gripper_open_value.inputs:value",
                        "gripper_select_if.inputs:ifTrue",
                    ),
                    (
                        "on_playback_tick.outputs:tick",
                        "gripper_write_prim_attribute.inputs:execIn",
                    ),
                    (
                        "gripper_attribute_name.inputs:value",
                        "gripper_write_prim_attribute.inputs:name",
                    ),
                    (
                        "gripper_path.inputs:value",
                        "gripper_write_prim_attribute.inputs:primPath",
                    ),
                    (
                        "gripper_select_if.outputs:result",
                        "gripper_write_prim_attribute.inputs:value",
                    ),
                    (
                        "on_playback_tick.outputs:tick",
                        "pickandplace_write_attribute.inputs:execIn",
                    ),
                    (
                        "script_node.outputs:ppcmd",
                        "pickandplace_write_attribute.inputs:value",
                    ),
                    (
                        "pick_and_place_node_01.outputs:pick_and_place_command",
                        "boolean_or.inputs:a",
                    ),
                    (
                        "pickandplace_read_attribute.outputs:value",
                        "boolean_or.inputs:b",
                    ),
                    (
                        "boolean_or.outputs:result",
                        "pickandplace_select_if.inputs:condition",
                    ),
                    (
                        "pickandplace_command_not.inputs:value",
                        "pickandplace_select_if.inputs:ifFalse",
                    ),
                    (
                        "pickandplace_command.inputs:value",
                        "pickandplace_select_if.inputs:ifTrue",
                    ),
                    (
                        "pickandplace_select_if.outputs:result",
                        "pickandplace_send_custom_event.inputs:eventName",
                    ),
                    (
                        "on_playback_tick.outputs:tick",
                        "pickandplace_send_custom_event.inputs:execIn",
                    ),
                    (
                        "pickandplace_recv_custom_event_01.outputs:execOut",
                        "target_follow_node.inputs:execIn",
                    ),
                    (
                        "constant_token.inputs:value",
                        "target_follow_node.inputs:robot_prim_path",
                    ),
                    (
                        "constant_token_01.inputs:value",
                        "target_follow_node.inputs:target_prim_path",
                    ),
                    ("on_playback_tick.outputs:tick", "script_node.inputs:execIn"),
                    (
                        "script_node.outputs:execOut",
                        "write_prim_attribute.inputs:execIn",
                    ),
                    (
                        "constant_token_01.inputs:value",
                        "write_prim_attribute.inputs:primPath",
                    ),
                    (
                        "script_node.outputs:target_pos",
                        "write_prim_attribute.inputs:value",
                    ),
                    (
                        "script_node.outputs:execOut",
                        "write_prim_attribute_01.inputs:execIn",
                    ),
                    (
                        "constant_token_01.inputs:value",
                        "write_prim_attribute_01.inputs:primPath",
                    ),
                    (
                        "script_node.outputs:target_ori",
                        "write_prim_attribute_01.inputs:value",
                    ),
                    (
                        "pickandplace_recv_custom_event.outputs:execOut",
                        "pick_and_place_node_01.inputs:execIn",
                    ),
                    (
                        "script_node.outputs:grasp_point_ori",
                        "pick_and_place_node_01.inputs:grasp_point_ori",
                    ),
                    (
                        "script_node.outputs:grasp_point_pos",
                        "pick_and_place_node_01.inputs:grasp_point_pos",
                    ),
                    (
                        "script_node.outputs:place_point_ori",
                        "pick_and_place_node_01.inputs:place_point_ori",
                    ),
                    (
                        "script_node.outputs:place_point_pos",
                        "pick_and_place_node_01.inputs:place_point_pos",
                    ),
                    (
                        "manipulator_path.inputs:value",
                        "pick_and_place_node_01.inputs:robot_prim_path",
                    ),
                ],
                og.Controller.Keys.CREATE_ATTRIBUTES: [
                    ("script_node.outputs:grasp_point_ori", "double[4]"),
                    ("script_node.outputs:grasp_point_pos", "double[3]"),
                    ("script_node.outputs:place_point_ori", "double[4]"),
                    ("script_node.outputs:place_point_pos", "double[3]"),
                    ("script_node.outputs:ppcmd", "bool"),
                    ("script_node.outputs:target_pos", "double[3]"),
                    ("script_node.outputs:target_ori", "double[4]"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("gripper_open_value.inputs:value", 40.0),
                    ("gripper_close_value.inputs:value", 0.0),
                    ("gripper_write_prim_attribute.inputs:usePath", True),
                    (
                        "gripper_attribute_name.inputs:value",
                        "drive:angular:physics:targetPosition",
                    ),
                    (
                        "gripper_path.inputs:value",
                        "/World/kinova_robot/robotiq_edited/Robotiq_2F_85/finger_joint",
                    ),
                    ("pickandplace_recv_custom_event.inputs:eventName", "pickandplace"),
                    ("manipulator_path.inputs:value", "/World/kinova_robot"),
                    ("pickandplace_read_attribute.inputs:variableName", "pickandplace"),
                    (
                        "pickandplace_write_attribute.inputs:variableName",
                        "pickandplace",
                    ),
                    ("pickandplace_command_not.inputs:value", "target_follow"),
                    ("pickandplace_command.inputs:value", "pickandplace"),
                    (
                        "pickandplace_recv_custom_event_01.inputs:eventName",
                        "target_follow",
                    ),
                    ("script_node.inputs:usePath", True),
                    (
                        "script_node.inputs:scriptPath",
                        "/isaac-sim/exts/omni.isaac.examples/omni/isaac/examples/lang2pose/ppcmd_receiver.py",
                    ),
                    ("constant_token.inputs:value", "/World/kinova_robot"),
                    (
                        "constant_token_01.inputs:value",
                        "/World/kinova_robot/target_point",
                    ),
                    ("write_prim_attribute.inputs:name", "xformOp:translate"),
                    ("write_prim_attribute.inputs:usePath", True),
                    ("write_prim_attribute_01.inputs:name", "xformOp:orient"),
                    ("write_prim_attribute_01.inputs:usePath", True),
                    ("pickandplace_read_attribute.inputs:graph", "/pick_and_place"),
                    ("pickandplace_read_attribute.inputs:variableName", "pickandplace"),
                    ("pickandplace_write_attribute.inputs:graph", "/pick_and_place"),
                    (
                        "pickandplace_write_attribute.inputs:variableName",
                        "pickandplace",
                    ),
                ],
                og.Controller.Keys.CREATE_VARIABLES: [
                    ("pickandplace", "bool"),
                ],
            },
        )

        try:
            og.Controller.edit(
                {"graph_path": "/camera_tf", "evaluator_name": "execution"},
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("OnTick", "omni.graph.action.OnTick"),
                        ("sim_time", "omni.isaac.core_nodes.IsaacReadSimulationTime"),
                        ("ros2_context", "omni.isaac.ros2_bridge.ROS2Context"),
                        ("tf", "omni.isaac.ros2_bridge.ROS2PublishTransformTree"),
                        (
                            "tf_base_link_prim_path",
                            "omni.replicator.core.OgnGetPrimAtPath",
                        ),
                        (
                            "tf_camera_link_prim_path",
                            "omni.replicator.core.OgnGetPrimAtPath",
                        ),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        ("OnTick.inputs:onlyPlayback", True),
                        (
                            "tf_base_link_prim_path.inputs:paths",
                            ["/World/kinova_robot/base_link"],
                        ),
                        (
                            "tf_camera_link_prim_path.inputs:paths",
                            ["/World/kinova_robot/bracelet_with_vision_link/Camera"],
                        ),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("OnTick.outputs:tick", "tf_base_link_prim_path.inputs:execIn"),
                        (
                            "OnTick.outputs:tick",
                            "tf_camera_link_prim_path.inputs:execIn",
                        ),
                        ("sim_time.outputs:simulationTime", "tf.inputs:timeStamp"),
                        ("OnTick.outputs:tick", "tf.inputs:execIn"),
                        ("ros2_context.outputs:context", "tf.inputs:context"),
                        (
                            "tf_base_link_prim_path.outputs:prims",
                            "tf.inputs:parentPrim",
                        ),
                        (
                            "tf_camera_link_prim_path.outputs:prims",
                            "tf.inputs:targetPrims",
                        ),
                    ],
                },
            )
        except Exception as e:
            print(e)

        og.Controller.edit(
            {"graph_path": "/camera", "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("on_playback_tick", "omni.graph.action.OnPlaybackTick"),
                    ("ros2_camera_helper", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                    (
                        "ros2_camera_helper_01",
                        "omni.isaac.ros2_bridge.ROS2CameraHelper",
                    ),
                    (
                        "isaac_set_camera",
                        "omni.isaac.core_nodes.IsaacSetCameraOnRenderProduct",
                    ),
                    (
                        "isaac_create_render_product",
                        "omni.isaac.core_nodes.IsaacCreateRenderProduct",
                    ),
                    (
                        "isaac_run_one_simulation_frame",
                        "omni.isaac.core_nodes.OgnIsaacRunOneSimulationFrame",
                    ),
                    ("ros2_context", "omni.isaac.ros2_bridge.ROS2Context"),
                    (
                        "ros2_camera_info_helper",
                        "omni.isaac.ros2_bridge.ROS2CameraInfoHelper",
                    ),
                    ("constant_string", "omni.graph.nodes.ConstantString"),
                    (
                        "ros2_camera_helper_02",
                        "omni.isaac.ros2_bridge.ROS2CameraHelper",
                    ),
                ],
                og.Controller.Keys.CONNECT: [
                    (
                        "ros2_context.outputs:context",
                        "ros2_camera_helper.inputs:context",
                    ),
                    (
                        "isaac_set_camera.outputs:execOut",
                        "ros2_camera_helper.inputs:execIn",
                    ),
                    (
                        "constant_string.inputs:value",
                        "ros2_camera_helper.inputs:nodeNamespace",
                    ),
                    (
                        "isaac_create_render_product.outputs:renderProductPath",
                        "ros2_camera_helper.inputs:renderProductPath",
                    ),
                    (
                        "ros2_context.outputs:context",
                        "ros2_camera_helper_01.inputs:context",
                    ),
                    (
                        "isaac_set_camera.outputs:execOut",
                        "ros2_camera_helper_01.inputs:execIn",
                    ),
                    (
                        "constant_string.inputs:value",
                        "ros2_camera_helper_01.inputs:nodeNamespace",
                    ),
                    (
                        "isaac_create_render_product.outputs:renderProductPath",
                        "ros2_camera_helper_01.inputs:renderProductPath",
                    ),
                    (
                        "isaac_create_render_product.outputs:execOut",
                        "isaac_set_camera.inputs:execIn",
                    ),
                    (
                        "isaac_create_render_product.outputs:renderProductPath",
                        "isaac_set_camera.inputs:renderProductPath",
                    ),
                    (
                        "isaac_run_one_simulation_frame.outputs:step",
                        "isaac_create_render_product.inputs:execIn",
                    ),
                    (
                        "on_playback_tick.outputs:tick",
                        "isaac_run_one_simulation_frame.inputs:execIn",
                    ),
                    (
                        "ros2_context.outputs:context",
                        "ros2_camera_info_helper.inputs:context",
                    ),
                    (
                        "isaac_set_camera.outputs:execOut",
                        "ros2_camera_info_helper.inputs:execIn",
                    ),
                    (
                        "constant_string.inputs:value",
                        "ros2_camera_info_helper.inputs:nodeNamespace",
                    ),
                    (
                        "isaac_create_render_product.outputs:renderProductPath",
                        "ros2_camera_info_helper.inputs:renderProductPath",
                    ),
                    (
                        "ros2_context.outputs:context",
                        "ros2_camera_helper_02.inputs:context",
                    ),
                    (
                        "isaac_set_camera.outputs:execOut",
                        "ros2_camera_helper_02.inputs:execIn",
                    ),
                    (
                        "constant_string.inputs:value",
                        "ros2_camera_helper_02.inputs:nodeNamespace",
                    ),
                    (
                        "isaac_create_render_product.outputs:renderProductPath",
                        "ros2_camera_helper_02.inputs:renderProductPath",
                    ),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("ros2_camera_helper.inputs:frameId", "Camera"),
                    ("ros2_camera_helper.inputs:topicName", "rgb"),
                    ("ros2_camera_helper_01.inputs:frameId", "Camera"),
                    ("ros2_camera_helper_01.inputs:topicName", "depth"),
                    ("ros2_camera_helper_01.inputs:type", "depth"),
                    ("ros2_camera_info_helper.inputs:frameId", "Camera"),
                    ("constant_string.inputs:value", "camera"),
                    ("ros2_camera_helper_02.inputs:enableSemanticLabels", True),
                    ("ros2_camera_helper_02.inputs:frameId", "Camera"),
                    ("ros2_camera_helper_02.inputs:topicName", "semantic_segmentation"),
                    ("ros2_camera_helper_02.inputs:type", "semantic_segmentation"),
                    (
                        "isaac_create_render_product.inputs:cameraPrim",
                        [
                            usdrt.Sdf.Path(
                                "/World/kinova_robot/bracelet_with_vision_link/Camera"
                            )
                        ],
                    ),
                    (
                        "isaac_set_camera.inputs:cameraPrim",
                        [
                            usdrt.Sdf.Path(
                                "/World/kinova_robot/bracelet_with_vision_link/Camera"
                            )
                        ],
                    ),
                ],
            },
        )

    async def setup_post_load(self):
        self._world = self.get_world()
        self.stage = omni.usd.get_context().get_stage()
        self.arm_prim = self.stage.GetPrimAtPath(self.arm_prim_path)
        return

    async def _start_physics_simulation(self):
        world = self.get_world()
        # Add a physics step callback to control the robot
        world.add_physics_callback("sim_step", self._on_physics_step)
        await world.play_async()
        return

    def _on_physics_step(self, step_size):
        return

    async def setup_pre_reset(self):
        world = self.get_world()
        # Remove the physics step callback before resetting the simulation
        self.count = 0
        if world.physics_callback_exists("sim_step"):
            world.remove_physics_callback("sim_step")

    def world_cleanup(self):
        # Implement any necessary cleanup logic here
        self.count = 0
        pass

    async def take_picture(self):
        url = "http://203.250.148.120:20520/reconstruct"
        images = []
        for idx, picture in enumerate(self.pictures):
            img_byte_arr = BytesIO()
            picture.save(img_byte_arr, format="JPEG")
            img_byte_arr.seek(0)
            images.append(
                ("files", ("filename{}.jpg".format(idx), img_byte_arr, "image/jpeg"))
            )

        data = {
            "image_size": 512,
            "min_conf_thr": 3,
            "as_pointcloud": "True",
            "mask_sky": "True",
            "clean_depth": "True",
            "transparent_cams": "False",
            "cam_size": 0.05,
            "scenegraph_type": "swin",
            "winsize": 32,
            "refid": 1,
            "schedule": "linear",
            "niter": 300,
        }

        async with aiohttp.ClientSession() as session:
            print("Sending request...")
            form = aiohttp.FormData()
            for name, file_info in images:
                form.add_field(
                    name, file_info[1], filename=file_info[0], content_type=file_info[2]
                )
            for key, value in data.items():
                form.add_field(key, str(value))

            async with session.post(url, data=form) as resp:
                if resp.status == 200:
                    content = await resp.read()
                    with open("/home/hyeonsu/Documents/assets/pointcloud.glb", "wb") as f:
                        f.write(content)
                    print("Success")
                else:
                    print("Error: ", resp.status)
                    print(await resp.text())

    def save_mask_images(self):
        """
        Mask 이미지에서 각 semantic label에 해당하는 이미지를 0, 1로 표현되는 PNG 파일로 저장합니다.
        """
        mask_array = np.array(self.mask)  # self.mask는 ndarray 형식으로 가정합니다.
        self.mask_images = []  # Reset the list to store mask images
        self.labels = []  # Reset the list to store corresponding labels

        debug_directory = "/home/hyeonsu/.local/share/ov/pkg/isaac-sim-4.0.0/exts/omni.isaac.examples/omni/isaac/examples/kinova_control/debug"
        os.makedirs(debug_directory, exist_ok=True)

        for label, label_id in self.semantic_labels.items():
            # Label에 해당하는 마스크를 추출 (해당 레이블 ID는 1, 나머지는 0으로 설정)
            mask_label = np.where(mask_array == label_id, 255, 0).astype(np.uint8)

            # 마스크 이미지를 Pillow 이미지로 변환
            mask_image = Image.fromarray(mask_label)

            # 디버깅을 위해 마스크 이미지를 저장 (0과 1로 표현)
            mask_filename = os.path.join(debug_directory, f"mask_{label}.png")
            mask_image.save(mask_filename)

            # 리스트에 저장하여 나중에 서버로 전송할 수 있도록 준비
            self.mask_images.append(mask_filename)
            self.labels.append(label)

        print("Mask images saved for debugging.")

    def send_image(self):
        url = "http://203.250.148.120:20520/test_images/"

        # 디버그 디렉토리에 저장
        debug_directory = "/home/hyeonsu/.local/share/ov/pkg/isaac-sim-4.0.0/exts/omni.isaac.examples/omni/isaac/examples/kinova_control/debug"
        os.makedirs(debug_directory, exist_ok=True)

        # RGB 이미지 파일 저장
        rgb_filename = os.path.join(debug_directory, "rgb.png")
        self.rgb.save(rgb_filename)

        # Depth 이미지 파일 저장
        if self.depth.mode == 'F':
            self.depth = self.depth.convert('I;16')
        # scaling
        depth_image = np.array(self.depth)
        depth_image *= 1000
        self.depth = Image.fromarray(depth_image)
        depth_filename = os.path.join(debug_directory, "depth.png")
        self.depth.save(depth_filename)

        # Mask 이미지 저장 및 변환
        self.mask_images = []
        self.save_mask_images()

        # 파일과 데이터를 전송할 딕셔너리 생성
        files = [
            ("files", open(rgb_filename, "rb")),
            ("files", open(depth_filename, "rb")),
        ]

        # Mask 이미지 파일 추가
        for mask_filename in self.mask_images:
            files.append(("files", open(mask_filename, "rb")))
        print(self.labels, self.camera_intrinsics.tolist())
        # 추가 데이터를 전송할 딕셔너리 생성
        data = {
            "semantic_labels": json.dumps(self.labels),
            "K_matrix": json.dumps(self.camera_intrinsics.tolist()),
            "debug": 2,
            "iteration": 5,
            "IS_BGR": False
        }

        # 동기적으로 HTTP POST 요청 보내기
        response = requests.post(url, files=files, data=data)

        # 응답 처리
        print(response.status_code)
        print(response.json())
