import asyncio
import os

import omni.ui as ui
from omni.isaac.examples.base_sample import BaseSampleExtension
from .lang2pose import Lang2Pose  # 경로 확인 필요
from omni.isaac.ui.ui_utils import btn_builder


class Lang2PoseExtension(BaseSampleExtension):
    def on_startup(self, ext_id: str):
        super().on_startup(ext_id)
        super().start_extension(
            menu_name="",
            submenu_name="",
            name="Lang2Pose",
            title="AISL Robrain Lang2Pose",
            doc_link="https://aisl.sejong.ac.kr/",
            overview=
            """Lang2Pose is an Isaac Sim extension that demonstrates robot control using
            a combination of NVIDIA FoundationPose and a Large Language Model (LLM) API. \n\n
            Natural language instructions are interpreted by the LLM and translated into
            target object poses estimated by FoundationPose. The Kinova Gen3 robot arm is
            then guided in simulation to execute the corresponding manipulation tasks. \n\n
            This extension shows how vision-based 6D pose estimation and language models
            can be integrated for intuitive robot control in both research and teaching.""",
            sample=Lang2Pose(),
            file_path=os.path.abspath(__file__),
            number_of_extra_frames=1,
        )
        self.task_ui_elements = {}
        frame = self.get_frame(index=0)
        self.build_ui(frame)

    def post_reset_button_event(self):
        self.task_ui_elements["Start Simulation"].enabled = True

    def _on_start_simulation_button_event(self):
        asyncio.ensure_future(self.sample._start_physics_simulation())
        self.task_ui_elements["Start Simulation"].enabled = True
        return

    def post_clear_button_event(self):
        self.task_ui_elements["Start Simulation"].enabled = False

    def build_ui(self, frame):
        with frame:
            with ui.VStack(spacing=5):
                frame.title = "Lang2Pose Tasks"
                frame.visible = True
                dict = {
                    "label": "Start Simulation",
                    "type": "button",
                    "text": "Start Simulation",
                    "tooltip": "Turn on ros2 launch files before start simulation.",
                    "on_clicked_fn": self._on_start_simulation_button_event,
                }
                self.task_ui_elements["Start Simulation"] = btn_builder(**dict)
                self.task_ui_elements["Start Simulation"].enabled = True
