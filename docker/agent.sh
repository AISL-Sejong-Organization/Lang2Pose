#!/bin/bash
# LLM 에이전트 실행: docker exec -it lang2pose agent [simrobot|realrobot]
source /opt/ros/humble/setup.bash
source /opt/lang2pose/agent/install/setup.bash
exec ros2 run aiagent "${1:-simrobot}" --ros-args -p use_sim_time:=true
