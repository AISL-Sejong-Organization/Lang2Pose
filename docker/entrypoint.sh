#!/bin/bash
# Isaac Sim 실행
# - /extensions 아래 폴더를 확장 검색 경로로 등록 (바인드 마운트한 확장이 자동으로 잡힘)
# - ENABLE_EXTS(공백 구분)에 적힌 확장을 켬
set -e

# 내장 ROS 2 브리지 (Isaac Sim 프로세스 전용)
export AMENT_PREFIX_PATH=/isaac-sim/exts/omni.isaac.ros2_bridge/humble
export LD_LIBRARY_PATH=/isaac-sim/exts/omni.isaac.ros2_bridge/humble/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

args=(--allow-root --ext-folder /extensions --enable omni.isaac.ros2_bridge)
for ext in ${ENABLE_EXTS}; do
    args+=(--enable "$ext")
done

exec /isaac-sim/isaac-sim.sh "${args[@]}" "$@"
