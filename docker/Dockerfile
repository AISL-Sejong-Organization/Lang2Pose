# Lang2Pose: Isaac Sim 4.2 + ROS 2 Humble + LLM 에이전트 (단일 이미지)
ARG ISAACSIM_VERSION=4.2.0
FROM nvcr.io/nvidia/isaac-sim:${ISAACSIM_VERSION}

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# ROS 2 Humble: LLM 에이전트용 (Isaac Sim은 내장 ROS 2 브리지를 사용)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl gnupg lsb-release tzdata x11-xserver-utils \
        python3-pip python3-dev gcc portaudio19-dev \
    && ln -fs /usr/share/zoneinfo/${TZ} /etc/localtime \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
        > /etc/apt/sources.list.d/ros2.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-ros-base \
        ros-humble-tf2-ros \
        ros-humble-visualization-msgs \
        python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# 에이전트 파이썬 의존성 (시스템 python3, Isaac Sim 내장 python과 별개)
RUN python3 -m pip install --no-cache-dir \
        numpy==2.2.6 \
        scipy==1.15.3 \
        langchain==0.3.27 \
        langchain-openai==0.3.30 \
        openai==1.99.9 \
        python-dotenv==1.1.1 \
        SpeechRecognition==3.10.4 \
        PyAudio==0.2.14

ENV ROS_DISTRO=humble \
    RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    FASTRTPS_DEFAULT_PROFILES_FILE=/etc/lang2pose/fastdds.xml
COPY fastdds_config/fastdds.xml /etc/lang2pose/fastdds.xml

# Isaac Examples 메뉴에 Lang2Pose 예제 등록
RUN sed -i '/^name = "omni.isaac.examples.humanoid"$/a \\n[[python.module]]\nname = "omni.isaac.examples.lang2pose"' \
        /isaac-sim/exts/omni.isaac.examples/config/extension.toml

# 기본 확장/예제 (compose에서 바인드 마운트로 덮어쓸 수 있음)
COPY resources/extensions/aisl.robrain.extension /extensions/aisl.robrain.extension
COPY resources/extensions/lang2pose /isaac-sim/exts/omni.isaac.examples/omni/isaac/examples/lang2pose

# LLM 에이전트 빌드
COPY resources/llmagent/src /opt/lang2pose/agent/src
RUN cd /opt/lang2pose/agent \
    && . /opt/ros/humble/setup.sh \
    && colcon build

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY docker/agent.sh /usr/local/bin/agent
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/agent

WORKDIR /isaac-sim
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
