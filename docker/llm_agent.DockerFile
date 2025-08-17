FROM osrf/ros:humble-desktop

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-pip python3-venv python3-dev \
    python3-colcon-common-extensions \
    build-essential git curl pkg-config \
    portaudio19-dev libasound2-dev ffmpeg \
    && apt-get remove -y python3-scipy || true \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install "numpy>=2.0,<3.0" "scipy>=1.11,<2.0"

ENV PIP_NO_CACHE_DIR=1 PIP_ROOT_USER_ACTION=ignore
RUN python3 -m pip install --upgrade pip setuptools wheel

# 2) 파이썬 의존성
RUN python3 -m pip install \
    "langchain>=0.2" \
    "langchain-openai>=0.1.0" \
    "openai>=1.40.0" \
    "tiktoken>=0.7.0" \
    "pydantic>=2.7.0" \
    "fastapi>=0.111.0" "uvicorn>=0.30.0" \
    "python-dotenv>=1.0.1" \
    "httpx[socks]>=0.27.0" \
    "tenacity>=8.3.0" \
    "numpy>=1.26" "pandas>=2.2" \
    "SpeechRecognition==3.10.4" \
    "pyaudio==0.2.14"


SHELL ["/bin/bash", "-lc"]
RUN echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc

WORKDIR /root/llmagent
