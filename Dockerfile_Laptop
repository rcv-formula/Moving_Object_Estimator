# 1. 베이스 이미지 선택: NVIDIA CUDA 이미지 (CUDA, cuDNN 포함)
#    ROS2 Humble은 Ubuntu 22.04 기반이므로, 해당 Ubuntu 버전을 지원하는 CUDA 이미지를 선택합니다.
#    'devel' 태그는 개발 도구 (컴파일러 등)를 포함하여 빌드에 용이합니다.
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# 2. 필요한 환경 변수 설정 (옵션이지만 권장)
#    이것은 컨테이너가 GPU를 인식하고 사용할 수 있도록 합니다.
#    NVIDIA CUDA 이미지에는 이미 설정되어 있을 수 있지만 명시적으로 넣어주는 것이 좋습니다.
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
# 그래픽 관련 기능도 필요할 수 있으므로 추가

# 3. 시스템 업데이트 및 필수 패키지 설치
#    ROS2 설치에 필요한 기본 유틸리티 및 라이브러리 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    locales \
    software-properties-common \
    curl \
    gnupg \
    lsb-release \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libxext6 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

# 4. ROS2 저장소 및 GPG 키 추가 (Humble 기준)
RUN locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && export LANG=en_US.UTF-8

RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# 5. ROS2 설치 

# ROS2 설치시 지역 설정 관련되 부분을 SKIP하기 위함
ENV DEBIAN_FRONTEND=noninteractive
RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen en_US.UTF-8 && \
    /usr/sbin/update-locale LANG=en_US.UTF-8

ENV LANG="en_US.UTF-8" \
    LC_ALL="en_US.UTF-8" \
    LANGUAGE="en_US.UTF-8"

# ROS2 설치 
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-desktop \
    ros-dev-tools \
    ros-humble-tiago-gazebo \
    && rm -rf /var/lib/apt/lists/*

RUN rosdep init && rosdep update

RUN apt update && \
    apt install --no-install-recommends -y build-essential cmake git libbullet-dev \
    python3-colcon-common-extensions python3-flake8 python3-pip \
    python3-pytest-cov python3-rosdep python3-setuptools python3-vcstool \
    wget python3-argcomplete && \
    \
    python3 -m pip install -U flake8-blind-except flake8-builtins \
    flake8-class-newline flake8-comprehensions flake8-deprecated \
    flake8-docstrings flake8-import-order flake8-quotes \
    pytest-repeat pytest-rerunfailures pytest && \
    \
    apt install --no-install-recommends -y libasio-dev libtinyxml2-dev libcunit1-dev \
    && rm -rf /var/lib/apt/lists/*

# 6. ROS2 환경 설정 스크립트 소싱 (매번 수동으로 할 필요 없도록)
COPY bashrc_config.txt /tmp/bashrc_config_temp.txt

# 복사된 파일의 내용을 /root/.bashrc에 추가
# /root/.bashrc 파일이 이미 존재하는 경우, 내용을 덮어쓰지 않고 추가합니다 (>>).
RUN cat /tmp/bashrc_config_temp.txt >> /root/.bashrc && \
    rm /tmp/bashrc_config_temp.txt # 임시 파일 삭제
    
# 7. (선택 사항) 필요한 ROS2 패키지 및 사용자 코드 추가
WORKDIR /moving_object_estimator_ws
COPY . .
RUN apt-get update && rosdep install --from-paths src --ignore-src -r -y && rm -rf /var/lib/apt/lists/*

# 8. 컨테이너 시작 시 기본 명령 설정 (선택 사항)
CMD ["bash"]