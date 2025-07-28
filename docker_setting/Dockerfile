FROM python:3

# 환경 설정
ENV DEBIAN_FRONTEND=noninteractive

# 필수 패키지 설치 (Python, pip, C/C++ 컴파일러 등)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Python 기본 설정
RUN ln -s /usr/bin/python3 /usr/bin/python && \
    pip3 install --upgrade pip && pip install uv

# 작업 디렉토리
WORKDIR /root
COPY requirements.txt /root/requirements.txt

# requirements 설치
RUN uv pip install --system -r requirements.txt
# RUN pip install -r requirements.txt

# ENTRYPOINT ["python", "discharge_note.py"]
# ENTRYPOINT ["/bin/bash", "-c", "source /app/.venv/bin/activate && accelerate launch --multi_gpu --num_processes 2 /data/code/CXR_embedding_research/train_with_accelerate.py --seed 42"]