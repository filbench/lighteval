FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

WORKDIR /stage

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    curl \
    unzip \
    git \
    vim

# Install uv
ADD --chmod=755 https://astral.sh/uv/install.sh /install.sh
RUN /install.sh && rm /install.sh

# Install dependencies
COPY pyproject.toml /stage/pyproject.toml
RUN /root/.local/bin/uv pip install --system --no-cache -e ".[dev,vllm]"

COPY . /stage