# =============================================================================
# AiM-Cosim - AiM Simulator + Verilator Co-verification Framework
# Docker Environment (based on PIMPyVerilog v3.0, simplified for C++ focus)
# =============================================================================

# Stage 1: Use official Verilator image
FROM verilator/verilator:v5.024 AS verilator-base

# Stage 2: Build final image
FROM python:3.10-slim

LABEL maintainer="jhlee"
LABEL description="AiM-Cosim: AiM Simulator + Verilator timing & functional co-verification"

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# =============================================================================
# Copy Verilator from official image
# =============================================================================
COPY --from=verilator-base /usr/local/bin/verilator* /usr/local/bin/
COPY --from=verilator-base /usr/local/share/verilator /usr/local/share/verilator

# Verilator's verilated.mk invokes /usr/bin/python3 directly
RUN ln -sf /usr/local/bin/python3 /usr/bin/python3

# =============================================================================
# System Dependencies
# =============================================================================
RUN apt-get update && apt-get install -y \
    # Build essentials
    build-essential \
    g++-12 \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    ca-certificates \
    pkg-config \
    # Verilator runtime dependencies
    perl \
    libfl2 \
    zlib1g \
    ccache \
    numactl \
    bc \
    # Debugging
    gdb \
    vim \
    nano \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Set GCC 12 as default (C++20 support)
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 \
    --slave /usr/bin/g++ g++ /usr/bin/g++-12

# Verify installations
RUN verilator --version && g++ --version && python3 --version

# =============================================================================
# Python Environment
# =============================================================================
WORKDIR /workspace

RUN python3 -m pip install --upgrade pip setuptools wheel --break-system-packages

RUN pip3 install --no-cache-dir --break-system-packages \
    "numpy>=1.21.0" \
    "matplotlib>=3.5.0" \
    pyyaml \
    "pytest>=7.0.0" \
    "torch>=2.0.0" \
    "transformers>=4.40.0" \
    "datasets>=2.0.0"

# =============================================================================
# Environment Configuration
# =============================================================================

# Verilator
ENV VERILATOR_ROOT=/usr/local/share/verilator

# C++20 compiler
ENV CC=gcc-12
ENV CXX=g++-12

# ccache
ENV CCACHE_DIR=/workspace/.ccache
ENV PATH="/usr/lib/ccache:$PATH"

# =============================================================================
# Working Directory Setup
# =============================================================================
WORKDIR /workspace

RUN mkdir -p \
    /workspace/build_mag32 \
    /workspace/.ccache \
    /workspace/extern/aim_simulator

# Default command
CMD ["/bin/bash"]
