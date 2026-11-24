# Build stock x86 gem5 on the container's native architecture.
FROM ubuntu:22.04@sha256:076e7bf82319619f28dd0aa9c02ab34b055ad37100395b4f49a2ffba04fb5e11
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ca-certificates python3 python3-dev scons pkg-config \
    zlib1g-dev libprotobuf-dev protobuf-compiler libboost-all-dev libhdf5-dev \
    libpng-dev libelf-dev g++-x86-64-linux-gnu && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /opt/gem5
RUN git init && git remote add origin https://github.com/gem5/gem5.git && \
    git fetch --depth 1 origin b9da2bfe1e21f6efa2730edd4e80e58c5d965458 && \
    git checkout --detach FETCH_HEAD
ARG JOBS=2
RUN scons build/X86/gem5.opt -j${JOBS} --ignore-style
WORKDIR /workspace
