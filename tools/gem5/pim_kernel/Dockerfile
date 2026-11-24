# Cross-compile the Linux RISC-V worker; this image does not build or run gem5.
FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    make scons python3 g++-riscv64-linux-gnu binutils-riscv64-linux-gnu \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
