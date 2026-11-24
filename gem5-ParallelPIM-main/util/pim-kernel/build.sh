#!/bin/bash
# Build the processing-in-memory (PIM) guest kernel and its disk image copy:
# Linux 6.8.12 with the gem5-resources kernel's configuration plus
# config.fragment (CONFIG_NUMA=y) and senvcfg.patch (user cbo.clean, cbo.flush;
# cbo.inval as a flush), its modules, and a copy of the Ubuntu 24.04 image
# whose /lib/modules/6.8.12 holds those modules, written with debugfs (no root;
# the cached image is only read).
#
# Usage: util/pim-kernel/build.sh [fetch|kernel|image|all]   (default all)
#
# Environment (defaults):
#   WORK          $HOME/pim-kernel: downloads, source, outputs
#   STOCK_KERNEL  $HOME/.cache/gem5/riscv-linux-6.8.12-kernel-1.0.0
#   STOCK_IMAGE   $HOME/.cache/gem5/riscv-ubuntu-24.04-img-2.0.0
#   CROSS_COMPILE riscv64-linux-gnu-   (GCC 13.3.0, as the stock kernel)
#
# Outputs in $WORK: vmlinux, config, modules/lib/modules/6.8.12 (with
# util/gem5_bridge as gem5/gem5_bridge.ko, as in the stock image),
# riscv-ubuntu-24.04-img-pim (5 GB), build.log lines on stdout.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${WORK:-$HOME/pim-kernel}
STOCK_KERNEL=${STOCK_KERNEL:-$HOME/.cache/gem5/riscv-linux-6.8.12-kernel-1.0.0}
STOCK_IMAGE=${STOCK_IMAGE:-$HOME/.cache/gem5/riscv-ubuntu-24.04-img-2.0.0}
CROSS_COMPILE=${CROSS_COMPILE:-riscv64-linux-gnu-}
VER=6.8.12
URL=https://cdn.kernel.org/pub/linux/kernel/v6.x
SRC=$WORK/src/linux-$VER
IMAGE=$WORK/riscv-ubuntu-24.04-img-pim
STAGE=${1:-all}
MAKE=(make -C "$SRC" ARCH=riscv CROSS_COMPILE="$CROSS_COMPILE")

fetch() {
    mkdir -p "$WORK/dl"
    cd "$WORK/dl"
    [ -f linux-$VER.tar.xz ] || curl -sSfLO $URL/linux-$VER.tar.xz
    [ -f sha256sums.asc ] || curl -sSfLO $URL/sha256sums.asc
    grep " linux-$VER.tar.xz\$" sha256sums.asc | sha256sum -c -
}

kernel() {
    rm -rf "$SRC"
    mkdir -p "$WORK/src"
    tar -C "$WORK/src" -xf "$WORK/dl/linux-$VER.tar.xz"
    patch -d "$SRC" -p1 < "$HERE/senvcfg.patch"
    # The stock kernel's own configuration (CONFIG_IKCONFIG: a gzip stream
    # after the marker IKCFG_ST).
    python3 -c '
import sys, zlib
d = open(sys.argv[1], "rb").read()
i = d.find(b"IKCFG_ST")
if i < 0:
    sys.exit("no IKCFG_ST in " + sys.argv[1])
z = zlib.decompressobj(16 + zlib.MAX_WBITS)
open(sys.argv[2], "wb").write(z.decompress(d[i + 8:]))
' "$STOCK_KERNEL" "$SRC/.config"
    cp "$SRC/.config" "$WORK/config.stock"
    (cd "$SRC" && ARCH=riscv scripts/kconfig/merge_config.sh -m .config \
        "$HERE/config.fragment")
    "${MAKE[@]}" olddefconfig
    grep -v '^#' "$HERE/config.fragment" | grep . | while read -r line; do
        grep -qx "$line" "$SRC/.config" || {
            echo "build.sh: $line missing from .config" >&2
            exit 1
        }
    done
    "${MAKE[@]}" -j"$(nproc)" vmlinux modules
    rm -rf "$WORK/modules"
    "${MAKE[@]}" INSTALL_MOD_PATH="$WORK/modules" INSTALL_MOD_STRIP=1 \
        modules_install
    # The image's out-of-tree module /lib/modules/$VER/gem5/gem5_bridge.ko
    # (the m5 utility's /dev/gem5_bridge), built from util/gem5_bridge.
    rm -rf "$WORK/gem5_bridge"
    cp -r "$HERE/../gem5_bridge" "$WORK/gem5_bridge"
    "${MAKE[@]}" M="$WORK/gem5_bridge" modules
    "${MAKE[@]}" M="$WORK/gem5_bridge" INSTALL_MOD_PATH="$WORK/modules" \
        INSTALL_MOD_DIR=gem5 INSTALL_MOD_STRIP=1 modules_install
    depmod -b "$WORK/modules" "$VER"
    rm -f "$WORK/modules/lib/modules/$VER/build" \
        "$WORK/modules/lib/modules/$VER/source"
    cp "$SRC/vmlinux" "$WORK/vmlinux"
    cp "$SRC/.config" "$WORK/config"
    "${MAKE[@]}" -s kernelrelease
}

image() {
    local mods=$WORK/modules/lib/modules/$VER
    [ -d "$mods" ] || { echo "build.sh: run the kernel stage first" >&2; exit 1; }
    local sum_before
    sum_before=$(md5sum < "$STOCK_IMAGE")
    cp --sparse=always "$STOCK_IMAGE" "$IMAGE.tmp"
    # The root file system is partition 1.
    local start
    start=$(sfdisk -d "$IMAGE.tmp" | sed -n 's/.*1 : start= *\([0-9]*\),.*/\1/p')
    [ -n "$start" ] || { echo "build.sh: no partition 1" >&2; exit 1; }
    local fs="$IMAGE.tmp?offset=$((start * 512))"
    local tmp
    tmp=$(mktemp -d)
    # Remove the stock modules: files and links first, deepest directories
    # last.
    debugfs -R "rdump /lib/modules/$VER $tmp" "$fs" 2> /dev/null
    {
        (cd "$tmp" && find "$VER" -depth \( -type f -o -type l \) \
            -printf 'rm "/lib/modules/%p"\n')
        (cd "$tmp" && find "$VER" -depth -type d \
            -printf 'rmdir "/lib/modules/%p"\n')
    } > "$tmp/rm.cmds"
    # Then write the new ones, owned by root.
    {
        (cd "$WORK/modules/lib/modules" && find "$VER" | while read -r p; do
            local dst="/lib/modules/$p"
            if [ -L "$p" ]; then
                echo "symlink \"$dst\" \"$(readlink "$p")\""
            elif [ -d "$p" ]; then
                echo "mkdir \"$dst\""
                echo "sif \"$dst\" mode 040755"
            else
                echo "write \"$PWD/$p\" \"$dst\""
                echo "sif \"$dst\" mode 0100644"
            fi
            echo "sif \"$dst\" uid 0"
            echo "sif \"$dst\" gid 0"
        done)
    } > "$tmp/add.cmds"
    debugfs -w -f "$tmp/rm.cmds" "$fs" > "$tmp/rm.log" 2>&1
    debugfs -w -f "$tmp/add.cmds" "$fs" > "$tmp/add.log" 2>&1
    if grep -E -i "error|cannot|not found|exists" "$tmp/rm.log" \
        "$tmp/add.log" | grep -v "^debugfs " | head -5 | grep .; then
        echo "build.sh: debugfs reported errors (logs in $tmp)" >&2
        exit 1
    fi
    e2fsck -fn "$fs" > "$tmp/fsck.log" 2>&1 || {
        echo "build.sh: e2fsck found problems ($tmp/fsck.log)" >&2
        exit 1
    }
    [ "$(md5sum < "$STOCK_IMAGE")" = "$sum_before" ] || {
        echo "build.sh: the stock image changed" >&2
        exit 1
    }
    mv "$IMAGE.tmp" "$IMAGE"
    rm -rf "$tmp"
    echo "build.sh: $IMAGE"
}

case $STAGE in
fetch) fetch ;;
kernel) kernel ;;
image) image ;;
all)
    fetch
    kernel
    image
    ;;
*)
    echo "usage: $0 [fetch|kernel|image|all]" >&2
    exit 2
    ;;
esac
