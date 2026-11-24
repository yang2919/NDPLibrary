# PIM guest kernel

The Processing-in-Memory (PIM) Full System (FS) runs that need more than the
stock gem5-resources kernel (`riscv-linux-6.8.12-kernel-1.0.0`) use one
rebuilt Linux 6.8.12 and a copy of the Ubuntu 24.04 disk image
(`riscv-ubuntu-24.04-img-2.0.0`) that holds its modules.

| File | Content |
|---|---|
| `build.sh` | Downloads Linux 6.8.12 from kernel.org (SHA-256 checked), configures it, builds `vmlinux` and the modules, and writes the image copy |
| `config.fragment` | Added to the stock kernel's configuration: `CONFIG_NUMA=y` |
| `senvcfg.patch` | Sets `senvcfg.CBCFE` and `senvcfg.CBIE` = 01 on every hart with Zicbom |

What differs from the stock kernel, and why:

- **Non-uniform memory access (NUMA).** The stock configuration has no
  `CONFIG_NUMA`. Way 2 shows the PIM memory to Linux as NUMA node 1, so the
  kernel needs it.
- **User cache-block operations.** Linux 6.8 sets only `senvcfg.CBZE`
  (user `cbo.zero`). The PIM library's flush protocol runs `cbo.clean` and
  `cbo.flush` in user mode, which the RISC-V cache-block management
  operation (CMO) specification allows only with `senvcfg.CBCFE` = 1. The
  patch sets it, and `CBIE` = 01, so that a user `cbo.inval` runs as a
  flush. The bits are per hart; the patch sets them where Linux sets
  `CBZE`, at boot of each hart.

The configuration is the one embedded in the stock kernel
(`CONFIG_IKCONFIG`), read from the cached resource, plus the fragment, then
`make olddefconfig`. A few symbols of the stock configuration are unknown
to the kernel.org 6.8.12 source (Ubuntu's, for example
`CONFIG_UBUNTU_ODM_DRIVERS` and `CONFIG_SECURITY_APPARMOR_RESTRICT_USERNS`)
and are dropped; `olddefconfig` turns on what NUMA needs.

The image copy is the cached image with `/lib/modules/6.8.12` replaced by
the new modules, including `gem5/gem5_bridge.ko` built from
`util/gem5_bridge` (the `m5` utility's `/dev/gem5_bridge`), as in the stock
image. It is written with `debugfs` on partition 1, so the build needs no
root; `e2fsck -fn` must find it clean, and the cached image's checksum must
not change.

## Use

```sh
util/pim-kernel/build.sh            # fetch, kernel, image; about 5 minutes
util/pim-kernel/build.sh kernel     # or one stage: fetch, kernel, image
```

`WORK` (default `~/pim-kernel`) receives the download, the source (about
2 GB), `vmlinux`, `config`, `modules/` and `riscv-ubuntu-24.04-img-pim`
(5 GB, sparse). Run with them:

```sh
build/RISCV_RAMULATOR2/gem5.opt configs/example/gem5_library/riscv-pim.py \
    --mode fs --kernel ~/pim-kernel/vmlinux \
    --disk-image ~/pim-kernel/riscv-ubuntu-24.04-img-pim ...
```

Checks after a build: `CONFIG_NUMA=y` in the kernel's embedded
configuration, and `riscv_user_isa_enable` writes 0x50 (bits 6 and 4) to
`senvcfg` beside the stock 0x80:

```sh
A=$(riscv64-linux-gnu-nm ~/pim-kernel/vmlinux | \
    awk '$3 == "riscv_user_isa_enable" {print $1}')
riscv64-linux-gnu-objdump -d --start-address=0x$A \
    --stop-address=$(printf '0x%x' $((0x$A + 0x60))) ~/pim-kernel/vmlinux | \
    grep -B1 'csrs.*senvcfg'
```
