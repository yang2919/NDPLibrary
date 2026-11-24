"""Boot stock x86 Linux and execute the selected C++ guest probe."""
import argparse
import base64
import textwrap
from pathlib import Path

import m5

from gem5.components.boards.x86_board import X86Board
from gem5.components.cachehierarchies.classic.private_l1_shared_l2_cache_hierarchy import (
    PrivateL1SharedL2CacheHierarchy,
)
from gem5.components.memory.multi_channel import DualChannelDDR4_2400
from gem5.components.processors.cpu_types import CPUTypes
from gem5.components.processors.simple_processor import SimpleProcessor
from gem5.isas import ISA
from gem5.resources.resource import KernelResource, DiskImageResource
from gem5.simulate.exit_event import ExitEvent
from gem5.simulate.simulator import Simulator

parser = argparse.ArgumentParser()
parser.add_argument("--stage", choices=("F0", "F2", "F3", "F4", "F5", "K4"), default="F0")
parser.add_argument("--binary", required=True)
parser.add_argument("--receiver-binary")
parser.add_argument("--resource-directory", required=True)
args = parser.parse_args()
if args.stage in ("F4", "F5") and not args.receiver_binary:
    parser.error("F4/F5 requires --receiver-binary")

board = X86Board(
    clk_freq="3GHz",
    processor=SimpleProcessor(cpu_type=CPUTypes.ATOMIC, isa=ISA.X86, num_cores=1),
    memory=DualChannelDDR4_2400(size="3GiB"),
    cache_hierarchy=PrivateL1SharedL2CacheHierarchy(
        l1d_size="64KiB", l1i_size="64KiB", l2_size="2MiB"
    ),
)
resources = Path(args.resource_directory)
board.set_kernel_disk_workload(
    kernel=KernelResource(local_path=str(resources / "vmlinux-5.4.49")),
    disk_image=DiskImageResource(
        local_path=str(resources / "x86-ubuntu.img"), root_partition="1"
    ),
    kernel_args=board.get_default_kernel_args() + [
        "rw", "init=/root/gem5_init.sh", "IGNORE_M5=1",
        "PATH=/usr/sbin:/usr/bin:/sbin:/bin",
    ],
)
def install_binary(source, destination, delimiter):
    encoded = base64.b64encode(Path(source).read_bytes()).decode("ascii")
    return (
        f"base64 -d > {destination} <<'{delimiter}'\n"
        + "\n".join(textwrap.wrap(encoded, 76)) + f"\n{delimiter}\n"
        + f"chmod +x {destination}\n"
    )

# Run the image's existing init script directly, retaining real Linux FS execution.
readfile = Path(m5.options.outdir) / "guest-smoke.sh"
script = "#!/bin/sh\nset -eu\n"
script += install_binary(args.binary, "/tmp/ndp-fs-smoke", "NDP_BINARY_EOF")
command = "/tmp/ndp-fs-smoke"
if args.stage in ("F4", "F5"):
    script += install_binary(args.receiver_binary, "/tmp/ndp-execution-receiver", "NDP_RECEIVER_EOF")
    command += " /tmp/ndp-execution-receiver"
if args.stage == "K4":
    script += install_binary(Path(__file__).with_name("k4_guest.sh"), "/tmp/k4_guest.sh", "K4_RELAY_EOF")
    command = "/bin/sh /tmp/k4_guest.sh /tmp/ndp-fs-smoke"
readfile.write_text(script + command + "\necho NDP_FS_SCRIPT_COMPLETE\n/sbin/m5 exit\n")
board.readfile = str(readfile.resolve())

def boot_events():
    print(f"{args.stage}: guest requested exit; check serial log for probe markers", flush=True)
    yield True

events = boot_events()
if args.stage == "K4":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from k4_relay import relay_events
    events = relay_events(Path(m5.options.outdir), readfile, 45240)

Simulator(board=board, on_exit_event={ExitEvent.EXIT: events}).run()
