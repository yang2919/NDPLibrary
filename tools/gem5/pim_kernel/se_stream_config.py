"""Run a persistent RISC-V PIM worker in SE with host FIFO-backed stdio."""
import argparse
from pathlib import Path
import stat

from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.cachehierarchies.classic.private_l1_private_l2_cache_hierarchy import PrivateL1PrivateL2CacheHierarchy
from gem5.components.memory import pim
from gem5.components.processors.cpu_types import CPUTypes
from gem5.components.processors.simple_processor import SimpleProcessor
from gem5.isas import ISA
from gem5.resources.resource import BinaryResource
from gem5.simulate.simulator import Simulator

parser = argparse.ArgumentParser(description=__doc__)
for name in ('binary', 'input', 'output', 'layout'):
    parser.add_argument('--' + name, required=True)
parser.add_argument('--cpu', choices=['atomic', 'timing'], default='atomic')
args = parser.parse_args()
for name in (args.input, args.output):
    if not stat.S_ISFIFO(Path(name).stat().st_mode):
        raise ValueError('stdio path must be a FIFO')
memory = pim.pim_memory_system('LPDDR6', channels=16, policy='RoRaBaChCo',
                               num_fmt='fp16', acc_fmt='fp32', layout_file=args.layout)
processor = SimpleProcessor(cpu_type={'atomic': CPUTypes.ATOMIC, 'timing': CPUTypes.TIMING}[args.cpu],
                            isa=ISA.RISCV, num_cores=1)
processor.get_cores()[0].core.isa[0].vlen = 256
board = SimpleBoard(clk_freq='3GHz', processor=processor, memory=memory,
                    cache_hierarchy=PrivateL1PrivateL2CacheHierarchy(
                        l1d_size='32KiB', l1i_size='32KiB', l2_size='256KiB'))
board.set_se_binary_workload(BinaryResource(local_path=args.binary),
                            stdout_file=Path(args.output), stderr_file=Path('guest.stderr'))
# FileResource rejects FIFOs; Process.input accepts host paths directly.
processor.get_cores()[0].core.workload[0].input = args.input
simulator = Simulator(board=board)
simulator.run()
