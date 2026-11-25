# Resume extended gem5 development on Linux

Use a Linux host with Docker Engine access, Git, Bash, a C++20 compiler, CMake
3.20+, and Python with development headers and venv. The recipe targets Ubuntu
24.04 with the CPU build of PyTorch 2.9.1. Allocate space on both the workspace
filesystem (gem5 build) and Docker's data filesystem (images, guest disk,
kernel sources and kernel build). Use two shells on the same server, preferably
inside tmux so losing the SSH connection leaves the simulation running.

## Transfer from the Mac

Run `./tools/migration/pack-linux.sh` at the project root. It prints the path
to a single handoff tar file. Substitute that path and your SSH destination:

```sh
ssh USER@SERVER 'mkdir -p ~/ndp-transfer'
scp /path/to/ndp-linux-handoff.tar USER@SERVER:ndp-transfer/
```

The workspace archive contains uncommitted edits, untracked supplier sources
and scripts. The bundle carries committed history. The evidence archive is a
point-in-time copy of compact validation logs. An active simulator's RAM,
sockets and Docker process stay on the Mac. Start a fresh run to reproduce a case on Linux. To collect results that arrive later, rerun the packer after the Mac
run finishes or copy its final logs separately.

## Restore into a new workspace on Linux

```sh
cd ~/ndp-transfer
tar -xf ndp-linux-handoff.tar
sha256sum -c SHA256SUMS
git clone --no-checkout -b "$(cat source-branch.txt)" history.bundle ~/NDPLibrary
cd ~/NDPLibrary
git reset --mixed "$(cat ~/ndp-transfer/source-head.txt)"
tar -xzf ~/ndp-transfer/workspace.tar.gz
git status --short
mkdir -p build/handoff/mac-evidence
tar -xzf ~/ndp-transfer/evidence.tar.gz -C build/handoff/mac-evidence
```

Compare the working tree with source-status.txt. The snapshot preserves current
file contents and deletions; the index is reset to the source commit. Set
`origin` to your usual project remote before fetching or pushing (clone sets it
to the local bundle). Build the host extension, simulator and worker on the
server. Docker creates a new ndp-pim-linux-data volume and obtains the pinned
RISC-V resources. Initial setup needs internet access. Subsequent runs reuse
the server's build tree and volume.

## Build the Linux Python client and prepare Docker

From the Linux project root, with system prerequisites installed (Ubuntu packages
include build-essential, cmake, python3-dev and python3-venv):

```sh
python3 -m venv build/venv
source build/venv/bin/activate
python -m pip install 'torch==2.9.1' --index-url https://download.pytorch.org/whl/cpu
cmake -S . -B build-python -DNDP_BUILD_PYTHON=ON -DPython3_EXECUTABLE="$VIRTUAL_ENV/bin/python"
cmake --build build-python --target ndp_python -j2
export PYTHONPATH="$PWD/build-python/python"
python -c 'import torch, ndp; s=ndp.Session(); s.close(); print(torch.__version__, ndp.__file__)'
docker info >/dev/null
STAGE=PREPARE ./tools/gem5/pim_kernel/run.sh
```

Wait for NDP_PIM_PREPARE_PASS. PREPARE builds and fetches resources without
booting the guest. Docker writes its build outputs into the mounted workspace
as the container user (normally root). Keep build-python and the venv owned by
your host user. The runner defaults to CPUs 0 and 1. On a server with scheduler
affinity restrictions, set CPUSET to allowed CPU IDs and JOBS to their count
when invoking the runner (for example CPUSET=8-9 JOBS=2).

## Check the restored environment

For a two-terminal FS smoke check, follow the GEMV case in the
[external FS client guide](extended-gem5-fs-client.md). Activate the Linux venv
in terminal B and use the same free port in both terminals. Keep the new run
directory to distinguish Linux results from the [recorded Mac runs](extended-gem5-validation-plan.md).
