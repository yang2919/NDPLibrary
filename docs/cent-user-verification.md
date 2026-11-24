# Reproduce the CENT conformance checks

Run these commands from the library repository root on macOS or Linux. Use a C++20
compiler, CMake, and Python with CPU PyTorch and NumPy. The validation runner
uses POSIX timers. The recorded run used Python 3.14.0 and Torch 2.9.1.

## Set up

Point `cent_source` at `cent_simulation/aim_sim.py` from CENT commit
`4c2aac6ce6aa3ddfe1944fc9c5d656b90d3fc7d2`. The harness uses only this file
from CENT and verifies its SHA-256 before importing it. For a fresh checkout,
the following command fetches the pinned source into the ignored build directory.

```sh
mkdir -p build/cent-upstream
cent_source="$PWD/build/cent-upstream/aim_sim.py"
curl --fail --location --silent --show-error \
  https://raw.githubusercontent.com/Yufeng98/CENT/4c2aac6ce6aa3ddfe1944fc9c5d656b90d3fc7d2/cent_simulation/aim_sim.py \
  --output "$cent_source"
python3 -c 'import numpy, torch; print(numpy.__version__, torch.__version__)'
cmake -S . -B build -DNDP_CENT_SOURCE="$cent_source" -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build build -j 4
ctest --test-dir build -R cent_ --output-on-failure
```

`NDP_CENT_SOURCE` enables the optional replay contract test. The Python decoder
contract test runs when a Python interpreter is available on Unix. Normal library
builds have no dependency on an installed CENT checkout.

## Run a sweep

```sh
output=$(mktemp -d "$PWD/build/cent-run-XXXXXX")
python3 tools/cent/run_cases.py --cent-source "$cent_source" --suite boundary --output "$output"
cat "$output/summary.json"
```

| Suite | Workload |
| --- | --- |
| `smoke` | 44 cases across five geometries, including GEMV, Matmul and biased Linear |
| `boundary` | 870 cases with the default profiles, deterministic random seeds and exact patterns |
| `large` | Two additional GEMVs, `[257,4097]` and `[129,11009]`, on the wide profile |
| `extended` | Boundary and large suites together |

Use a fresh output directory per run. `--limit N` runs only the first N cases and
records that limit. `--profiles tiny,asymmetric` selects GEMV boundary profiles.
The PNM matrix uses the asymmetric profile. Large cases use the wide profile.
The runner runs one export/replay case at a time with a 60-second deadline and
rejects a fixture whose estimated model and fixture storage exceeds 256 MiB.
This estimate excludes Python and Torch process overhead. The producer separately
limits the emulated Shared memory to 64 MiB.

## Inspect one result

```sh
fixture=$(mktemp -d "$PWD/build/cent-case-XXXXXX")
./build/ndp_cent_export "$fixture" 2 2 64 64 5 35 7 exact gemv 1
python3 tools/cent/check_outputs.py "$fixture"
python3 tools/cent/replay.py "$fixture" --cent-source "$cent_source" --mode native
python3 tools/cent/replay.py "$fixture" --cent-source "$cent_source" --mode profile
```

Exporter arguments after the output directory are `C B row_bytes gb_bytes N K
seed pattern operation M`. Patterns are `exact`, `random`, `ones`, and `impulse`.
Operations are `gemv`, `matmul` and `linear`. GEMV uses a single vector. Each
manifest describes the test inputs and geometry. The event log refers to files
containing the actual transport bytes and packed snapshots from the producer.

Each successful sweep writes two rows per case in `report.jsonl` and a final
`summary.json`. `native` runs the original CENT burst MAC. `profile` substitutes
the project reduction order while retaining CENT bank, GB and latch operations.
`max_fp32_reference_error` is the maximum absolute difference from the finite
float64 reference before FP16 readback. `native_different_downloads` counts
random-input readbacks whose FP16 bytes differ from production. The native pass
criterion uses the accumulator error bound. Profile mode checks the download bytes.

`check_outputs.py` also accepts a sweep directory. It reconstructs final outputs
from dense host operands in API order. It checks the full projection chain and
guards independently of the command operands, bank addresses and tile schedule.
A suite passing this check has `logical_downloads` checked against its original
host-level operations, including PNM Matmul or biased Linear where applicable.

On failure retain the fixture, the exception and the failing `report.jsonl` row.
An event/command replay failure also writes `failure-native.json` or
`failure-profile.json` with the event, the current command where available,
and register observations. Fixture contents allow the individual run above
to reproduce the failure. See the [recorded results](cent-conformance-results.md).
