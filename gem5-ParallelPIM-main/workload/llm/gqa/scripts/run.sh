#!/bin/sh
# run the GQA decode-block example (see workload/llm/README.md).
# Extra arguments go to workload/common/bench.py.
here=$(dirname "$0")
exec python3 "$here/../../../common/bench.py" run "$here/.." "$@"
