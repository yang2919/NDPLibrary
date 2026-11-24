#!/bin/sh
# run the gemv microbenchmark (see workload/README.md). Extra arguments go to bench.py.
here=$(dirname "$0")
exec python3 "$here/../../../common/bench.py" run "$here/.." "$@"
