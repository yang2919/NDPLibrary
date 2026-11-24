#!/bin/sh
# Compare the GQA example's gem5 runs with the host reference build
# (build.sh --host first). Writes results/check.md.
here=$(dirname "$0")
exec python3 "$here/../../common/check_digest.py" "$here/.."
