#!/usr/bin/env bash
# Snapshot the working tree, its Git base, and compact FS validation evidence.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
mkdir -p build/handoff
out=$(mktemp -d "$root/build/handoff/linux-XXXXXXXX")
git bundle create "$out/history.bundle" HEAD --branches --tags
git status --short -- . ':(exclude)**/.DS_Store' ':(exclude).DS_Store' > "$out/source-status.txt"
git rev-parse HEAD > "$out/source-head.txt"
git branch --show-current > "$out/source-branch.txt"
export COPYFILE_DISABLE=1
tar -czf "$out/workspace.tar.gz" --exclude='./.git' --exclude='./build*' \
    --exclude='./.venv*' --exclude='./dist' --exclude='.DS_Store' \
    --exclude='__pycache__' --exclude='./gem5-ParallelPIM-main/libpim/build' .
mkdir "$out/evidence"
for run in build/extended-gem5/run-*; do
    [[ -d "$run" ]] || continue
    dest="$out/evidence/${run##*/}"
    mkdir "$dest"
    for name in result.txt options.txt cases.json artifacts.sha256 simulation.log driver.log stream.log; do
        [[ ! -f "$run/$name" ]] || cp "$run/$name" "$dest/"
    done
done
tar -czf "$out/evidence.tar.gz" -C "$out/evidence" .
(cd "$out" && shasum -a 256 workspace.tar.gz history.bundle evidence.tar.gz source-*.txt > SHA256SUMS)
cp docs/linux-server-handoff.md "$out/"
tar -cf "$out/ndp-linux-handoff.tar" -C "$out" workspace.tar.gz history.bundle evidence.tar.gz \
    source-head.txt source-branch.txt source-status.txt SHA256SUMS linux-server-handoff.md
echo "$out/ndp-linux-handoff.tar"
