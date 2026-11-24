#!/bin/sh
# Bridge m5 files to one persistent worker without a guest Python runtime.
set -eu
worker=$1
m5=${2:-/sbin/m5}
tmp=$(mktemp -d)
cd "$tmp"
trap 'rm -rf "$tmp"' EXIT
mkfifo requests replies
"$worker" < requests > replies &
pid=$!
trap 'kill "$pid" 2>/dev/null || :; rm -rf "$tmp"' EXIT
exec 3>requests 4<replies
printf 'K4 guest ready\n' > k4-ready.txt
"$m5" writefile k4-ready.txt
while :; do
    "$m5" exit
    "$m5" readfile > frame
    cat frame >&3
    dd bs=1 count=4 2>/dev/null <&4 > k4-response.bin
    size=$(od -An -tu4 k4-response.bin | tr -d ' ')
    test "$(wc -c < k4-response.bin)" -eq 4
    test "$size" -ge 1 && test "$size" -le 16777216 || exit 1
    head -c "$size" <&4 >> k4-response.bin
    test "$(wc -c < k4-response.bin)" -eq "$((size + 4))"
    if [ "$(wc -c < frame)" -eq 5 ] && [ "$(od -An -tu1 -j4 frame | tr -d ' ')" = 81 ]; then
        exec 3>&-
        wait "$pid"
        trap 'rm -rf "$tmp"' EXIT
        "$m5" writefile k4-response.bin
        echo K4_GUEST_WORKER_CLOSED
        exit 0
    fi
    "$m5" writefile k4-response.bin
done
