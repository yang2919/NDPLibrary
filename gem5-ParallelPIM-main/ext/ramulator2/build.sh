#!/usr/bin/env sh
set -eu

case "$0" in
  */*) script_path=$0 ;;
  *) script_path=$(command -v "$0") ;;
esac

root_dir=$(CDPATH= cd -P "$(dirname "$script_path")" && pwd)
build_dir=${BUILD_DIR:-"$root_dir/build"}
build_type=${BUILD_TYPE:-Release}

if [ -z "${JOBS:-}" ]; then
  if command -v nproc >/dev/null 2>&1; then
    jobs=$(nproc)
  elif command -v getconf >/dev/null 2>&1; then
    jobs=$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '1\n')
  else
    jobs=1
  fi
else
  jobs=$JOBS
fi

case "$jobs" in
  ''|*[!0-9]*) jobs=1 ;;
esac

if [ "$jobs" -lt 1 ]; then
  jobs=1
fi

printf 'Configuring Ramulator in %s (%s)\n' "$build_dir" "$build_type"
cmake -S "$root_dir" -B "$build_dir" -DCMAKE_BUILD_TYPE="$build_type" "$@"

printf 'Building Ramulator with %s job(s)\n' "$jobs"
cmake --build "$build_dir" --parallel "$jobs"
