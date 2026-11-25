#!/usr/bin/env bash
# Run any mnet tool under a memory cap so a bad model/flag fails cleanly
# instead of triggering the kernel OOM killer (which also kills the shell on
# WSL and can take the whole session down).
#
# Usage: scripts/mnet-capped.sh [--mem-mb N] -- <tool> [args...]
#
# Default cap: 70% of MemTotal on Linux, or MNET_MAX_MEM_MB when exported.
# Two layers of protection:
#   1. MNET_MAX_MEM_MB  - mnet's own budget (activations, workspace, KV cache)
#   2. RLIMIT_AS        - address-space backstop for weight loads and anything
#                         that bypasses the budgeted allocators
set -u

MEM_MB="${MNET_MAX_MEM_MB:-}"
if [ "${1:-}" = "--mem-mb" ] && [ $# -ge 2 ]; then
    MEM_MB="$2"
    shift 2
fi
if [ "${1:-}" = "--" ]; then
    shift
fi
if [ $# -eq 0 ]; then
    echo "usage: $0 [--mem-mb N] -- <tool> [args...]" >&2
    exit 2
fi

if [ -z "$MEM_MB" ]; then
    if [ -r /proc/meminfo ]; then
        total_kb=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
        MEM_MB=$(( total_kb * 70 / 100 / 1024 ))
    else
        MEM_MB=8192
    fi
fi

export MNET_MAX_MEM_MB="$MEM_MB"

# Address space backstop: give the process 125% of the budget so the in-process
# check fires first with a readable message, while a runaway allocation still
# fails instead of swapping the machine to death. POSIX only.
if [ "$(uname -s)" = "Linux" ]; then
    limit_kb=$(( MEM_MB * 128 * 1024 / 100 ))
    # Keep a floor so the loader and libc can start; the in-process budget is
    # what actually enforces the limit for model memory.
    floor_kb=$(( 1024 * 1024 ))
    [ "$limit_kb" -lt "$floor_kb" ] && limit_kb="$floor_kb"
    ulimit -v "$limit_kb" 2>/dev/null || true
fi

echo "mnet-capped: budget=${MEM_MB} MiB" >&2
exec "$@"
