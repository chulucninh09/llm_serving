#!/usr/bin/env bash
# Add swap so vLLM native KV offload + workers are less likely to OOM on small-RAM hosts.
# Requires root (run: sudo ./scripts/ensure_swap.sh).
set -euo pipefail

SWAPFILE="${SWAPFILE:-/swapfile}"
SIZE_GB="${SIZE_GB:-32}"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Run as root: sudo $0" >&2
  exit 1
fi

if swapon --show 2>/dev/null | grep -qE "^${SWAPFILE}[[:space:]]"; then
  echo "Already active: $SWAPFILE"
  swapon --show
  exit 0
fi

if [[ ! -f "$SWAPFILE" ]]; then
  echo "Creating ${SIZE_GB}G swap file at $SWAPFILE ..."
  if command -v fallocate >/dev/null; then
    fallocate -l "${SIZE_GB}G" "$SWAPFILE"
  else
    dd if=/dev/zero of="$SWAPFILE" bs=1M count=$((SIZE_GB * 1024)) status=progress
  fi
  chmod 600 "$SWAPFILE"
  mkswap "$SWAPFILE"
fi

echo "Enabling swap on $SWAPFILE"
swapon "$SWAPFILE"
swapon --show
sudo sysctl vm.swappiness=10
echo "Persist across reboots: add this line to /etc/fstab:"
echo "  $SWAPFILE none swap sw 0 0"
