#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

RUNSCRIPT="markowitz_scale_runscript.sh"
LOG_FILE="markowitz_scale_submit.log"
STATE_FILE="markowitz_scale_submitted_sizes.txt"
SLEEP_SECONDS=1800
SIZES=(7 6 5 4 3 2 1 0)

touch "$LOG_FILE" "$STATE_FILE"

log_msg() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

already_submitted() {
  local size="$1"
  grep -qx "$size" "$STATE_FILE"
}

mark_submitted() {
  local size="$1"
  if ! already_submitted "$size"; then
    echo "$size" >> "$STATE_FILE"
  fi
}

log_msg "Starting Markowitz scale submission loop."
log_msg "Will try sizes: ${SIZES[*]}"

while true; do
  any_remaining=0

  for size in "${SIZES[@]}"; do
    if already_submitted "$size"; then
      continue
    fi

    any_remaining=1
    log_msg "Trying to submit size_indexer=${size}."

    output="$(sbatch "$RUNSCRIPT" "$size" 2>&1)"
    status=$?

    if [ "$status" -eq 0 ]; then
      mark_submitted "$size"
      log_msg "SUCCESS size_indexer=${size}: ${output}"
    else
      log_msg "FAILED size_indexer=${size}: ${output}"
      log_msg "Sleeping for ${SLEEP_SECONDS} seconds before retrying size_indexer=${size}."
      sleep "$SLEEP_SECONDS"
      break
    fi
  done

  if [ "$any_remaining" -eq 0 ]; then
    log_msg "All requested size arrays have been submitted. Exiting."
    exit 0
  fi
done
