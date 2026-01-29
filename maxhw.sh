#!/usr/bin/env bash
set -u

N=30
Q=251
SIGMA=3

MAX_H=15
MAX_CONSEC_FAIL=2
DUMP_BASE="checkpoint/sweeps"
RESULTS_FILE="results_max_hw.txt"

mkdir -p "$DUMP_BASE"

run_condition() {
  local cond_name="$1"
  local extra_args="$2"
  local start_h="$3"

  local h="$start_h"
  local max_ok=0

  echo "=== Condition: ${cond_name} ==="

  while true; do
    local consec_fail=0
    local t=1

    while true; do
      local ts
      ts=$(date +%Y%m%d_%H%M%S)
      local exp_name="hw_sweep_${cond_name}"
      local exp_id="h${h}_t${t}_${ts}_$$"

      local log_file="log_${cond_name}_h${h}_t${t}_${ts}.log"

      echo "[${cond_name}] hamming=${h} trial=${t} exp_id=${exp_id}"

      bash -lc "
        python3 train.py \
          --N \"$N\" --Q \"$Q\" --sigma \"$SIGMA\" --hamming \"$h\" \
          --dump_path \"$DUMP_BASE\" \
          --exp_name \"$exp_name\" --exp_id \"$exp_id\" \
          $extra_args
      " > "$log_file" 2>&1

      exit_code=$?

      if [[ $exit_code -eq 0 ]]; then
        echo "  -> success (exit_code=0)"
        max_ok=$h

        if [[ $h -ge $MAX_H ]]; then
          echo "[${cond_name}] reached MAX_H=${MAX_H}; stop."
          echo "${cond_name}\t${max_ok}" >> "$RESULTS_FILE"
          return 0
        fi

        h=$((h + 1))
        break
      else
        consec_fail=$((consec_fail + 1))
        echo "  -> failure (exit_code=${exit_code}) (${consec_fail}/${MAX_CONSEC_FAIL})"
      fi

      if [[ $consec_fail -ge $MAX_CONSEC_FAIL ]]; then
        echo "[${cond_name}] stop at hamming=${h}; max recoverable=${max_ok}"
        echo "${cond_name}\t${max_ok}" >> "$RESULTS_FILE"
        return 0
      fi

      t=$((t + 1))
    done
  done
}

: > "$RESULTS_FILE"

#run_condition "rf_0541" "--a_reduced_source BKZ_sample_generater/RF=0.541" 3

##run_condition "rf_0686" "--a_reduced_source BKZ_sample_generater/RF=0.686" 3
run_condition "rf_0864" "--a_reduced_source BKZ_sample_generater/RF=0.864" 7
