#!/usr/bin/env bash

# Run multiple train.py commands, repeating each until it succeeds a target number of times.
# After each run, move checkpoint artifacts and logs into checkpoint/n=<N>/RF=<value>/...,
# and failures into checkpoint/n=<N>/RF=<value>/Failed/.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/nohup_rf_runs"
mkdir -p "$LOG_DIR"

CHECKPOINT_DIR="$SCRIPT_DIR/checkpoint"
mkdir -p "$CHECKPOINT_DIR"

SUCCESS_TARGET=10

# "python3 train.py --N 50 --Q 251 --sigma 3 --hamming 3 --epoch_size 100000 --max_epoch 75 --env_base_seed 24 --batch_size 128"
#COMMANDS=(
#  "CUDA_VISIBLE_DEVICES=0 python3 train.py --N 50 --Q 251 --sigma 3 --hamming 3 --epoch_size 100000 --max_epoch 75 --env_base_seed 24 --batch_size 128"
#  "CUDA_VISIBLE_DEVICES=1 python3 train.py --N 50 --Q 251 --sigma 3 --hamming 3 --epoch_size 100000 --max_epoch 75 --env_base_seed 25 --batch_size 128"
#)


COMMANDS=(
#  "python3 train.py --N 40 --Q 251 --sigma 3 --hamming 3 --a_reduced_source 'BKZ_sample_generater/RF=0.541' --epoch_size 75000 --max_epoch 75 --env_base_seed 24 --batch_size 128"
#  "python3 train.py --N 40 --Q 251 --sigma 3 --hamming 3 --a_reduced_source 'BKZ_sample_generater/RF=0.686' --epoch_size 75000 --max_epoch 75 --env_base_seed 24 --batch_size 128"
#  "python3 train.py --N 40 --Q 251 --sigma 3 --hamming 3 --a_reduced_source 'BKZ_sample_generater/RF=0.864' --epoch_size 75000 --max_epoch 75 --env_base_seed 24 --batch_size 128"
  "python3 train.py --N 40 --Q 251 --sigma 3 --hamming 3 --epoch_size 75000 --max_epoch 75 --env_base_seed 24 --batch_size 128"
)
extract_rf_label() {
  local cmd="$1"
  if [[ "$cmd" =~ (^|[[:space:]])--rf[[:space:]]+([^[:space:]]+) ]]; then
    echo "${BASH_REMATCH[2]}"
  else
    # No --rf means the standard uniform experiment; treat it as RF=1.0 for folder naming.
    echo "1.0"
  fi
}

extract_n_label() {
  local cmd="$1"
  if [[ "$cmd" =~ (^|[[:space:]])--N[[:space:]]+([^[:space:]]+) ]]; then
    echo "${BASH_REMATCH[2]}"
  else
    echo "unknown"
  fi
}

build_cmd_with_dump_path() {
  local cmd="$1"
  local dump_path="$2"

  if [[ "$cmd" =~ (^|[[:space:]])--dump_path[[:space:]]+ ]]; then
    echo "$cmd"
  else
    echo "$cmd --dump_path \"$dump_path\""
  fi
}

overall_idx=0
for cmd in "${COMMANDS[@]}"; do
  rf_label="$(extract_rf_label "$cmd")"
  n_label="$(extract_n_label "$cmd")"
  success_count=0

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting command: ${cmd}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Target successes: ${SUCCESS_TARGET} (N=${n_label}, RF=${rf_label})"

  while (( success_count < SUCCESS_TARGET )); do
    overall_idx=$((overall_idx + 1))
    attempt=$((success_count + 1))
    ts="$(date '+%Y%m%d_%H%M%S')"
    log_file="${LOG_DIR}/RF=${rf_label}_overall_${overall_idx}_success_target_${attempt}_${ts}.log"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Run attempt (RF=${rf_label}) ${attempt}/${SUCCESS_TARGET}; logging to ${log_file}"

    rf_dir="${CHECKPOINT_DIR}/n=${n_label}/RF=${rf_label}"
    mkdir -p "$rf_dir"

    tmp_dir="${rf_dir}/_tmp_overall_${overall_idx}_${ts}"
    mkdir -p "$tmp_dir"
    cmd_run="$(build_cmd_with_dump_path "$cmd" "$tmp_dir")"

    nohup bash -lc "$cmd_run" >"${log_file}" 2>&1 &
    pid=$!
    wait "${pid}"
    exit_code=$?

    if [[ ${exit_code} -eq 0 ]]; then
      success_count=$((success_count + 1))
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] Success ${success_count}/${SUCCESS_TARGET} (RF=${rf_label})."
      run_dir="${rf_dir}/run_${success_count}_${ts}"
      mv "$tmp_dir" "$run_dir"
      {
        echo "timestamp=${ts}"
        echo "rf=${rf_label}"
        echo "result=success"
        echo "success_index=${success_count}"
        echo "overall_index=${overall_idx}"
        echo "exit_code=${exit_code}"
        echo "command=${cmd}"
      } >"${run_dir}/status.txt"
      if [[ -f "$log_file" ]]; then
        mv "$log_file" "$run_dir/"
      fi
    else
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] Failed (RF=${rf_label}) exit=${exit_code}; retrying."
      fail_dir="${rf_dir}/Failed/attempt_${overall_idx}_${ts}"
      mkdir -p "$fail_dir"
      mv "$tmp_dir" "$fail_dir"
      {
        echo "timestamp=${ts}"
        echo "rf=${rf_label}"
        echo "result=failure"
        echo "overall_index=${overall_idx}"
        echo "exit_code=${exit_code}"
        echo "command=${cmd}"
      } >"${fail_dir}/status.txt"
      if [[ -f "$log_file" ]]; then
        mv "$log_file" "$fail_dir/"
      fi
    fi
  done

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed RF=${rf_label} with ${SUCCESS_TARGET} successes."
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All commands completed."
