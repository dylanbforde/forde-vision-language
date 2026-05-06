#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:."

COMMON_ARGS=(
  --dataset_name wikitext
  --dataset_config wikitext-103-raw-v1
  --train_split train
  --eval_split validation
  --eval_interval 100
  --eval_batches 2
  --num_epochs 1
  --max_steps 500
  --batch_size 8
  --d_model 256
  --num_layers 4
  --num_heads 4
  --num_experts 4
  --max_seq_len 512
  --log_interval 25
  --slow_loop_interval 100
)

run_experiment() {
  local name="$1"
  shift
  python src/training/train.py \
    "${COMMON_ARGS[@]}" \
    --experiment_name "${name}" \
    --checkpoint_dir "checkpoints/${name}" \
    "$@"
}

run_experiment dense_baseline --no_moe --no_nsa --no_mhc --slow_loop_interval 0
run_experiment moe_baseline --no_nsa --no_mhc --slow_loop_interval 0
run_experiment moe_gmm_slow_loop --no_nsa --no_mhc --slow_loop_assignment_method gmm
run_experiment moe_balanced_ot --no_nsa --no_mhc --no_ot_dustbin --ot_tau_q 10.0 --ot_tau_k 10.0
run_experiment moe_unbalanced_ot --no_nsa --no_mhc
run_experiment full_forde_ot

