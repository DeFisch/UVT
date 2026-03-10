#!/bin/bash
# Full u_t-VLA pipeline: LAM precompute -> Fusion data -> MVAE -> u_t -> VLA training
#
# Usage:
#   bash scripts/run_full_pipeline.sh <dataset_name> <data_root> <gpu_id> [window_size] [action_dim] [max_steps]
#
# Example:
#   bash scripts/run_full_pipeline.sh lift_pot /data 0 25 14 10000

set -e

DATASET=${1:?Usage: $0 <dataset_name> <data_root> <gpu_id> [window_size] [action_dim] [max_steps]}
DATA_ROOT=${2:?}
GPU=${3:?}
WINDOW_SIZE=${4:-25}
ACTION_DIM=${5:-14}
MAX_STEPS=${6:-10000}

PROJECT=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT"

export PYTHONPATH="${PROJECT}:${PYTHONPATH}"
export TF_FORCE_GPU_ALLOW_GROWTH=true

LAM_CKPT="${PROJECT}/checkpoints/univla-latent-action-model/lam-stage-2.ckpt"
OUTPUT_DIR="${PROJECT}/precomputed_lam_indices"
LR=1e-4

echo "============================================"
echo "u_t-VLA Full Pipeline"
echo "  Dataset:     $DATASET"
echo "  Data root:   $DATA_ROOT"
echo "  GPU:         $GPU"
echo "  Window size: $WINDOW_SIZE"
echo "  Action dim:  $ACTION_DIM"
echo "  Max steps:   $MAX_STEPS"
echo "============================================"

# Step 1: Precompute LAM indices
echo ""
echo "=== Step 1/5: Precompute LAM indices ==="
CUDA_VISIBLE_DEVICES=$GPU python scripts/precompute_lam_indices.py \
    --dataset_name $DATASET \
    --dataset_path $DATA_ROOT/$DATASET \
    --lam_checkpoint $LAM_CKPT \
    --output_dir $OUTPUT_DIR \
    --window_size $WINDOW_SIZE \
    --num_shards 1 --shard_idx 0

python scripts/precompute_lam_indices.py \
    --dataset_name $DATASET \
    --output_dir $OUTPUT_DIR \
    --merge --num_shards 1

echo "  LAM indices: $(wc -l < $OUTPUT_DIR/$DATASET/keys.json) windows"

# Step 2: Extract fusion data
echo ""
echo "=== Step 2/5: Extract fusion data ==="
python scripts/extract_fusion_data_generic.py \
    --dataset_name $DATASET \
    --dataset_path $DATA_ROOT/$DATASET \
    --lam_indices_dir $OUTPUT_DIR/$DATASET \
    --output_path $OUTPUT_DIR/$DATASET/fusion_data.npz \
    --chunk_size $WINDOW_SIZE \
    --normalize_actions

# Step 3: Train MVAE
echo ""
echo "=== Step 3/5: Train Fusion MVAE ==="
python scripts/fusion_mvae/train.py \
    --data_path $OUTPUT_DIR/$DATASET/fusion_data.npz \
    --output_dir $OUTPUT_DIR/$DATASET/mvae_checkpoints \
    --action_dim $ACTION_DIM \
    --chunk_size $WINDOW_SIZE \
    --latent_dim 32 \
    --epochs 500 \
    --batch_size 256 \
    --lr 1e-3

# Step 4: Precompute u_t vectors
echo ""
echo "=== Step 4/5: Precompute u_t vectors ==="
python scripts/compute_u_for_subset.py \
    --dataset_name $DATASET \
    --dataset_path $DATA_ROOT/$DATASET \
    --lam_indices_dir $OUTPUT_DIR \
    --mvae_checkpoint $OUTPUT_DIR/$DATASET/mvae_checkpoints/best_model.pt \
    --output_dir $OUTPUT_DIR/$DATASET \
    --chunk_size $WINDOW_SIZE

# Step 5: Train VLA
echo ""
echo "=== Step 5/5: Train VLA with u_t head ==="
NOTE="ut-decoder-w${WINDOW_SIZE}-lr${LR}--${DATASET}--$(date +%Y%m%d_%H%M%S)"

CUDA_VISIBLE_DEVICES=$GPU torchrun --standalone --nnodes 1 --nproc-per-node 1 \
    vla-scripts/finetune.py \
    --vlm_path "${PROJECT}/checkpoints/prism-qwen25-extra-dinosiglip-224px-0_5b" \
    --config_file_path pretrained_models/configs \
    --data_root_dir "$DATA_ROOT" \
    --dataset_name "$DATASET" \
    --run_root_dir "./runs_${DATASET}" \
    --use_film False \
    --num_images_in_input 3 \
    --use_proprio True \
    --use_lora True \
    --use_fz False \
    --use_minivlm True \
    --image_aug True \
    --shuffle_buffer_size 5000 \
    --num_steps_before_decay 200000 \
    --max_steps $MAX_STEPS \
    --save_freq 1000 \
    --save_latest_checkpoint_only False \
    --merge_lora_during_training True \
    --batch_size 4 \
    --grad_accumulation_steps 4 \
    --learning_rate "$LR" \
    --lora_rank 64 \
    --use_pro_version True \
    --use_l1_regression False \
    --use_ut_head True \
    --ut_latent_dim 32 \
    --ut_loss_type l1 \
    --ut_loss_weight 1.0 \
    --ut_decode_loss_weight 1.0 \
    --ut_decoder_weights_path "${OUTPUT_DIR}/${DATASET}/decoder_weights.pt" \
    --lam_indices_path "$OUTPUT_DIR" \
    --run_id_note "$NOTE"

echo ""
echo "=== Pipeline complete! ==="
echo "Checkpoints saved to: runs_${DATASET}/"
