# Unified Visuomotor Targets: Supervising VLAs Beyond Physical Actions

This repository contains the official implementation of **Unified Visuomotor Targets (UVT)** from the paper *"Unified Visuomotor Targets: Supervising VLAs Beyond Physical Actions."*
Our method requires **no architectural changes and no additional data**, and improves training efficiency and policy performance across simulation benchmarks and real-world bimanual manipulation tasks.

---

**Table of Contents**
- [Requirements](#requirements)
- [Installation](#installation)
- [Checkpoint Downloads](#checkpoint-downloads)
- [Quick Start: Inference](#quick-start-inference)
- [Full Training Pipeline](#full-training-pipeline)
  - [Step 0: Prepare RLDS Dataset](#step-0-prepare-rlds-dataset)
  - [Step 1: Precompute LAM Indices](#step-1-precompute-lam-indices)
  - [Step 2: Extract Fusion Data](#step-2-extract-fusion-data)
  - [Step 3: Train Fusion MVAE](#step-3-train-fusion-mvae)
  - [Step 4: Precompute u_t Vectors](#step-4-precompute-u_t-vectors)
  - [Step 5: Train VLA with u_t Head](#step-5-train-vla-with-u_t-head)
- [Evaluation](#evaluation)
- [End-to-End Pipeline Script](#end-to-end-pipeline-script)
- [Repository Structure](#repository-structure)

---

## Requirements

| | Minimum | Recommended |
|---|---------|-------------|
| **GPU (Training)** | 1x 24GB (RTX 4090) | 1x 80GB (A100/H100/B200) |
| **GPU (Inference)** | 1x 16GB | 1x 24GB |
| **GPU (MVAE only)** | CPU or any GPU | CPU is sufficient |
| **RAM** | 32 GB | 64 GB+ |
| **Disk** | 50 GB (code + checkpoints) | 200 GB+ (with datasets) |
| **Python** | 3.10 | 3.10 |
| **CUDA** | 12.1 | 12.1+ |

> **Note:** Steps 1-4 (LAM precompute through u_t precompute) are CPU/light-GPU tasks. Only Step 5 (VLA fine-tuning) requires a high-end GPU.

## Installation

```bash
# 1. Create conda environment
conda create -n uvt python=3.10 -y
conda activate uvt

# 2. Install PyTorch (pick ONE line matching your CUDA version)
# CUDA 12.1:
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
# CUDA 12.8+ (Blackwell GPUs):
# pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128

# 3. Clone and install (all dependencies resolved via pyproject.toml)
git clone https://github.com/YOUR_USERNAME/unified-visuomotor-targets.git
cd unified-visuomotor-targets
pip install -e .

# 4. (Optional) Install LIBERO for simulation evaluation
pip install robosuite==1.4.1 mujoco bddl easydict cloudpickle gym==0.26.2
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
export PYTHONPATH="$(pwd)/LIBERO:$PYTHONPATH"
```

**Verify installation:**
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')"
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"
python -c "from prismatic.models.load import load; print('prismatic OK')"
```

## Checkpoint Downloads

The following pre-trained checkpoints are required before training:

| Checkpoint | Description | Size | Source |
|-----------|-------------|------|--------|
| **Prismatic VLM Base** | Qwen2.5-0.5B + DINOSigLIP-224px backbone | ~3 GB | TODO |
| **UniVLA LAM** | Latent Action Model (Stage-2 VQ encoder) | ~200 MB | [UniVLA](https://huggingface.co/PLM2526/UniVLA) |

```bash
mkdir -p checkpoints/univla-latent-action-model

# Place downloaded files as:
# checkpoints/prism-qwen25-extra-dinosiglip-224px-0_5b/  (config.json, model.safetensors, metadata.pt, assets/)
# checkpoints/univla-latent-action-model/lam-stage-2.ckpt
```

## Quick Start: Inference

```python
import torch
from prismatic.models.load import load_vla
from scripts.fusion_mvae.model import ActionChunkDecoder
from scripts.fusion_mvae.configs import FusionMVAEConfig

# Load VLA model
checkpoint_path = "runs_lift_pot_w25/<run_name>--10000_chkpt"
vla = load_vla(checkpoint_path)
vla.eval()

# Load action decoder (frozen MVAE decoder)
dec_ckpt = torch.load(f"{checkpoint_path}/action_decoder--10000_checkpoint.pt", map_location="cpu")
dec_cfg = dec_ckpt.get("config", {})
mvae_cfg = FusionMVAEConfig(
    latent_dim=dec_cfg.get("latent_dim", 32),
    action_dim=dec_cfg.get("action_dim", 14),
    chunk_size=dec_cfg.get("chunk_size", 25),
)
decoder = ActionChunkDecoder(mvae_cfg).eval()
decoder.load_state_dict(dec_ckpt["action_decoder"])

# Predict u_t from image + language instruction
with torch.no_grad():
    u_pred = vla.predict_u(image, "pick up the pot")  # (1, 32)
    actions = decoder(u_pred)                           # (1, 25, 14)
```

## Full Training Pipeline

```
Raw RLDS Data -> LAM Precompute -> Fusion Data -> MVAE Training -> u_t Precompute -> VLA Fine-tuning
                 (Step 1)          (Step 2)       (Step 3)         (Step 4)          (Step 5)
```

### Step 0: Prepare RLDS Dataset

Your data must be in [RLDS format](https://github.com/google-research/rlds) (TFRecord-based). Each episode should contain:
- `observation/image`: RGB images `(H, W, 3)` uint8
- `action`: action vectors `(action_dim,)` float32
- `language_instruction`: task description string

**LIBERO datasets:** We use the modified LIBERO RLDS datasets from [OpenVLA](https://huggingface.co/datasets/openvla/modified_libero_rlds). Download with:
```bash
# Requires git-lfs: https://git-lfs.com
git clone git@hf.co:datasets/openvla/modified_libero_rlds

# Directory structure after download:
# modified_libero_rlds/
# ├── libero_spatial_no_noops/
# ├── libero_object_no_noops/
# ├── libero_goal_no_noops/
# └── libero_10_no_noops/

# Then set DATA_ROOT to point to this directory:
export DATA_ROOT=modified_libero_rlds
```

The LIBERO dataset configs (`libero_spatial_no_noops`, `libero_object_no_noops`, `libero_goal_no_noops`, `libero_10_no_noops`) are already registered in `prismatic/vla/datasets/rlds/oxe/configs.py`.

**Custom datasets:** Register your dataset in the OXE config:
1. Add an entry to `prismatic/vla/datasets/rlds/oxe/configs.py` with your dataset's image/state/action keys
2. Add a transform function to `prismatic/vla/datasets/rlds/oxe/transforms.py`
3. Add platform constants to `prismatic/vla/constants.py` (action dim, proprio dim, normalization type, chunk size)

---

### Step 1: Precompute LAM Indices

```bash
export DATASET=my_task
export DATA_ROOT=/data
export WINDOW_SIZE=25
export LAM_CKPT=checkpoints/univla-latent-action-model/lam-stage-2.ckpt
export OUTPUT_DIR=precomputed_lam_indices

CUDA_VISIBLE_DEVICES=0 python scripts/precompute_lam_indices.py \
    --dataset_name $DATASET \
    --dataset_path $DATA_ROOT/$DATASET \
    --lam_checkpoint $LAM_CKPT \
    --output_dir $OUTPUT_DIR \
    --window_size $WINDOW_SIZE \
    --num_shards 1 \
    --shard_idx 0

# Merge shards
python scripts/precompute_lam_indices.py \
    --dataset_name $DATASET \
    --output_dir $OUTPUT_DIR \
    --merge \
    --num_shards 1
```

> **Multi-dataset:** Run for each dataset. The `metadata.json` must list all datasets -- re-run merge or manually edit after all datasets are processed.

| Dataset size | Time estimate |
|-------------|---------------|
| ~50 episodes | ~2-5 min (1 GPU) |
| ~500 episodes | ~15-30 min (1 GPU) |
| ~5000 episodes | ~1-2 hours (1 GPU) |

**Outputs:** `precomputed_lam_indices/<dataset>/` containing `indices.npy`, `hash_to_idx.json`, `key_to_idx.json`, `keys.json`

---

### Step 2: Extract Fusion Data

```bash
python scripts/extract_fusion_data_generic.py \
    --dataset_name $DATASET \
    --dataset_path $DATA_ROOT/$DATASET \
    --lam_indices_dir $OUTPUT_DIR/$DATASET \
    --output_path $OUTPUT_DIR/$DATASET/fusion_data.npz \
    --chunk_size $WINDOW_SIZE \
    --normalize_actions
```

~1-5 min (CPU). Outputs `fusion_data.npz` with `action_chunks`, `lam_codes`, normalization stats.

---

### Step 3: Train Fusion MVAE

```bash
python scripts/fusion_mvae/train.py \
    --data_path $OUTPUT_DIR/$DATASET/fusion_data.npz \
    --output_dir $OUTPUT_DIR/$DATASET/mvae_checkpoints \
    --action_dim 14 \
    --chunk_size $WINDOW_SIZE \
    --latent_dim 32 \
    --epochs 500 \
    --batch_size 256 \
    --lr 1e-3 \
    --wandb_project "MVAE-$DATASET"
```

~5-15 min (CPU or 1 GPU). Outputs `mvae_checkpoints/best_model.pt`.

**Expected metrics at convergence:** action roundtrip L1 < 0.05, LAM code accuracy > 99%.

---

### Step 4: Precompute u_t Vectors

```bash
python scripts/compute_u_for_subset.py \
    --dataset_name $DATASET \
    --dataset_path $DATA_ROOT/$DATASET \
    --lam_indices_dir $OUTPUT_DIR \
    --mvae_checkpoint $OUTPUT_DIR/$DATASET/mvae_checkpoints/best_model.pt \
    --output_dir $OUTPUT_DIR/$DATASET \
    --chunk_size $WINDOW_SIZE
```

~1-5 min (CPU). Outputs `u_vectors.npy`, `decoder_weights.pt`, `u_metadata.json`.

---

### Step 5: Train VLA with u_t Head

**LIBERO example** (libero_spatial_no_noops, single GPU):
```bash
export DATASET=libero_spatial_no_noops
export DATA_ROOT=modified_libero_rlds
export OUTPUT_DIR=precomputed_lam_indices

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes 1 --nproc-per-node 1 \
    vla-scripts/finetune.py \
    --vlm_path checkpoints/prism-qwen25-extra-dinosiglip-224px-0_5b \
    --config_file_path pretrained_models/configs \
    --data_root_dir $DATA_ROOT \
    --dataset_name $DATASET \
    --run_root_dir runs_ut \
    --use_film False \
    --num_images_in_input 2 \
    --use_proprio True \
    --use_lora True \
    --lora_rank 64 \
    --use_fz False \
    --use_minivlm True \
    --image_aug True \
    --shuffle_buffer_size 52000 \
    --num_steps_before_decay 200000 \
    --max_steps 100000 \
    --save_freq 1000 \
    --save_latest_checkpoint_only False \
    --merge_lora_during_training True \
    --batch_size 8 \
    --grad_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --use_pro_version True \
    --use_l1_regression False \
    --use_ut_head True \
    --ut_latent_dim 32 \
    --ut_loss_type l1 \
    --ut_loss_weight 1.0 \
    --ut_decode_loss_weight 1.0 \
    --ut_decoder_weights_path $OUTPUT_DIR/$DATASET/decoder_weights.pt \
    --lam_indices_path $OUTPUT_DIR \
    --wandb_project "VLA-UT" \
    --wandb_entity YOUR_WANDB_ENTITY
```

| Steps | GPU | Time estimate |
|-------|-----|---------------|
| 10,000 | 1x A100 (80GB) | ~4-5 hours |
| 10,000 | 1x B200 (96GB) | ~5 hours |
| 10,000 | 1x RTX 4090 (24GB) | ~8-10 hours (reduce batch_size to 2) |
| 100,000 | 1x A100 (80GB) | ~40-50 hours |

**Multi-task parallel training** (one dataset per GPU):
```bash
for i in 0 1 2; do
    DATASETS=(libero_object_no_noops libero_goal_no_noops libero_10_no_noops)
    CUDA_VISIBLE_DEVICES=$i torchrun --standalone --nnodes 1 --nproc-per-node 1 \
        --master_port $((29500 + i)) \
        vla-scripts/finetune.py \
        ... \
        --dataset_name ${DATASETS[$i]} &
    sleep 30  # stagger launches
done
wait
```

**Outputs:** `runs_ut/<run_name>--<step>_chkpt/` containing `lora_adapter/`, `action_head--*_checkpoint.pt`, `action_decoder--*_checkpoint.pt`, `proprio_projector--*_checkpoint.pt`, `dataset_statistics.json`.

---

## Evaluation

### LIBERO Benchmark

Requires LIBERO installed (see Installation step 4).

```bash
export PYTHONPATH="$(pwd)/LIBERO:$PYTHONPATH"
export MUJOCO_GL=egl  # headless rendering

CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint runs_ut/<run_name>--10000_chkpt \
    --task_suite_name libero_spatial \
    --model_family openvla \
    --use_l1_regression False \
    --use_ut_head True \
    --use_minivlm True \
    --use_film False \
    --num_images_in_input 2 \
    --use_proprio True \
    --center_crop True \
    --num_open_loop_steps 8 \
    --use_pro_version True \
    --save_version vla-adapter \
    --num_trials_per_task 20 \
    --local_log_dir experiments/logs
```

Available `--task_suite_name` options: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`.

---

## End-to-End Pipeline Script

Run all steps sequentially with a single command:

```bash
bash scripts/run_full_pipeline.sh <dataset_name> <data_root> <gpu_id> [window_size] [action_dim] [max_steps]

# Example:
bash scripts/run_full_pipeline.sh lift_pot /data 0 25 14 10000
```

### Time Budget (single dataset, 1 GPU)

| Step | What | Time |
|------|------|------|
| Step 1 | LAM precompute | 5-30 min |
| Step 2 | Fusion data extraction | 1-5 min |
| Step 3 | MVAE training (500 epochs) | 5-15 min |
| Step 4 | u_t precomputation | 1-5 min |
| Step 5 | VLA training (10k steps) | 4-10 hours |
| **Total** | | **~5-11 hours** |

Steps 1-4 together take under 1 hour. Step 5 dominates wall-clock time.

---

## Repository Structure

```
unified-visuomotor-targets/
├── README.md
├── pyproject.toml                          # All dependencies (pip install -e .)
│
├── scripts/
│   ├── precompute_lam_indices.py           # Step 1: LAM discrete code precomputation
│   ├── extract_fusion_data_generic.py      # Step 2: Extract (action, LAM code) pairs
│   ├── compute_u_for_subset.py             # Step 4: Precompute u_t vectors
│   ├── run_full_pipeline.sh                # End-to-end pipeline script
│   └── fusion_mvae/
│       ├── configs.py                      # FusionMVAEConfig dataclass
│       ├── model.py                        # PrecisionAwarePoEMVAE architecture
│       ├── dataset.py                      # Fusion data loading
│       ├── train.py                        # Step 3: MVAE training loop
│       └── precompute_u.py                 # Alternative u_t precomputation
│
├── vla-scripts/
│   ├── finetune.py                         # Step 5: Main VLA fine-tuning script
│   ├── merge_lora_weights_and_save.py      # Merge LoRA weights into base model
│   └── vla_evaluation.py                   # VLA evaluation utilities
│
├── prismatic/                              # Model architecture (from VLA-Adapter)
│   ├── vla/
│   │   ├── constants.py                    # Platform constants (action dim, chunk size)
│   │   ├── action_tokenizer.py             # Action tokenization
│   │   └── datasets/
│   │       ├── datasets_lam_precomputed.py # RLDS dataset with u_t hash lookup
│   │       └── rlds/                       # RLDS data loading pipeline
│   │           └── oxe/                    # Open X-Embodiment dataset configs
│   ├── models/
│   │   ├── action_heads.py                 # L1RegressionActionHead, LAMCodeHead
│   │   ├── load.py                         # Model loading utilities
│   │   ├── backbones/                      # Vision (DINOSigLIP) + LLM (Qwen2.5) backbones
│   │   └── vlms/                           # Prismatic VLM implementation
│   ├── training/                           # Training utilities and strategies
│   └── util/                               # Data collation, batching utilities
│
├── latent_action_model/                    # UniVLA Latent Action Model
│   ├── main.py
│   └── genie/                              # LAM architecture (VQ encoder/decoder)
│
├── experiments/
│   └── robot/
│       └── libero/                         # LIBERO benchmark evaluation
│
├── pretrained_models/
│   └── configs/                            # HuggingFace tokenizer/processor configs
│
└── checkpoints/                            # (git-ignored) Downloaded model checkpoints
```

