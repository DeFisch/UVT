"""
Precompute LAM (UniVLA Latent Action Model) discrete indices for RLDS datasets.

Uses TF data pipeline for parallel I/O and Python multiprocessing for image transforms.

Supports sharding:
    CUDA_VISIBLE_DEVICES=0 python script.py --shard_idx 0 --num_shards 5
    ...
Then merge:
    python script.py --merge --output_dir precomputed_lam_indices --datasets calvin_abc --num_shards 5
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as tv_transforms
from tqdm import tqdm

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, REPO_ROOT)


HASH_RESIZE = (224, 224)  # Must match RLDS pipeline resize_resolution


def compute_image_hash(img: np.ndarray) -> str:
    """MD5 hash of full image bytes. Must match datasets_lam_precomputed.py."""
    return hashlib.md5(np.ascontiguousarray(img).tobytes()).hexdigest()


def make_hash_key(img_first: np.ndarray, img_last: np.ndarray) -> str:
    """Compute hash key from already-resized image pair."""
    return f"{compute_image_hash(img_first)}_{compute_image_hash(img_last)}"


def tf_batch_resize_episode(raw_images: list) -> list:
    """Resize all images in an episode using TF Lanczos3 (same as RLDS pipeline)."""
    import tensorflow as tf
    if not raw_images:
        return []
    h, w = raw_images[0].shape[:2]
    if (h, w) == HASH_RESIZE:
        return raw_images
    batch = tf.constant(np.stack(raw_images), dtype=tf.uint8)
    batch = tf.image.resize(batch, HASH_RESIZE, method="lanczos3", antialias=True)
    batch = tf.cast(tf.clip_by_value(tf.round(batch), 0, 255), tf.uint8)
    return [batch[i].numpy() for i in range(len(raw_images))]


def load_lam_encoder(checkpoint_path: str, device: str = "cuda"):
    from latent_action_model.genie.modules.lam import ControllableDINOLatentActionModel

    print(f"Loading LAM encoder from {checkpoint_path}")
    model = ControllableDINOLatentActionModel(
        in_dim=3, model_dim=768, latent_dim=128, num_latents=16,
        patch_size=14, enc_blocks=12, dec_blocks=12, num_heads=12, dropout=0.0,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = {k.replace("lam.", ""): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    model.requires_grad_(False)
    print("LAM encoder loaded successfully")
    return model


LAM_TO_TENSOR = tv_transforms.ToTensor()


def process_single_window(args):
    """Process a single window: convert TF-resized images to tensors and compute hash. Runs in thread pool."""
    img_first_resized, img_last_resized, dataset_name, global_ep_idx, step_start = args
    # Use TF-resized images for BOTH LAM encoding and hashing
    # This matches the RLDS pipeline which feeds TF Lanczos3 resized images at training time
    first_tensor = LAM_TO_TENSOR(Image.fromarray(img_first_resized))
    last_tensor = LAM_TO_TENSOR(Image.fromarray(img_last_resized))
    key = f"{dataset_name}_{global_ep_idx}_{step_start}"
    hash_key = make_hash_key(img_first_resized, img_last_resized)
    return first_tensor, last_tensor, key, hash_key


def batch_compute_lam_indices(model, batch_first, batch_last, device):
    first = torch.stack(batch_first, dim=0)
    last = torch.stack(batch_last, dim=0)
    video = torch.stack([first, last], dim=1).to(device)
    with torch.no_grad():
        result = model.vq_encode(video)
        indices = result['indices']
    return indices.reshape(-1, 4).cpu().numpy()


def process_rlds_dataset(
    data_root: Path,
    dataset_name: str,
    model,
    device: str,
    window_size: int = 8,
    batch_size: int = 32,
    shard_idx: int = 0,
    num_shards: int = 1,
    num_workers: int = 16,
) -> Tuple[np.ndarray, List[str], List[str]]:
    import tensorflow_datasets as tfds

    dataset_path = data_root / dataset_name
    version_dirs = [d for d in dataset_path.iterdir() if d.is_dir() and d.name[0].isdigit()]
    if version_dirs:
        dataset_path = version_dirs[0]

    builder = tfds.builder_from_directory(str(dataset_path))

    pct_start = (shard_idx * 100) // num_shards
    pct_end = ((shard_idx + 1) * 100) // num_shards
    split_str = f"train[{pct_start}%:{pct_end}%]"
    print(f"Shard {shard_idx}/{num_shards}: split '{split_str}', {num_workers} CPU workers", flush=True)

    # Use TF data pipeline with parallel reads and prefetching
    read_config = tfds.ReadConfig(
        interleave_cycle_length=16,
        interleave_block_length=1,
        num_parallel_calls_for_decode=num_workers,
        num_parallel_calls_for_interleave_files=num_workers,
    )
    ds = builder.as_dataset(split=split_str, read_config=read_config)
    # Prefetch episodes so CPU I/O overlaps with GPU compute
    ds = ds.prefetch(buffer_size=8)

    all_indices = []
    all_keys = []
    all_hash_keys = []

    batch_first = []
    batch_last = []
    batch_keys = []
    batch_hash_keys = []

    total_windows = 0
    episodes_processed = 0

    executor = ThreadPoolExecutor(max_workers=num_workers)

    for local_idx, episode in enumerate(tqdm(ds, desc=f"Shard {shard_idx}")):
        steps = list(episode['steps'])
        num_steps = len(steps)

        if num_steps < window_size:
            continue

        global_ep_idx = f"s{shard_idx}_{local_idx}"

        # Extract images - find the right key
        raw_images = []
        for step in steps:
            obs = step['observation']
            if 'rgb_static' in obs:
                img = obs['rgb_static'].numpy()
            elif 'image_primary' in obs:
                img = obs['image_primary'].numpy()
            elif 'image' in obs:
                img = obs['image'].numpy()
            else:
                raise KeyError(f"No image key found. Available: {list(obs.keys())}")
            raw_images.append(img)

        # Resize all images using TF Lanczos3 (same as RLDS pipeline) for hash computation
        resized_images = tf_batch_resize_episode(raw_images)

        # Build window tasks for this episode (use TF-resized images for both LAM and hash)
        window_tasks = []
        for step_start in range(0, num_steps - window_size + 1):
            window_tasks.append((
                resized_images[step_start],
                resized_images[step_start + window_size - 1],
                dataset_name, global_ep_idx, step_start,
            ))

        # Process windows in parallel threads (resize + hash)
        results = list(executor.map(process_single_window, window_tasks))

        for first_t, last_t, key, hash_key in results:
            batch_first.append(first_t)
            batch_last.append(last_t)
            batch_keys.append(key)
            batch_hash_keys.append(hash_key)

            if len(batch_first) >= batch_size:
                indices = batch_compute_lam_indices(model, batch_first, batch_last, device)
                all_indices.append(indices)
                all_keys.extend(batch_keys)
                all_hash_keys.extend(batch_hash_keys)
                total_windows += len(batch_first)
                batch_first = []
                batch_last = []
                batch_keys = []
                batch_hash_keys = []

        episodes_processed += 1
        if episodes_processed % 200 == 0:
            print(f"  Shard {shard_idx}: {episodes_processed} eps, {total_windows} windows", flush=True)

    executor.shutdown(wait=False)

    if batch_first:
        indices = batch_compute_lam_indices(model, batch_first, batch_last, device)
        all_indices.append(indices)
        all_keys.extend(batch_keys)
        all_hash_keys.extend(batch_hash_keys)
        total_windows += len(batch_first)

    if all_indices:
        all_indices = np.concatenate(all_indices, axis=0)
    else:
        all_indices = np.zeros((0, 4), dtype=np.int16)

    print(f"  Shard {shard_idx}: Done. {episodes_processed} eps, {total_windows} windows, shape: {all_indices.shape}")
    return all_indices, all_keys, all_hash_keys


def merge_shards(output_dir: Path, dataset_name: str, num_shards: int):
    print(f"Merging {num_shards} shards for {dataset_name}...")

    all_indices = []
    all_keys = []
    all_hash_keys = []

    for shard_idx in range(num_shards):
        shard_dir = output_dir / f"{dataset_name}_shard{shard_idx}"
        if not shard_dir.exists():
            print(f"  WARNING: Shard {shard_idx} not found at {shard_dir}")
            continue

        indices = np.load(shard_dir / "indices.npy")
        with open(shard_dir / "keys.json") as f:
            keys = json.load(f)
        with open(shard_dir / "hash_keys.json") as f:
            hash_keys = json.load(f)

        all_indices.append(indices)
        all_keys.extend(keys)
        all_hash_keys.extend(hash_keys)
        print(f"  Shard {shard_idx}: {len(keys)} samples")

    all_indices = np.concatenate(all_indices, axis=0)

    dataset_dir = output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    np.save(dataset_dir / "indices.npy", all_indices.astype(np.int16))
    with open(dataset_dir / "keys.json", "w") as f:
        json.dump(all_keys, f)

    key_to_idx = {k: i for i, k in enumerate(all_keys)}
    with open(dataset_dir / "key_to_idx.json", "w") as f:
        json.dump(key_to_idx, f)

    hash_to_idx = {h: i for i, h in enumerate(all_hash_keys)}
    with open(dataset_dir / "hash_to_idx.json", "w") as f:
        json.dump(hash_to_idx, f)

    metadata = {
        "num_codes": 4,
        "num_categories": 16,
        "datasets": {
            dataset_name: {
                "num_samples": len(all_keys),
                "indices_shape": list(all_indices.shape),
                "num_unique_hashes": len(set(all_hash_keys)),
            }
        }
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Merged: {len(all_keys)} total samples, shape {all_indices.shape}")
    print(f"Saved to {dataset_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--lam_checkpoint", type=str, default="")
    parser.add_argument("--datasets", type=str, nargs="+", required=True)
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.merge:
        for dataset_name in args.datasets:
            merge_shards(output_dir, dataset_name, args.num_shards)
        return

    data_root = Path(args.data_root)
    model = load_lam_encoder(args.lam_checkpoint, args.device)

    for dataset_name in args.datasets:
        shard_dir = output_dir / f"{dataset_name}_shard{args.shard_idx}"
        shard_dir.mkdir(parents=True, exist_ok=True)

        indices, keys, hash_keys = process_rlds_dataset(
            data_root, dataset_name, model, args.device,
            args.window_size, args.batch_size,
            args.shard_idx, args.num_shards,
            args.num_workers,
        )

        np.save(shard_dir / "indices.npy", indices.astype(np.int16))
        with open(shard_dir / "keys.json", "w") as f:
            json.dump(keys, f)
        with open(shard_dir / "hash_keys.json", "w") as f:
            json.dump(hash_keys, f)

        print(f"Shard {args.shard_idx} saved to {shard_dir}")

    print(f"Shard {args.shard_idx} complete!")


if __name__ == "__main__":
    main()
