#!/usr/bin/env python3
"""Calculate metrics sequentially, sharding every metric across all GPUs."""

import argparse
import gc
import importlib
import json
import math
import os
import shutil
import sys
import types
import urllib.request
import zipfile
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from scipy import linalg
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from utils.dino_utils import (
    group_indices_by_prompt,
    knn_squared_radii,
    manifold_coverage,
    mean_pairwise_cosine_distance,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
HPS_URL = "https://huggingface.co/spaces/xswu/HPSv2/resolve/main/HPS_v2_compressed.pt"
AESTHETIC_URL = (
    "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/"
    "sac%2Blogos%2Bava1-l14-linearMSE.pth"
)
DINOV2_ARCHIVE_URL = "https://github.com/facebookresearch/dinov2/archive/refs/heads/main.zip"


class ImagePromptDataset(Dataset):
    def __init__(self, image_dir, prompt_dict, image_files):
        self.image_dir = Path(image_dir)
        self.prompt_dict = prompt_dict
        self.image_files = image_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        filename = self.image_files[index]
        with Image.open(self.image_dir / filename) as image:
            image = image.convert("RGB").copy()
        return filename, image, self.prompt_dict[filename]


class ImageOnlyDataset(Dataset):
    def __init__(self, image_dir, image_files, transform):
        self.image_dir = Path(image_dir)
        self.image_files = image_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        with Image.open(self.image_dir / self.image_files[index]) as image:
            return self.transform(image.convert("RGB"))


def collate_image_prompts(batch):
    filenames, images, prompts = zip(*batch)
    return list(filenames), list(images), list(prompts)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher_samples_dir",
        required=True,
        help="Directory containing teacher/reference images.",
    )
    parser.add_argument(
        "--student_samples_dir",
        required=True,
        help="Directory containing generated student images.",
    )
    parser.add_argument(
        "--prompt_json_path",
        default=None,
        help="Image-to-prompt JSON; defaults to <student_samples_dir>.json.",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--fid_batch_size", type=int, default=32)
    parser.add_argument(
        "--dinov2_recall_k", type=int, default=3,
        help="Student k-NN manifold size for DINOv2 Recall-to-Teacher (excluding self).",
    )
    parser.add_argument(
        "--dinov2_distance_chunk_size", type=int, default=1024,
        help="Row/column block size for DINOv2 recall distance calculations.",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--aesthetic_checkpoint",
        required=True,
    )
    parser.add_argument(
        "--clip_cache_dir",
        required=True,
    )
    parser.add_argument(
        "--image_reward_dir",
        required=True,
    )
    parser.add_argument(
        "--dinov2_repo",
        required=True,
    )
    args = parser.parse_args()
    if args.dinov2_recall_k < 1:
        parser.error("--dinov2_recall_k must be positive")
    if args.dinov2_distance_chunk_size < 1:
        parser.error("--dinov2_distance_chunk_size must be positive")
    return args


def init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl", timeout=timedelta(hours=6))
    torch.cuda.set_device(local_rank)
    return rank, world_size, torch.device("cuda", local_rank)


def download_file(url, destination):
    destination = Path(destination)
    if destination.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".download")
    print(f"Downloading {url} -> {destination}")
    urllib.request.urlretrieve(url, temporary_path)
    temporary_path.replace(destination)


def download_dinov2_repository(repository_path):
    repository_path = Path(repository_path)
    if (repository_path / "hubconf.py").is_file():
        return
    repository_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path = repository_path.parent / "dinov2-main.zip"
    extracted_path = repository_path.parent / "dinov2-main"
    download_file(DINOV2_ARCHIVE_URL, archive_path)
    if extracted_path.exists():
        shutil.rmtree(extracted_path)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(repository_path.parent)
    if repository_path.exists():
        shutil.rmtree(repository_path)
    extracted_path.rename(repository_path)
    archive_path.unlink()


def import_image_reward():
    """Import ImageReward without its unused ReFL training dependencies."""
    refl_module_name = "ImageReward.ReFL"
    if refl_module_name not in sys.modules:
        refl_stub = types.ModuleType(refl_module_name)
        refl_stub.__all__ = []
        sys.modules[refl_module_name] = refl_stub
    return importlib.import_module("ImageReward")


def prepare_models(args, rank, world_size):
    """Download missing model files once on rank 0, then let all ranks load locally."""
    if rank == 0:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import LocalEntryNotFoundError

        huggingface_cache = os.environ.get("HF_HUB_CACHE")
        for repository_id in (
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            "yuvalkirstain/PickScore_v1",
            "openai/clip-vit-base-patch32",
            "bert-base-uncased",
        ):
            print(f"Checking Hugging Face model: {repository_id}")
            try:
                snapshot_download(
                    repo_id=repository_id,
                    cache_dir=huggingface_cache,
                    local_files_only=True,
                )
            except LocalEntryNotFoundError:
                print(f"Downloading Hugging Face model: {repository_id}")
                snapshot_download(repo_id=repository_id, cache_dir=huggingface_cache)

        import clip

        print("Checking OpenAI CLIP ViT-L/14")
        clip_model, _ = clip.load("ViT-L/14", device="cpu", download_root=args.clip_cache_dir)
        del clip_model

        RM = import_image_reward()

        image_reward_dir = Path(args.image_reward_dir)
        if not (image_reward_dir / "ImageReward.pt").is_file() or not (
            image_reward_dir / "med_config.json"
        ).is_file():
            print("Downloading ImageReward-v1.0")
            image_reward_model = RM.load(
                "ImageReward-v1.0", device="cpu", download_root=str(image_reward_dir)
            )
            del image_reward_model

        hps_root = Path(os.environ["HPS_ROOT"])
        download_file(HPS_URL, hps_root / "HPS_v2_compressed.pt")
        download_file(AESTHETIC_URL, args.aesthetic_checkpoint)

        print("Checking DINOv2 repository and ViT-L/14 register checkpoint")
        download_dinov2_repository(args.dinov2_repo)
        dinov2_model = torch.hub.load(
            args.dinov2_repo,
            "dinov2_vitl14_reg",
            source="local",
            pretrained=True,
        )
        del dinov2_model
        gc.collect()

    if world_size > 1:
        dist.barrier()


def shard(items, rank, world_size):
    return items[rank::world_size]


def make_loader(dataset, batch_size, num_workers, collate_fn=None):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn,
    )


def iter_progress(loader, name, rank):
    return tqdm(loader, desc=name, disable=rank != 0)


def gather_results(local_results, rank, world_size):
    if world_size == 1:
        return sorted(local_results)
    gathered = [None] * world_size if rank == 0 else None
    dist.gather_object(local_results, gathered, dst=0)
    if rank != 0:
        return None
    return sorted(item for rank_items in gathered for item in rank_items)


def finish_metric(objects, world_size):
    objects.clear()
    gc.collect()
    torch.cuda.empty_cache()
    if world_size > 1:
        dist.barrier()


def calculate_pickscore(loader, device, rank):
    from utils.pickscore_utils import PickScoreModel

    selector = PickScoreModel(device)
    results = []
    with torch.inference_mode():
        for filenames, images, prompts in iter_progress(loader, "PickScore", rank):
            image_inputs = selector.processor(images=images, return_tensors="pt").to(device)
            text_inputs = selector.processor(
                text=prompts, padding=True, truncation=True, max_length=77, return_tensors="pt"
            ).to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                image_features = selector.model.get_image_features(**image_inputs)
                text_features = selector.model.get_text_features(**text_inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                scores = (image_features * text_features).sum(dim=-1)
            results.extend(zip(filenames, scores.float().cpu().tolist(), prompts))
    return results, [selector]


def calculate_aesthetic(loader, device, rank, checkpoint_path, clip_cache_dir):
    import clip
    from utils.aes_utils import MLP

    clip_model, preprocess = clip.load("ViT-L/14", device=device, download_root=clip_cache_dir)
    aesthetic_model = MLP(768).to(device).eval()
    aesthetic_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    results = []
    with torch.inference_mode():
        for filenames, images, prompts in iter_progress(loader, "Aesthetic", rank):
            image_tensor = torch.stack([preprocess(image) for image in images]).to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                features = clip_model.encode_image(image_tensor)
                features = features / features.norm(dim=-1, keepdim=True)
                scores = aesthetic_model(features.float()).flatten()
            results.extend(zip(filenames, scores.cpu().tolist(), prompts))
    return results, [clip_model, aesthetic_model]


def calculate_hps(loader, device, rank):
    from utils.hps_utils import HPSModel

    selector = HPSModel(device)
    results = []
    with torch.inference_mode():
        for filenames, images, prompts in iter_progress(loader, "HPSv2", rank):
            image_tensor = torch.stack([selector.preprocess_val(image) for image in images]).to(
                device, non_blocking=True
            )
            text = selector.tokenizer(prompts).to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                outputs = selector.model(image_tensor, text)
                scores = (outputs["image_features"] * outputs["text_features"]).sum(dim=-1)
            results.extend(zip(filenames, scores.float().cpu().tolist(), prompts))
    return results, [selector]


def calculate_image_reward(loader, device, rank, image_reward_dir):
    RM = import_image_reward()

    image_reward_dir = Path(image_reward_dir)
    model = RM.load(
        str(image_reward_dir / "ImageReward.pt"),
        str(device),
        med_config=str(image_reward_dir / "med_config.json"),
    )
    results = []
    with torch.inference_mode():
        for filenames, images, prompts in iter_progress(loader, "ImageReward", rank):
            text = model.blip.tokenizer(
                prompts, padding="max_length", truncation=True, max_length=35, return_tensors="pt"
            ).to(device)
            image_tensor = torch.stack([model.preprocess(image) for image in images]).to(
                device, non_blocking=True
            )
            with torch.autocast("cuda", dtype=torch.float16):
                image_embeds = model.blip.visual_encoder(image_tensor)
                image_attention = torch.ones(image_embeds.shape[:-1], dtype=torch.long, device=device)
                text_output = model.blip.text_encoder(
                    text.input_ids,
                    attention_mask=text.attention_mask,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_attention,
                    return_dict=True,
                )
                rewards = model.mlp(text_output.last_hidden_state[:, 0, :].float()).flatten()
                rewards = (rewards - model.mean) / model.std
            results.extend(zip(filenames, rewards.cpu().tolist(), prompts))
    return results, [model]


def calculate_clip(loader, device, rank):
    from transformers import CLIPModel, CLIPProcessor

    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id, local_files_only=True).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_id, local_files_only=True)
    results = []
    with torch.inference_mode():
        for filenames, images, prompts in iter_progress(loader, "CLIP Score", rank):
            inputs = processor(
                text=prompts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            ).to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                scores = model(**inputs).logits_per_image.diagonal()
            results.extend(zip(filenames, scores.float().cpu().tolist(), prompts))
    return results, [model, processor]


def gather_feature_shards(local_features, total_count, world_size):
    """All ranks receive features in image_files order, including empty shards."""
    if world_size == 1:
        return local_features
    max_count = math.ceil(total_count / world_size)
    padded = local_features.new_zeros((max_count, local_features.shape[1]))
    padded[: len(local_features)] = local_features
    gathered = [torch.empty_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded)
    features = local_features.new_empty((total_count, local_features.shape[1]))
    for source_rank, rank_features in enumerate(gathered):
        count = len(range(source_rank, total_count, world_size))
        features[source_rank::world_size] = rank_features[:count]
    return features


def distributed_feature_statistics(image_dir, image_files, model, transform, args, device, rank, world_size, name):
    if len(image_files) < 2:
        raise ValueError(f"{name} requires at least two images, got {len(image_files)}")
    local_files = shard(image_files, rank, world_size)
    loader = make_loader(
        ImageOnlyDataset(image_dir, local_files, transform), args.fid_batch_size, args.num_workers
    )
    dimension = model.embed_dim
    feature_sum = torch.zeros(dimension, device=device, dtype=torch.float64)
    feature_outer = torch.zeros(dimension, dimension, device=device, dtype=torch.float64)
    count = torch.zeros(1, device=device, dtype=torch.float64)
    feature_batches = []
    with torch.inference_mode():
        for images in iter_progress(loader, name, rank):
            images = images.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                features = model(images)
            feature_batches.append(features.float())
            features = features.to(torch.float64)
            feature_sum += features.sum(dim=0)
            feature_outer += features.T @ features
            count += features.shape[0]
    if world_size > 1:
        dist.all_reduce(count)
        dist.all_reduce(feature_sum)
        dist.all_reduce(feature_outer)
    mean = feature_sum / count
    covariance = (feature_outer - count * torch.outer(mean, mean)) / (count - 1)
    local_features = (
        torch.cat(feature_batches) if feature_batches
        else torch.empty((0, dimension), device=device, dtype=torch.float32)
    )
    features = gather_feature_shards(local_features, len(image_files), world_size)
    return mean.cpu().numpy(), covariance.cpu().numpy(), features


def calculate_dino_diversity(student_features, student_files, prompt_dict, rank, world_size):
    groups = group_indices_by_prompt(student_files, prompt_dict)
    valid_groups = [(prompt, indices) for prompt, indices in groups.items() if len(indices) >= 2]
    if rank == 0:
        matched_count = sum(len(indices) for indices in groups.values())
        print(
            f"DINO Feature Diversity: {len(valid_groups)} valid prompts, "
            f"{len(groups) - len(valid_groups)} singleton prompts skipped, "
            f"{len(student_files) - matched_count} images without prompts skipped"
        )
        if not valid_groups:
            print("DINO Feature Diversity is N/A: no prompt has at least two images.")
    local_results = []
    for prompt, indices in iter_progress(shard(valid_groups, rank, world_size), "DINO diversity", rank):
        score = mean_pairwise_cosine_distance(student_features[indices])
        local_results.append((prompt, score, len(indices)))
    results = gather_results(local_results, rank, world_size)
    if rank != 0:
        return None
    if not results:
        return "DINO Feature Diversity", None, None, 0
    # Each valid prompt contributes one score, regardless of its sample count.
    return summarize("DINO Feature Diversity", results)


def calculate_dino_recall(teacher_features, student_features, args, rank, world_size):
    student_indices = torch.arange(len(student_features), device=student_features.device)[rank::world_size]
    local_radii = knn_squared_radii(
        student_features, student_indices, args.dinov2_recall_k, args.dinov2_distance_chunk_size
    )
    radii = gather_feature_shards(local_radii[:, None], len(student_features), world_size).flatten()
    covered = manifold_coverage(
        teacher_features[rank::world_size], student_features, radii, args.dinov2_distance_chunk_size
    )
    covered_count = covered.sum()
    if world_size > 1:
        dist.all_reduce(covered_count)
    recall = covered_count.item() / len(teacher_features)
    return "DINOv2 Recall-to-Teacher", recall, None, len(teacher_features)


def calculate_dino_metrics(args, prompt_dict, device, rank, world_size):
    teacher_files = sorted(
        path.name for path in Path(args.teacher_samples_dir).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    student_files = sorted(
        path.name for path in Path(args.student_samples_dir).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if len(teacher_files) < 2 or len(student_files) < 2:
        raise ValueError("DINO metrics require at least two teacher images and two student images")
    if len(student_files) <= args.dinov2_recall_k:
        raise ValueError(
            f"DINOv2 Recall-to-Teacher requires more than {args.dinov2_recall_k} student images; "
            "supply more samples or lower --dinov2_recall_k"
        )
    model = torch.hub.load(
        args.dinov2_repo, "dinov2_vitl14_reg", source="local", pretrained=True
    ).eval().to(device)
    transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    mu_real, sigma_real, teacher_features = distributed_feature_statistics(
        args.teacher_samples_dir, teacher_files, model, transform, args, device, rank, world_size, "DINO teacher"
    )
    mu_gen, sigma_gen, student_features = distributed_feature_statistics(
        args.student_samples_dir, student_files, model, transform, args, device, rank, world_size, "DINO student"
    )
    fid = None
    if rank == 0:
        difference = mu_real - mu_gen
        covariance_mean = linalg.sqrtm(sigma_real @ sigma_gen)
        if not np.isfinite(covariance_mean).all():
            offset = np.eye(sigma_real.shape[0]) * 1e-6
            covariance_mean = linalg.sqrtm((sigma_real + offset) @ (sigma_gen + offset))
        if np.iscomplexobj(covariance_mean):
            covariance_mean = covariance_mean.real
        fid = float(
            difference @ difference + np.trace(sigma_real) + np.trace(sigma_gen) - 2 * np.trace(covariance_mean)
        )
    diversity = calculate_dino_diversity(student_features, student_files, prompt_dict, rank, world_size)
    if rank == 0:
        print(f"DINOv2 Recall-to-Teacher: k={args.dinov2_recall_k}, raw CLS Euclidean distance")
    recall = calculate_dino_recall(teacher_features, student_features, args, rank, world_size)
    return fid, diversity, recall, [model]


def summarize(name, results):
    values = np.asarray([score for _, score, _ in results], dtype=np.float64)
    return name, float(values.mean()), float(values.std()), len(values)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    rank, world_size, device = init_distributed()
    prepare_models(args, rank, world_size)

    prompt_path = Path(args.prompt_json_path or (args.student_samples_dir + ".json"))
    with open(prompt_path, "r", encoding="utf-8") as file:
        prompt_dict = json.load(file)
    image_files = sorted(
        [name for name in prompt_dict if (Path(args.student_samples_dir) / name).is_file()],
        key=lambda name: (0, int(Path(name).stem), name) if Path(name).stem.isdigit() else (1, name),
    )
    local_files = shard(image_files, rank, world_size)
    dataset = ImagePromptDataset(args.student_samples_dir, prompt_dict, local_files)
    if rank == 0:
        print(f"Processing {len(image_files)} images on {world_size} GPUs")
        print(f"Each GPU receives approximately {math.ceil(len(image_files) / world_size)} images")

    summaries = []
    metrics = [
        ("PickScore", lambda loader: calculate_pickscore(loader, device, rank)),
        (
            "Aesthetic",
            lambda loader: calculate_aesthetic(
                loader, device, rank, args.aesthetic_checkpoint, args.clip_cache_dir
            ),
        ),
        ("HPSv2", lambda loader: calculate_hps(loader, device, rank)),
        (
            "ImageReward",
            lambda loader: calculate_image_reward(loader, device, rank, args.image_reward_dir),
        ),
        ("CLIP Score", lambda loader: calculate_clip(loader, device, rank)),
    ]
    for name, calculate in metrics:
        if rank == 0:
            print(f"\n=== {name} ===")
        loader = make_loader(dataset, args.batch_size, args.num_workers, collate_image_prompts)
        local_results, objects = calculate(loader)
        results = gather_results(local_results, rank, world_size)
        if rank == 0:
            summaries.append(summarize(name, results))
        del loader
        finish_metric(objects, world_size)

    if rank == 0:
        print("\n=== DINOv2 FID / DINO Feature Diversity / DINOv2 Recall-to-Teacher ===")
    fid, diversity, recall, objects = calculate_dino_metrics(args, prompt_dict, device, rank, world_size)
    finish_metric(objects, world_size)

    if rank == 0:
        summaries.extend([("DINOv2 FID", fid, None, None), diversity, recall])
        print("\n" + "=" * 80)
        print(f"{'Metric':<28} {'Mean':<14} {'Std':<14} {'Samples/Prompts':<16}")
        print("-" * 80)
        for name, mean, std, count in summaries:
            mean_text = "N/A" if mean is None else f"{mean:.4f}"
            std_text = "N/A" if std is None else f"{std:.4f}"
            count_text = "N/A" if count is None else str(count)
            print(f"{name:<28} {mean_text:<14} {std_text:<14} {count_text:<16}")
        print("Diversity count = valid prompts; Recall count = teacher images.")
        print("=" * 80)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
