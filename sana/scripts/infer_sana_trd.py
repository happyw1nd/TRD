#!/usr/bin/env python3
"""Generate images with a Sana checkpoint trained by the TRD script."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


LORA_WEIGHT_NAME = "pytorch_lora_weights.safetensors"
ARCFLOW_OUTPUT_HEADS_FILE = "arcflow_output_heads.pt"
ARCFLOW_OUTPUT_HEAD_PREFIXES = (
    "arc_tail_blocks",
    "arc_basic_velocity_heads",
    "arc_gate_heads",
    "arc_momentum_heads",
)


class PromptTSVDataset(Dataset):
    def __init__(self, path: str):
        self.prompts = []
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                if parts[1]:
                    self.prompts.append(parts[1])
        if not self.prompts:
            raise ValueError(f"No prompts were found in TSV file: {path}")

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        return self.prompts[index]


class ShardedPromptDataset(Dataset):
    def __init__(self, dataset, selected_indices, rank, world_size):
        self.items = [
            (output_index, dataset.prompts[selected_indices[output_index]])
            for output_index in range(rank, len(selected_indices), world_size)
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        output_index, prompt = self.items[index]
        return {"output_index": output_index, "prompt": prompt}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_name_or_path",
        required=True,
        help="Local model path or Hugging Face model identifier.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        required=True,
        help="A checkpoint-N directory, or an output directory whose latest checkpoint will be used.",
    )
    parser.add_argument(
        "--prompt_database",
        required=True,
        help="Path to a UTF-8 TSV file whose second column contains prompts.",
    )
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--prompt_map_path", default=None)
    parser.add_argument("--revision", default=None, help="Optional model revision.")
    parser.add_argument("--variant", default=None, help="Optional model file variant such as bf16 or fp16.")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
        help="Torch dtype used to load and run the pipeline.",
    )
    parser.add_argument("--num_gen_pics", type=int, default=5000)
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="Number of images to generate for each selected prompt.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Inference batch size per GPU.")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers per process/GPU.")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=1)
    parser.add_argument("--guidance_scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_seed", type=int, default=666)
    # These values must match the ArcFlow structure used during training.
    parser.add_argument("--num_arc_heads", type=int, default=4)
    parser.add_argument("--num_arc_head_layers", type=int, default=1)
    parser.add_argument("--momentum_scale", type=float, default=1.0)
    parser.add_argument("--momentum_clamp", type=float, default=20.0)
    parser.add_argument(
        "--consistency_sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Re-noise intermediate predictions between multi-step TRD updates.",
    )
    return parser.parse_args()


def resolve_checkpoint(path: str) -> Path:
    checkpoint = Path(path).expanduser().resolve()
    if (checkpoint / LORA_WEIGHT_NAME).is_file() and (
        checkpoint / ARCFLOW_OUTPUT_HEADS_FILE
    ).is_file():
        return checkpoint

    candidates = []
    if checkpoint.is_dir():
        for child in checkpoint.iterdir():
            if not child.is_dir() or not child.name.startswith("checkpoint-"):
                continue
            try:
                step = int(child.name.split("-", 1)[1])
            except ValueError:
                continue
            candidates.append((step, child))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-N directory found under {checkpoint}")
    return max(candidates, key=lambda item: item[0])[1]


def load_trd_checkpoint(pipe, checkpoint_dir: Path) -> None:
    lora_path = checkpoint_dir / LORA_WEIGHT_NAME
    heads_path = checkpoint_dir / ARCFLOW_OUTPUT_HEADS_FILE
    if not lora_path.is_file():
        raise FileNotFoundError(f"LoRA checkpoint not found: {lora_path}")
    if not heads_path.is_file():
        raise FileNotFoundError(f"TRD ArcFlow output heads not found: {heads_path}")

    pipe.load_lora_weights(str(checkpoint_dir), weight_name=LORA_WEIGHT_NAME)
    state_dict = torch.load(heads_path, map_location="cpu")
    missing_keys, unexpected_keys = pipe.transformer.load_state_dict(state_dict, strict=False)
    missing_flow_keys = [
        key for key in missing_keys if key.startswith(ARCFLOW_OUTPUT_HEAD_PREFIXES)
    ]
    unexpected_flow_keys = [
        key for key in unexpected_keys if key.startswith(ARCFLOW_OUTPUT_HEAD_PREFIXES)
    ]
    if missing_flow_keys or unexpected_flow_keys:
        raise RuntimeError(
            "TRD output-head checkpoint does not match the model structure. "
            f"Missing keys: {missing_flow_keys}; unexpected keys: {unexpected_flow_keys}. "
            "Check --num_arc_heads and --num_arc_head_layers."
        )


def create_pipeline(args, checkpoint_dir: Path, device: torch.device):
    from sana_arcflow import SanaArcFlowPipeline

    torch_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    pipe = SanaArcFlowPipeline.from_sana_pretrained(
        args.model_name_or_path,
        revision=args.revision or None,
        variant=args.variant or None,
        torch_dtype=torch_dtype,
        num_arc_heads=args.num_arc_heads,
        num_arc_head_layers=args.num_arc_head_layers,
    )
    load_trd_checkpoint(pipe, checkpoint_dir)
    pipe.text_encoder.to(dtype=torch_dtype)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank}, but only {torch.cuda.device_count()} CUDA devices are visible."
        )
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size, torch.device("cuda", local_rank)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Sana TRD inference.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.num_gen_pics <= 0:
        raise ValueError("--num_gen_pics must be positive.")
    if args.num_images_per_prompt <= 0:
        raise ValueError("--num_images_per_prompt must be positive.")

    rank, local_rank, world_size, device = init_distributed()

    checkpoint_dir = resolve_checkpoint(args.checkpoint_dir)
    save_dir = Path(args.save_dir).expanduser().resolve()
    prompt_map_path = (
        Path(args.prompt_map_path).expanduser().resolve()
        if args.prompt_map_path
        else save_dir.with_suffix(".json")
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    prompt_map_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = PromptTSVDataset(args.prompt_database)
    num_to_generate = min(args.num_gen_pics, len(dataset))
    shuffle_generator = torch.Generator().manual_seed(args.dataset_seed)
    selected_indices = torch.randperm(len(dataset), generator=shuffle_generator)[
        :num_to_generate
    ].tolist()
    rank_dataset = ShardedPromptDataset(
        dataset, selected_indices, rank=rank, world_size=world_size
    )
    dataloader = DataLoader(
        rank_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    pipe = create_pipeline(args, checkpoint_dir, device)

    print(
        f"[rank {rank}/{world_size}, cuda:{local_rank}] Using TRD checkpoint: "
        f"{checkpoint_dir}; assigned {len(rank_dataset)} prompts / "
        f"{len(rank_dataset) * args.num_images_per_prompt} images"
    )
    prompt_map = {}
    progress = tqdm(
        total=len(rank_dataset) * args.num_images_per_prompt,
        desc=f"GPU {local_rank}",
        position=local_rank,
        disable=world_size > 1 and rank != 0,
    )
    with progress:
        for batch in dataloader:
            prompt_indices = batch["output_index"].tolist()
            batch_prompts = list(batch["prompt"])
            output_indices = [
                prompt_index * args.num_images_per_prompt + image_index
                for prompt_index in prompt_indices
                for image_index in range(args.num_images_per_prompt)
            ]
            generators = [
                torch.Generator(device=device).manual_seed(args.seed + output_index)
                for output_index in output_indices
            ]
            images = pipe(
                prompt=batch_prompts,
                height=args.height,
                width=args.width,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                num_images_per_prompt=args.num_images_per_prompt,
                generator=generators,
                momentum_scale=args.momentum_scale,
                momentum_clamp=args.momentum_clamp,
                consistency_sampling=args.consistency_sampling,
            ).images

            output_prompts = [
                prompt for prompt in batch_prompts for _ in range(args.num_images_per_prompt)
            ]
            for output_index, prompt, image in zip(output_indices, output_prompts, images):
                filename = f"{output_index:06d}.jpg"
                image.save(save_dir / filename)
                prompt_map[filename] = prompt
                progress.update(1)

    fragment_path = prompt_map_path.parent / f".{prompt_map_path.name}.rank-{rank}.json"
    with open(fragment_path, "w", encoding="utf-8") as file:
        json.dump(prompt_map, file, ensure_ascii=False, indent=4)

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        expected_image_count = num_to_generate * args.num_images_per_prompt
        merged_prompt_map = {}
        for source_rank in range(world_size):
            source_path = (
                prompt_map_path.parent
                / f".{prompt_map_path.name}.rank-{source_rank}.json"
            )
            with open(source_path, "r", encoding="utf-8") as file:
                merged_prompt_map.update(json.load(file))
        merged_prompt_map = dict(sorted(merged_prompt_map.items()))
        if len(merged_prompt_map) != expected_image_count:
            raise RuntimeError(
                f"Expected {expected_image_count} prompt records, got {len(merged_prompt_map)}."
            )
        with open(prompt_map_path, "w", encoding="utf-8") as file:
            json.dump(merged_prompt_map, file, ensure_ascii=False, indent=4)
        for source_rank in range(world_size):
            (
                prompt_map_path.parent
                / f".{prompt_map_path.name}.rank-{source_rank}.json"
            ).unlink()
        print(f"Saved {expected_image_count} images to {save_dir}")
        print(f"Saved prompt map to {prompt_map_path}")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
