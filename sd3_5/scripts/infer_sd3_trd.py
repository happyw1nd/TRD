#!/usr/bin/env python3
"""Distributed inference for SD3/SD3.5 TRD checkpoints."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sd3_arcflow import SD3ArcFlowTransformer2DModel, StableDiffusion3ArcFlowPipeline


LORA_WEIGHT_NAME = "pytorch_lora_weights.safetensors"
ARCFLOW_HEADS_NAME = "arcflow_output_heads.pt"
ARCFLOW_PREFIXES = (
    "arc_tail_blocks.",
    "arc_basic_velocity_heads.",
    "arc_gate_heads.",
    "arc_momentum_heads.",
)


class PromptTSVDataset(Dataset):
    def __init__(self, path):
        self.prompts = []
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2 or not parts[1].strip():
                    continue
                self.prompts.append(parts[1])
        if not self.prompts:
            raise ValueError(f"Prompt TSV contains no valid rows: {path}")

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        return self.prompts[index]


class ShardedPromptDataset(Dataset):
    def __init__(self, dataset, selected_indices, rank, world_size):
        self.items = [
            (output_index, dataset[selected_indices[output_index]])
            for output_index in range(rank, len(selected_indices), world_size)
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        output_index, prompt = self.items[index]
        return {"output_index": output_index, "prompt": prompt}


def parse_timesteps(value):
    if value is None or not value.strip():
        return None
    try:
        timesteps = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("timesteps must be comma-separated numbers") from error
    if not timesteps or any(left <= right for left, right in zip(timesteps, timesteps[1:])):
        raise argparse.ArgumentTypeError("timesteps must be a strictly decreasing list")
    return timesteps


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--prompt_database", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--prompt_map_path")
    parser.add_argument("--num_gen_pics", type=int, default=5000)
    parser.add_argument("--num_images_per_prompt", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=2)
    parser.add_argument("--denoising_timesteps", type=parse_timesteps)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--max_sequence_length", type=int, default=256)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--revision")
    parser.add_argument("--variant")
    parser.add_argument("--num_arc_heads", type=int)
    parser.add_argument("--num_arc_head_layers", type=int)
    parser.add_argument("--momentum_scale", type=float, default=1.0)
    parser.add_argument("--momentum_clamp", type=float, default=20.0)
    parser.add_argument("--consistency_sampling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable_model_cpu_offload", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_seed", type=int, default=666)
    args = parser.parse_args(input_args)

    positive = (
        "num_gen_pics",
        "num_images_per_prompt",
        "batch_size",
        "height",
        "width",
        "num_inference_steps",
        "max_sequence_length",
        "momentum_clamp",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.num_workers < 0:
        parser.error("--num_workers must be nonnegative")
    if args.max_sequence_length > 512:
        parser.error("--max_sequence_length must not exceed 512")
    if args.denoising_timesteps is not None:
        if args.num_inference_steps != len(args.denoising_timesteps):
            parser.error("--num_inference_steps must equal the number of --denoising_timesteps")
    return args


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


def resolve_checkpoint(path):
    checkpoint = Path(path).expanduser().resolve()

    def complete(directory):
        return all((directory / name).is_file() for name in (LORA_WEIGHT_NAME, ARCFLOW_HEADS_NAME))

    if complete(checkpoint):
        return checkpoint
    candidates = [
        child
        for child in checkpoint.glob("checkpoint-*")
        if child.is_dir() and child.name.removeprefix("checkpoint-").isdigit() and complete(child)
    ]
    if candidates:
        return max(candidates, key=lambda child: int(child.name.split("-")[1]))
    raise FileNotFoundError(
        f"No complete TRD checkpoint under {checkpoint}; required: "
        f"{LORA_WEIGHT_NAME}, {ARCFLOW_HEADS_NAME}"
    )


def arcflow_structure(state):
    heads = sorted(
        {int(key.split(".")[1]) for key in state if key.startswith("arc_basic_velocity_heads.")}
    )
    layers = {int(key.split(".")[2]) for key in state if key.startswith("arc_tail_blocks.")}
    if not heads or heads != list(range(len(heads))):
        raise ValueError("Invalid ArcFlow output-head indices in checkpoint.")
    return len(heads), max(layers) + 1 if layers else 0


def load_arcflow_heads(transformer, state):
    expected = {key for key in transformer.state_dict() if key.startswith(ARCFLOW_PREFIXES)}
    if set(state) != expected:
        raise ValueError(
            f"ArcFlow head mismatch: missing={sorted(expected - set(state))}, "
            f"unexpected={sorted(set(state) - expected)}"
        )
    transformer.load_state_dict(state, strict=False)


def create_pipeline(args, checkpoint, device):
    from diffusers import SD3Transformer2DModel

    dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
        args.mixed_precision
    ]
    loading = {"revision": args.revision, "variant": args.variant, "torch_dtype": dtype}
    state = torch.load(checkpoint / ARCFLOW_HEADS_NAME, map_location="cpu", weights_only=True)
    heads, layers = arcflow_structure(state)
    if args.num_arc_heads is not None and args.num_arc_heads != heads:
        raise ValueError(f"Checkpoint has {heads} ArcFlow heads, not {args.num_arc_heads}.")
    if args.num_arc_head_layers is not None and args.num_arc_head_layers != layers:
        raise ValueError(f"Checkpoint has {layers} ArcFlow tail layers, not {args.num_arc_head_layers}.")

    base = SD3Transformer2DModel.from_pretrained(
        args.model_name_or_path, subfolder="transformer", **loading
    )
    transformer = SD3ArcFlowTransformer2DModel.from_sd3_transformer(
        base, num_arc_heads=heads, num_arc_head_layers=layers
    )
    del base
    pipeline = StableDiffusion3ArcFlowPipeline.from_pretrained(
        args.model_name_or_path, transformer=transformer, **loading
    )
    pipeline.load_lora_weights(
        str(checkpoint), weight_name=LORA_WEIGHT_NAME, adapter_name="student"
    )
    pipeline.set_adapters("student")
    load_arcflow_heads(pipeline.transformer, state)

    divisor = pipeline.vae_scale_factor * pipeline.transformer.config.patch_size
    if args.height % divisor or args.width % divisor:
        raise ValueError(f"height and width must be divisible by {divisor}.")
    pipeline.transformer.eval()
    if args.enable_model_cpu_offload:
        pipeline.enable_model_cpu_offload(gpu_id=device.index)
    else:
        pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def main(input_args=None):
    args = parse_args(input_args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TRD inference.")
    checkpoint = resolve_checkpoint(args.checkpoint_dir)
    dataset = PromptTSVDataset(args.prompt_database)
    rank, local_rank, world_size, device = init_distributed()
    try:
        save_dir = Path(args.save_dir).expanduser().resolve()
        prompt_map_path = (
            Path(args.prompt_map_path).expanduser().resolve()
            if args.prompt_map_path
            else save_dir.with_suffix(".json")
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        prompt_map_path.parent.mkdir(parents=True, exist_ok=True)

        count = min(args.num_gen_pics, len(dataset))
        selected = torch.randperm(
            len(dataset), generator=torch.Generator().manual_seed(args.dataset_seed)
        )[:count].tolist()
        shard = ShardedPromptDataset(dataset, selected, rank, world_size)
        dataloader = DataLoader(
            shard, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )
        pipeline = create_pipeline(args, checkpoint, device) if len(shard) else None
        prompt_map = {}
        for batch in tqdm(dataloader, desc=f"TRD GPU {local_rank}", disable=rank != 0):
            prompt_indices = batch["output_index"].tolist()
            prompts = list(batch["prompt"])
            output_indices = [
                prompt_index * args.num_images_per_prompt + image_index
                for prompt_index in prompt_indices
                for image_index in range(args.num_images_per_prompt)
            ]
            generators = [
                torch.Generator(device=device).manual_seed(args.seed + index)
                for index in output_indices
            ]
            images = pipeline(
                prompt=prompts,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                timesteps=args.denoising_timesteps,
                guidance_scale=args.guidance_scale,
                num_images_per_prompt=args.num_images_per_prompt,
                generator=generators,
                max_sequence_length=args.max_sequence_length,
                momentum_scale=args.momentum_scale,
                momentum_clamp=args.momentum_clamp,
                consistency_sampling=args.consistency_sampling,
            ).images
            output_prompts = [
                prompt for prompt in prompts for _ in range(args.num_images_per_prompt)
            ]
            for index, prompt, image in zip(output_indices, output_prompts, images):
                filename = f"{index:06d}.jpg"
                image.save(save_dir / filename)
                prompt_map[filename] = prompt

        fragment = prompt_map_path.parent / f".{prompt_map_path.name}.rank-{rank}.json"
        with fragment.open("w", encoding="utf-8") as file:
            json.dump(prompt_map, file, ensure_ascii=False, indent=2)
        if world_size > 1:
            dist.barrier()
        if rank == 0:
            merged = {}
            for source_rank in range(world_size):
                source = prompt_map_path.parent / (
                    f".{prompt_map_path.name}.rank-{source_rank}.json"
                )
                with source.open(encoding="utf-8") as file:
                    merged.update(json.load(file))
            expected = count * args.num_images_per_prompt
            if len(merged) != expected:
                raise RuntimeError(f"Expected {expected} generated records; got {len(merged)}.")
            with prompt_map_path.open("w", encoding="utf-8") as file:
                json.dump(dict(sorted(merged.items())), file, ensure_ascii=False, indent=2)
            for source_rank in range(world_size):
                (
                    prompt_map_path.parent
                    / f".{prompt_map_path.name}.rank-{source_rank}.json"
                ).unlink()
            print(f"Saved {expected} TRD images to {save_dir}; prompt map: {prompt_map_path}")
        if world_size > 1:
            dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
