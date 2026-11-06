import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .open_clip.factory import create_model_and_transforms


class HPSModel:
    def __init__(self, device):
        model, _, self.preprocess_val = create_model_and_transforms(
            "ViT-H-14",
            "laion2B-s32B-b79K",
            precision="amp",
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False,
        )
        checkpoint_path = Path(os.environ["HPS_ROOT"]) / "HPS_v2_compressed.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"HPS checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        self._tokenizer = AutoTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32",
            local_files_only=True,
        )
        self.model = model.to(device).eval()

    def tokenizer(self, prompts):
        return self._tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).input_ids
