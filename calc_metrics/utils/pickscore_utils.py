from transformers import AutoModel, AutoProcessor

processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"


class PickScoreModel:
    def __init__(self, device):
        self.processor = AutoProcessor.from_pretrained(
            processor_name_or_path, local_files_only=True
        )
        self.model = AutoModel.from_pretrained(
            model_pretrained_name_or_path, local_files_only=True
        ).eval().to(device)
