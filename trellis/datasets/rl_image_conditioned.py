import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .components import StandardDatasetBase


class ImageConditionedRL(StandardDatasetBase):
    """
    Lightweight image-conditioned dataset for online RL.

    It only loads conditioning images from `renders_cond/<sha256>/...` and avoids
    loading any stage-2 latent supervision.
    """

    def __init__(
        self,
        roots: str,
        *,
        image_size: int = 518,
        min_aesthetic_score: float = 4.5,
    ):
        self.image_size = image_size
        self.min_aesthetic_score = min_aesthetic_score
        self.value_range = (0, 1)
        super().__init__(roots)
        self.loads = [1 for _ in self.instances]

    def filter_metadata(self, metadata: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        stats = {}
        metadata = metadata[metadata["cond_rendered"]]
        stats["Cond rendered"] = len(metadata)
        if "aesthetic_score" in metadata:
            metadata = metadata[metadata["aesthetic_score"] >= self.min_aesthetic_score]
            stats[f"Aesthetic score >= {self.min_aesthetic_score}"] = len(metadata)
        return metadata, stats

    def get_instance(self, root: str, instance: str):
        image_root = os.path.join(root, "renders_cond", instance)
        with open(os.path.join(image_root, "transforms.json"), "r") as fp:
            metadata = json.load(fp)
        view = np.random.randint(len(metadata["frames"]))
        frame = metadata["frames"][view]
        image_path = os.path.join(image_root, frame["file_path"])
        image = Image.open(image_path)

        alpha = np.array(image.getchannel(3))
        bbox = np.array(alpha).nonzero()
        bbox = [bbox[1].min(), bbox[0].min(), bbox[1].max(), bbox[0].max()]
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        half_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
        half_size *= 1.2
        crop_box = [
            int(center[0] - half_size),
            int(center[1] - half_size),
            int(center[0] + half_size),
            int(center[1] + half_size),
        ]
        image = image.crop(crop_box)
        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        alpha = image.getchannel(3)
        image = image.convert("RGB")
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        alpha = torch.tensor(np.array(alpha)).float() / 255.0
        image = image * alpha.unsqueeze(0)
        return {
            "cond": image,
            "instance": instance,
        }

    @staticmethod
    def collate_fn(batch):
        cond = torch.stack([b["cond"] for b in batch])
        instances = [b["instance"] for b in batch]
        return {
            "cond": cond,
            "instance": instances,
        }

    @torch.no_grad()
    def visualize_sample(self, sample):
        return sample["cond"] if isinstance(sample, dict) else sample
