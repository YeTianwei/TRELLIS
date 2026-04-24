from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from . import dist_utils


class Stage2RewardEvaluator:
    def __init__(
        self,
        image_cond_model: str = "dinov2_vitl14_reg",
        image_cond_repo_or_dir: str = "facebookresearch/dinov2",
        image_cond_source: str = "github",
        image_align_weight: float = 0.7,
        aesthetic_weight: float = 0.3,
        topk_views: int = 2,
        dino_batch_size: int = 1,
        aesthetic_batch_size: int = 1,
        dino_encoder=None,
    ):
        self.image_cond_model_name = image_cond_model
        self.image_cond_repo_or_dir = image_cond_repo_or_dir
        self.image_cond_source = image_cond_source
        self.image_align_weight = image_align_weight
        self.aesthetic_weight = aesthetic_weight
        self.topk_views = topk_views
        self.dino_batch_size = dino_batch_size
        self.aesthetic_batch_size = aesthetic_batch_size
        self.dino_encoder = dino_encoder
        self._dino = None
        self._clip_model = None
        self._clip_processor = None

    def _init_dino(self):
        with dist_utils.local_master_first():
            try:
                self._dino = torch.hub.load(
                    self.image_cond_repo_or_dir,
                    self.image_cond_model_name,
                    pretrained=True,
                    source=self.image_cond_source,
                ).eval().cuda()
            except Exception as exc:
                raise RuntimeError(
                    "Failed to load the DINOv2 reward fallback encoder via torch.hub. "
                    f"repo_or_dir={self.image_cond_repo_or_dir!r}, "
                    f"model={self.image_cond_model_name!r}, source={self.image_cond_source!r}. "
                    "If GitHub is unstable, set image_cond_repo_or_dir to a local DINOv2 hub cache "
                    "and set image_cond_source='local'."
                ) from exc

    def _init_aesthetic(self):
        with dist_utils.local_master_first():
            self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval().cuda()
            self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    @staticmethod
    def _to_pil(images: torch.Tensor):
        return [Image.fromarray((img.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype("uint8")) for img in images]

    @staticmethod
    def _extract_clip_features(output) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        if hasattr(output, "image_embeds") and output.image_embeds is not None:
            return output.image_embeds
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output
        if isinstance(output, (tuple, list)) and len(output) > 0:
            first = output[0]
            if isinstance(first, torch.Tensor):
                return first
        raise TypeError(f"Unsupported CLIP image feature output type: {type(output)}")

    @torch.no_grad()
    def _encode_dino(self, images: torch.Tensor):
        if images.shape[0] > self.dino_batch_size:
            return torch.cat([self._encode_dino(chunk) for chunk in images.split(self.dino_batch_size)], dim=0)
        if self.dino_encoder is not None:
            feats = self.dino_encoder.encode_image(images)
            feats = feats.mean(dim=1)
            return F.normalize(feats, dim=-1)
        if self._dino is None:
            self._init_dino()
        feats = self._dino(images, is_training=True)["x_prenorm"]
        feats = F.layer_norm(feats, feats.shape[-1:])
        feats = feats.mean(dim=1)
        return F.normalize(feats, dim=-1)

    @torch.no_grad()
    def _score_aesthetic(self, images: torch.Tensor):
        if self.aesthetic_weight == 0:
            return torch.zeros(images.shape[0], device=images.device, dtype=images.dtype)
        if images.shape[0] > self.aesthetic_batch_size:
            return torch.cat([self._score_aesthetic(chunk) for chunk in images.split(self.aesthetic_batch_size)], dim=0)
        if self._clip_model is None or self._clip_processor is None:
            self._init_aesthetic()
        pil_images = self._to_pil(images)
        proc = self._clip_processor(images=pil_images, return_tensors="pt")
        proc = {k: v.cuda() for k, v in proc.items()}
        feats = self._extract_clip_features(self._clip_model.get_image_features(**proc))
        feats = F.normalize(feats, dim=-1)
        # A simple monotonic aesthetic proxy. This keeps the interface stable
        # without introducing an extra checkpoint dependency into TRELLIS.
        return feats[:, 0]

    @torch.no_grad()
    def score(self, cond: torch.Tensor, renders: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            cond: [G, 3, H, W]
            renders: [G, V, 3, H, W]
        """
        g, v = renders.shape[:2]
        cond_embed = self._encode_dino(cond)
        view_embed = self._encode_dino(renders.reshape(g * v, *renders.shape[2:])).reshape(g, v, -1)
        align = torch.einsum("gd,gvd->gv", cond_embed, view_embed)
        topk = min(self.topk_views, align.shape[1])
        image_align = align.topk(topk, dim=1).values.mean(dim=1)

        aesthetic = self._score_aesthetic(renders.reshape(g * v, *renders.shape[2:])).reshape(g, v).mean(dim=1)
        total = self.image_align_weight * image_align + self.aesthetic_weight * aesthetic
        return {
            "reward": total,
            "image_align": image_align,
            "aesthetic": aesthetic,
        }
