from __future__ import annotations

from typing import Dict, Optional

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
        normal_weight: float = 0.0,
        normal_mask_threshold: float = 0.1,
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
        self.normal_weight = normal_weight
        self.normal_mask_threshold = normal_mask_threshold
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
    def _depth_to_normal(self, depth: torch.Tensor) -> torch.Tensor:
        dzdx = F.pad(depth[..., :, 2:] - depth[..., :, :-2], (1, 1, 0, 0), mode="replicate") * 0.5
        dzdy = F.pad(depth[..., 2:, :] - depth[..., :-2, :], (0, 0, 1, 1), mode="replicate") * 0.5
        normal = torch.stack([-dzdx, -dzdy, torch.ones_like(depth)], dim=-3)
        return F.normalize(normal, dim=-3, eps=1e-6)

    @torch.no_grad()
    def _score_normal_depth_consistency(
        self,
        renders: torch.Tensor,
        render_info: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        g, v = renders.shape[:2]
        if self.normal_weight == 0 or not render_info or "depth" not in render_info:
            return torch.zeros(g, device=renders.device, dtype=renders.dtype)

        depth = render_info["depth"].to(device=renders.device, dtype=renders.dtype)
        if depth.dim() == 5 and depth.shape[2] == 1:
            depth = depth[:, :, 0]
        if depth.shape[:2] != (g, v):
            return torch.zeros(g, device=renders.device, dtype=renders.dtype)

        alpha = render_info.get("alpha")
        if alpha is None:
            mask = torch.isfinite(depth) & (depth > 0)
        else:
            alpha = alpha.to(device=renders.device, dtype=renders.dtype)
            if alpha.dim() == 5 and alpha.shape[2] == 1:
                alpha = alpha[:, :, 0]
            mask = alpha > self.normal_mask_threshold
        mask = mask & torch.isfinite(depth)

        depth = depth.nan_to_num(0.0).clamp(0, 1)
        depth_normal = self._depth_to_normal(depth)

        raw_normal = render_info.get("normal")
        if raw_normal is not None:
            raw_normal = raw_normal.to(device=renders.device, dtype=renders.dtype)
            if raw_normal.min() >= 0 and raw_normal.max() <= 1:
                raw_normal = raw_normal * 2 - 1
            raw_normal = F.normalize(raw_normal, dim=-3, eps=1e-6)
            cosine = (raw_normal * depth_normal).sum(dim=-3).clamp(-1, 1)
            score = (cosine + 1) * 0.5
        else:
            smooth_normal = F.avg_pool2d(
                depth_normal.reshape(g * v, 3, *depth_normal.shape[-2:]),
                kernel_size=3,
                stride=1,
                padding=1,
            ).reshape_as(depth_normal)
            smooth_normal = F.normalize(smooth_normal, dim=-3, eps=1e-6)
            score = (depth_normal * smooth_normal).sum(dim=-3).clamp(0, 1)

        valid = mask.reshape(g, v, -1)
        score = score.reshape(g, v, -1)
        denom = valid.float().sum(dim=(1, 2)).clamp_min(1.0)
        return (score * valid.float()).sum(dim=(1, 2)) / denom

    @torch.no_grad()
    def score(
        self,
        cond: torch.Tensor,
        renders: torch.Tensor,
        render_info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
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
        normal_depth_consistency = self._score_normal_depth_consistency(renders, render_info)
        total = (
            self.image_align_weight * image_align
            + self.aesthetic_weight * aesthetic
            + self.normal_weight * normal_depth_consistency
        )
        return {
            "reward": total,
            "image_align": image_align,
            "aesthetic": aesthetic,
            "normal_depth_consistency": normal_depth_consistency,
        }
