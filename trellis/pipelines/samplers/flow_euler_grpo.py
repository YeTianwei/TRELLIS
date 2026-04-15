from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
from easydict import EasyDict as edict

from ...modules import sparse as sp
from .flow_euler import FlowEulerSampler


def _sparse_gaussian_log_prob(x: sp.SparseTensor, mean: sp.SparseTensor, std: torch.Tensor) -> torch.Tensor:
    var = std.square().clamp_min(1e-8)
    log_scale = torch.log(std.clamp_min(1e-8))
    log_probs = -0.5 * (((x.feats - mean.feats) ** 2) / var + 2 * log_scale + np.log(2 * np.pi))
    per_sample = []
    for sample_slice in x.layout:
        per_sample.append(log_probs[sample_slice].mean())
    return torch.stack(per_sample)


class FlowEulerGRPOSampler(FlowEulerSampler):
    """
    Flow Euler sampler that caches trajectory tensors and per-step log-probabilities
    for GRPO-style RL updates.
    """

    def _cfg_predict_v(
        self,
        model,
        x_t,
        t: float,
        cond,
        neg_cond: Optional[Any] = None,
        cfg_strength: float = 1.0,
        **kwargs,
    ):
        pred = self._inference_model(model, x_t, t, cond, **kwargs)
        if neg_cond is None or cfg_strength == 1.0:
            return pred
        pred_neg = self._inference_model(model, x_t, t, neg_cond, **kwargs)
        return pred_neg + cfg_strength * (pred - pred_neg)

    def step_with_logprob(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond,
        neg_cond: Optional[Any] = None,
        cfg_strength: float = 1.0,
        noise_level: float = 0.0,
        generator: Optional[torch.Generator] = None,
        next_sample=None,
        **kwargs,
    ):
        pred_v = self._cfg_predict_v(model, x_t, t, cond, neg_cond=neg_cond, cfg_strength=cfg_strength, **kwargs)
        pred_x_prev = x_t - (t - t_prev) * pred_v
        dt = max(float(t - t_prev), 0.0)
        std = torch.full((x_t.shape[0],), noise_level * np.sqrt(dt), device=x_t.device, dtype=x_t.feats.dtype if isinstance(x_t, sp.SparseTensor) else x_t.dtype)
        if next_sample is None:
            if noise_level > 0 and t_prev > 0:
                if isinstance(x_t, sp.SparseTensor):
                    eps = torch.randn(pred_x_prev.feats.shape, generator=generator, device=pred_x_prev.device, dtype=pred_x_prev.dtype)
                    next_sample = pred_x_prev.replace(pred_x_prev.feats + eps * std[pred_x_prev.coords[:, 0]].unsqueeze(-1))
                else:
                    eps = torch.randn_like(pred_x_prev, generator=generator)
                    next_sample = pred_x_prev + eps * std.view(-1, *([1] * (pred_x_prev.ndim - 1)))
            else:
                next_sample = pred_x_prev
        log_prob = _sparse_gaussian_log_prob(next_sample, pred_x_prev, std[next_sample.coords[:, 0]].unsqueeze(-1)) if isinstance(next_sample, sp.SparseTensor) else None
        return edict(
            {
                "prev_sample": next_sample,
                "pred_x_prev_mean": pred_x_prev,
                "log_prob": log_prob,
                "std": std,
                "pred_v": pred_v,
            }
        )

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond=None,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 1.0,
        noise_level: float = 0.0,
        generator: Optional[torch.Generator] = None,
        verbose: bool = True,
        **kwargs,
    ):
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        ret = edict(
            {
                "samples": None,
                "latents": [],
                "next_latents": [],
                "timesteps": [],
                "log_probs": [],
                "step_means": [],
                "step_stds": [],
            }
        )
        for step_idx in range(steps):
            t = float(t_seq[step_idx])
            t_prev = float(t_seq[step_idx + 1])
            out = self.step_with_logprob(
                model,
                sample,
                t,
                t_prev,
                cond,
                neg_cond=neg_cond,
                cfg_strength=cfg_strength,
                noise_level=noise_level,
                generator=generator,
                **kwargs,
            )
            ret.latents.append(sample.detach())
            ret.next_latents.append(out.prev_sample.detach())
            ret.timesteps.append(t)
            ret.log_probs.append(out.log_prob.detach())
            ret.step_means.append(out.pred_x_prev_mean.detach())
            ret.step_stds.append(out.std.detach())
            sample = out.prev_sample
        ret.samples = sample
        ret.timesteps = torch.tensor(ret.timesteps, device=sample.device, dtype=torch.float32)
        ret.log_probs = torch.stack(ret.log_probs, dim=1)
        ret.step_stds = torch.stack(ret.step_stds, dim=1)
        return ret
