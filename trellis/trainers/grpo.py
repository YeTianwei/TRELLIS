from __future__ import annotations

import copy
import os
from contextlib import nullcontext
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms

from .. import models
from ..pipelines import samplers
from ..pipelines.rollout_stage2_grpo import TrellisStage2GRPORollout
from ..utils import dist_utils
from ..utils.data_utils import ResumableSampler, cycle, recursive_to_device
from ..utils.general_utils import dict_reduce
from ..utils.lora_utils import (
    LoraSpec,
    apply_lora,
    count_trainable_parameters,
    disable_lora,
    freeze_module,
    get_lora_trainable_parameters,
    load_lora_state_dict,
    lora_state_dict,
)
from ..utils.reward_evaluator import Stage2RewardEvaluator
from .base import Trainer


class FrozenImageConditionEncoder:
    def __init__(
        self,
        model_name: str,
        repo_or_dir: str = "facebookresearch/dinov2",
        source: str = "github",
        image_size: int = 518,
    ):
        self.model_name = model_name
        self.repo_or_dir = repo_or_dir
        self.source = source
        self.image_size = image_size
        self.model = None
        self.transform = transforms.Compose(
            [
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _init(self):
        if self.model is not None:
            return
        with dist_utils.local_master_first():
            try:
                self.model = torch.hub.load(
                    self.repo_or_dir,
                    self.model_name,
                    pretrained=True,
                    source=self.source,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to load the DINOv2 image-condition encoder via torch.hub. "
                    f"repo_or_dir={self.repo_or_dir!r}, model={self.model_name!r}, source={self.source!r}. "
                    "If GitHub is unstable, set image_cond_repo_or_dir to a local DINOv2 hub cache "
                    "such as '/home/timer/.cache/torch/hub/facebookresearch_dinov2_main' and "
                    "set image_cond_source='local'."
                ) from exc
        self.model = self.model.eval().cuda()
        freeze_module(self.model)

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        self._init()
        if image.shape[-2:] != (self.image_size, self.image_size):
            image = torch.nn.functional.interpolate(
                image,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        image = self.transform(image).cuda()
        features = self.model(image, is_training=True)["x_prenorm"]
        return torch.nn.functional.layer_norm(features, features.shape[-1:])


class ImageConditionedStage2GRPOTrainer(Trainer):
    def __init__(
        self,
        *args,
        image_cond_model: str = "dinov2_vitl14_reg",
        image_cond_repo_or_dir: str = "facebookresearch/dinov2",
        image_cond_source: str = "github",
        image_cond_size: int = 518,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        sparse_structure_flow_model_path: str = "",
        sparse_structure_decoder_path: str = "",
        slat_decoder_gs_path: str = "",
        slat_normalization: Optional[Dict[str, List[float]]] = None,
        stage1_sampler: Optional[dict] = None,
        group_size: int = 4,
        train_rollout_steps: int = 6,
        eval_rollout_steps: int = 12,
        train_noise_level: float = 0.8,
        eval_noise_level: float = 0.0,
        train_cfg_strength: float = 3.0,
        eval_cfg_strength: float = 3.0,
        clip_range: float = 1e-4,
        beta: float = 0.01,
        reward_image_align_weight: float = 0.7,
        reward_aesthetic_weight: float = 0.3,
        reward_topk_views: int = 2,
        reward_dino_batch_size: int = 1,
        reward_aesthetic_batch_size: int = 1,
        train_num_views: int = 4,
        eval_num_views: int = 12,
        train_render_resolution: int = 256,
        eval_render_resolution: int = 512,
        render_bg_color=(0, 0, 0),
        render_r: float = 2.0,
        render_fov: float = 40.0,
        sigma_min: float = 1e-5,
        log_reward_details: bool = True,
        enable_model_snapshot: bool = False,
        dataloader_num_workers: Optional[int] = None,
        dataloader_persistent_workers: Optional[bool] = None,
        **kwargs,
    ):
        self.image_cond_model_name = image_cond_model
        self.image_cond_repo_or_dir = image_cond_repo_or_dir
        self.image_cond_source = image_cond_source
        self.image_cond_size = image_cond_size
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.sparse_structure_flow_model_path = sparse_structure_flow_model_path
        self.sparse_structure_decoder_path = sparse_structure_decoder_path
        self.slat_decoder_gs_path = slat_decoder_gs_path
        self.slat_normalization = slat_normalization
        self.stage1_sampler_cfg = stage1_sampler or {"name": "FlowEulerCfgSampler", "args": {"sigma_min": 1e-5}}
        self.group_size = group_size
        self.train_rollout_steps = train_rollout_steps
        self.eval_rollout_steps = eval_rollout_steps
        self.train_noise_level = train_noise_level
        self.eval_noise_level = eval_noise_level
        self.train_cfg_strength = train_cfg_strength
        self.eval_cfg_strength = eval_cfg_strength
        self.clip_range = clip_range
        self.beta = beta
        self.reward_image_align_weight = reward_image_align_weight
        self.reward_aesthetic_weight = reward_aesthetic_weight
        self.reward_topk_views = reward_topk_views
        self.reward_dino_batch_size = reward_dino_batch_size
        self.reward_aesthetic_batch_size = reward_aesthetic_batch_size
        self.train_num_views = train_num_views
        self.eval_num_views = eval_num_views
        self.train_render_resolution = train_render_resolution
        self.eval_render_resolution = eval_render_resolution
        self.render_bg_color = render_bg_color
        self.render_r = render_r
        self.render_fov = render_fov
        self.sigma_min = sigma_min
        self.log_reward_details = log_reward_details
        self.enable_model_snapshot = enable_model_snapshot
        self.dataloader_num_workers = dataloader_num_workers
        self.dataloader_persistent_workers = dataloader_persistent_workers
        super().__init__(*args, **kwargs)

    def __str__(self):
        base = super().__str__()
        lines = [
            base,
            f"  - Group size: {self.group_size}",
            f"  - Train rollout steps: {self.train_rollout_steps}",
            f"  - Eval rollout steps: {self.eval_rollout_steps}",
            f"  - LoRA params: {count_trainable_parameters(self.models['denoiser'])}",
            f"  - Dataloader workers: {self.dataloader.num_workers}",
            f"  - Persistent workers: {self.dataloader.persistent_workers}",
        ]
        return "\n".join(lines)

    def prepare_dataloader(self, **kwargs):
        self.data_sampler = ResumableSampler(self.dataset, shuffle=True)
        default_num_workers = int(np.ceil(os.cpu_count() / torch.cuda.device_count()))
        num_workers = default_num_workers if self.dataloader_num_workers is None else self.dataloader_num_workers
        persistent_workers = num_workers > 0 if self.dataloader_persistent_workers is None else self.dataloader_persistent_workers
        if num_workers == 0:
            persistent_workers = False
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=persistent_workers,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, "collate_fn") else None,
            sampler=self.data_sampler,
        )
        self.data_iterator = cycle(self.dataloader)

    def init_models_and_more(self, **kwargs):
        denoiser = self.models["denoiser"].cuda()
        for param in denoiser.parameters():
            param.requires_grad_(False)

        target_suffixes = (
            "self_attn.to_qkv",
            "self_attn.to_out",
            "cross_attn.to_q",
            "cross_attn.to_kv",
            "cross_attn.to_out",
            "mlp.mlp.0",
            "mlp.mlp.2",
        )
        replaced = apply_lora(
            denoiser,
            LoraSpec(
                target_suffixes=target_suffixes,
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                dropout=self.lora_dropout,
            ),
        )
        if self.is_master:
            print("LoRA injected into:")
            for name in replaced:
                print(f"  - {name}")

        self.image_encoder = FrozenImageConditionEncoder(
            self.image_cond_model_name,
            repo_or_dir=self.image_cond_repo_or_dir,
            source=self.image_cond_source,
            image_size=self.image_cond_size,
        )
        self.reward_evaluator = Stage2RewardEvaluator(
            image_cond_model=self.image_cond_model_name,
            image_cond_repo_or_dir=self.image_cond_repo_or_dir,
            image_cond_source=self.image_cond_source,
            image_align_weight=self.reward_image_align_weight,
            aesthetic_weight=self.reward_aesthetic_weight,
            topk_views=self.reward_topk_views,
            dino_batch_size=self.reward_dino_batch_size,
            aesthetic_batch_size=self.reward_aesthetic_batch_size,
            dino_encoder=self.image_encoder,
        )

        self.stage1_flow_model = models.from_pretrained(self.sparse_structure_flow_model_path).cuda().eval()
        self.stage1_decoder = models.from_pretrained(self.sparse_structure_decoder_path).cuda().eval()
        self.slat_decoder_gs = models.from_pretrained(self.slat_decoder_gs_path).cuda().eval()
        freeze_module(self.stage1_flow_model)
        freeze_module(self.stage1_decoder)
        freeze_module(self.slat_decoder_gs)

        self.stage1_sampler = getattr(samplers, self.stage1_sampler_cfg["name"])(**self.stage1_sampler_cfg.get("args", {}))
        self.stage2_sampler = samplers.FlowEulerGRPOSampler(self.sigma_min)

        self.rollout = TrellisStage2GRPORollout(
            stage1_flow_model=self.stage1_flow_model,
            stage1_decoder=self.stage1_decoder,
            slat_decoder_gs=self.slat_decoder_gs,
            image_encoder=self.image_encoder,
            stage1_sampler=self.stage1_sampler,
            stage2_sampler=self.stage2_sampler,
            slat_normalization=self.slat_normalization,
            train_cfg_strength=self.train_cfg_strength,
            eval_cfg_strength=self.eval_cfg_strength,
            train_steps=self.train_rollout_steps,
            eval_steps=self.eval_rollout_steps,
            train_noise_level=self.train_noise_level,
            eval_noise_level=self.eval_noise_level,
            train_num_views=self.train_num_views,
            eval_num_views=self.eval_num_views,
            train_render_resolution=self.train_render_resolution,
            eval_render_resolution=self.eval_render_resolution,
            render_bg_color=self.render_bg_color,
            render_r=self.render_r,
            render_fov=self.render_fov,
            sigma_min=self.sigma_min,
        )

        if self.world_size > 1:
            self.training_models = {
                "denoiser": DDP(
                    denoiser,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=False,
                    bucket_cap_mb=128,
                )
            }
        else:
            self.training_models = {"denoiser": denoiser}

        self.model_params = get_lora_trainable_parameters(self.models["denoiser"])
        self.master_params = self.model_params
        if self.is_master:
            self.ema_params = []

        self.optimizer = getattr(torch.optim, self.optimizer_config["name"])(self.master_params, **self.optimizer_config["args"])
        if self.lr_scheduler_config is not None:
            self.lr_scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler_config["name"])(self.optimizer, **self.lr_scheduler_config["args"])

    def load(self, load_dir, step=0):
        if self.is_master:
            print(f"\nLoading LoRA checkpoint from step {step}...", end="")
        lora_path = os.path.join(load_dir, "ckpts", f"denoiser_lora_step{step:07d}.pt")
        state = torch.load(lora_path, map_location="cpu", weights_only=True)
        load_lora_state_dict(self.models["denoiser"], state)

        misc_path = os.path.join(load_dir, "ckpts", f"misc_step{step:07d}.pt")
        misc = torch.load(misc_path, map_location="cpu", weights_only=False)
        self.optimizer.load_state_dict(misc["optimizer"])
        self.step = misc["step"]
        self.data_sampler.load_state_dict(misc["data_sampler"])
        if self.lr_scheduler_config is not None and "lr_scheduler" in misc:
            self.lr_scheduler.load_state_dict(misc["lr_scheduler"])
        if self.world_size > 1:
            dist.barrier()
        if self.is_master:
            print(" Done.")

    def save(self):
        assert self.is_master, "save() should be called only on rank 0."
        print(f"\nSaving LoRA checkpoint at step {self.step}...", end="")
        torch.save(lora_state_dict(self.models["denoiser"]), os.path.join(self.output_dir, "ckpts", f"denoiser_lora_step{self.step:07d}.pt"))
        misc = {
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
            "data_sampler": self.data_sampler.state_dict(),
            "base": {
                "sparse_structure_flow_model_path": self.sparse_structure_flow_model_path,
                "sparse_structure_decoder_path": self.sparse_structure_decoder_path,
                "slat_decoder_gs_path": self.slat_decoder_gs_path,
                "image_cond_model": self.image_cond_model_name,
            },
        }
        if self.lr_scheduler_config is not None:
            misc["lr_scheduler"] = self.lr_scheduler.state_dict()
        torch.save(misc, os.path.join(self.output_dir, "ckpts", f"misc_step{self.step:07d}.pt"))
        print(" Done.")

    def finetune_from(self, finetune_ckpt):
        if "denoiser" not in finetune_ckpt:
            return
        state = torch.load(finetune_ckpt["denoiser"], map_location="cpu", weights_only=True)
        load_lora_state_dict(self.models["denoiser"], state)

    def update_ema(self):
        return

    def check_ddp(self):
        if self.world_size <= 1:
            return
        if self.is_master:
            print("\nPerforming DDP LoRA check...")
        for param in self.master_params:
            gathered = [torch.empty_like(param) for _ in range(self.world_size)]
            dist.all_gather(gathered, param.detach())
            assert all(torch.equal(gathered[0], tensor) for tensor in gathered[1:]), "LoRA params are not synchronized"
        if self.is_master:
            print("Done.")

    def training_losses(self, **mb_data):
        raise NotImplementedError("GRPO trainer implements a custom run_step.")

    @staticmethod
    def _advantage(rewards: torch.Tensor) -> torch.Tensor:
        std = rewards.std(unbiased=False)
        if torch.isclose(std, torch.tensor(0.0, device=rewards.device, dtype=rewards.dtype)):
            return torch.zeros_like(rewards)
        return (rewards - rewards.mean()) / (std + 1e-6)

    def _t_prev(self, timesteps: torch.Tensor, j: int) -> float:
        if j == timesteps.shape[0] - 1:
            return 0.0
        return float(timesteps[j + 1].item())

    def _unwrap_training_model(self):
        model = self.training_models["denoiser"]
        return model.module if isinstance(model, DDP) else model

    @torch.no_grad()
    def snapshot(self, *args, **kwargs):
        if not self.enable_model_snapshot:
            if self.is_master:
                print("\nSkipping GRPO model snapshot. Set enable_model_snapshot=true to render rollout previews.")
            return
        return super().snapshot(*args, **kwargs)

    @torch.no_grad()
    def run_snapshot(self, num_samples: int, batch_size: int = 1, verbose: bool = False):
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, "collate_fn") else None,
        )
        samples = []
        conds = []
        rewards = []
        for batch in dataloader:
            batch = recursive_to_device(batch, self.device)
            for cond in batch["cond"]:
                rollout = self.rollout.rollout_group(self._unwrap_training_model(), cond, group_size=1, train=False)
                score = self.reward_evaluator.score(cond.unsqueeze(0), rollout["renders"])
                samples.append(rollout["renders"][0, 0])
                conds.append(cond)
                rewards.append(score["reward"][0:1])
                if len(samples) >= num_samples:
                    return {
                        "sample": {"value": torch.stack(samples), "type": "image"},
                        "cond": {"value": torch.stack(conds), "type": "image"},
                        "reward": {"value": torch.stack(rewards).reshape(-1, 1, 1, 1), "type": "number"},
                    }
        return {
            "sample": {"value": torch.stack(samples), "type": "image"},
            "cond": {"value": torch.stack(conds), "type": "image"},
            "reward": {"value": torch.stack(rewards).reshape(-1, 1, 1, 1), "type": "number"},
        }

    def run_step(self, data_list):
        amp_context = torch.autocast(device_type="cuda") if self.fp16_mode == "amp" else nullcontext()
        stats = []
        train_model = self.training_models["denoiser"]
        self.optimizer.zero_grad()

        for data in data_list:
            data = recursive_to_device(data, self.device, non_blocking=True)
            batch_stats = []
            for cond in data["cond"]:
                with torch.no_grad():
                    rollout = self.rollout.rollout_group(self._unwrap_training_model(), cond, group_size=self.group_size, train=True)
                    reward_info = self.reward_evaluator.score(cond.unsqueeze(0).repeat(self.group_size, 1, 1, 1), rollout["renders"])
                    advantages = self._advantage(reward_info["reward"])

                if advantages.abs().sum() == 0:
                    if self.log_reward_details:
                        batch_stats.append(
                            {
                                "reward": reward_info["reward"].mean().item(),
                                "image_align": reward_info["image_align"].mean().item(),
                                "aesthetic": reward_info["aesthetic"].mean().item(),
                                "skipped_zero_adv": 1.0,
                            }
                        )
                    continue

                policy_terms = []
                kl_terms = []
                clipfracs = []
                for j in range(self.train_rollout_steps):
                    with amp_context:
                        step_out = self.stage2_sampler.step_with_logprob(
                            train_model,
                            rollout["trajectory"].latents[j],
                            float(rollout["trajectory"].timesteps[j].item()),
                            self._t_prev(rollout["trajectory"].timesteps, j),
                            rollout["cond"]["cond"],
                            neg_cond=rollout["cond"]["neg_cond"],
                            cfg_strength=self.train_cfg_strength,
                            noise_level=self.train_noise_level,
                            next_sample=rollout["trajectory"].next_latents[j],
                        )
                        with torch.no_grad():
                            with disable_lora(self._unwrap_training_model()):
                                ref_out = self.stage2_sampler.step_with_logprob(
                                    train_model,
                                    rollout["trajectory"].latents[j],
                                    float(rollout["trajectory"].timesteps[j].item()),
                                    self._t_prev(rollout["trajectory"].timesteps, j),
                                    rollout["cond"]["cond"],
                                    neg_cond=rollout["cond"]["neg_cond"],
                                    cfg_strength=self.train_cfg_strength,
                                    noise_level=self.train_noise_level,
                                    next_sample=rollout["trajectory"].next_latents[j],
                                )

                        old_log_prob = rollout["trajectory"].log_probs[:, j]
                        ratio = torch.exp(step_out.log_prob - old_log_prob)
                        unclipped = -advantages * ratio
                        clipped = -advantages * torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                        policy_loss = torch.maximum(unclipped, clipped).mean()
                        std = step_out.std.view(-1, 1, 1)
                        mean_diff = step_out.pred_x_prev_mean.feats - ref_out.pred_x_prev_mean.feats
                        kl_loss = (mean_diff.square() / (2 * std[step_out.pred_x_prev_mean.coords[:, 0]].square().clamp_min(1e-8))).mean()
                        policy_terms.append(policy_loss)
                        kl_terms.append(kl_loss)
                        clipfracs.append((torch.abs(ratio - 1.0) > self.clip_range).float().mean())

                loss = torch.stack(policy_terms).mean() + self.beta * torch.stack(kl_terms).mean()
                loss.backward()

                batch_stats.append(
                    {
                        "reward": reward_info["reward"].mean().item(),
                        "image_align": reward_info["image_align"].mean().item(),
                        "aesthetic": reward_info["aesthetic"].mean().item(),
                        "adv_abs": advantages.abs().mean().item(),
                        "policy_loss": torch.stack(policy_terms).mean().item(),
                        "kl_loss": torch.stack(kl_terms).mean().item(),
                        "clipfrac": torch.stack(clipfracs).mean().item(),
                    }
                )

            if batch_stats:
                stats.extend(batch_stats)

        if not stats:
            return {"loss": {"loss": 0.0}, "status": {"skipped": 1.0}}

        grad_norm = torch.nn.utils.clip_grad_norm_(self.master_params, self.grad_clip) if self.grad_clip is not None else torch.tensor(0.0)
        self.optimizer.step()
        if self.lr_scheduler_config is not None:
            self.lr_scheduler.step()
        reduced = dict_reduce(stats, lambda x: float(np.mean(x)))
        reduced.setdefault("status", {})
        reduced["status"]["grad_norm"] = float(grad_norm)
        reduced["loss"] = {
            "loss": reduced.get("policy_loss", 0.0) + self.beta * reduced.get("kl_loss", 0.0),
            "policy_loss": reduced.get("policy_loss", 0.0),
            "kl_loss": reduced.get("kl_loss", 0.0),
        }
        return reduced
