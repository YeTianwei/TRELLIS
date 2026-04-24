from __future__ import annotations

from typing import Dict, List

import torch

from ..modules import sparse as sp
from ..utils import render_utils


class TrellisStage2GRPORollout:
    def __init__(
        self,
        stage1_flow_model,
        stage1_decoder,
        slat_decoder_gs,
        image_encoder,
        stage1_sampler,
        stage2_sampler,
        slat_normalization: Dict[str, List[float]],
        train_cfg_strength: float,
        eval_cfg_strength: float,
        train_steps: int,
        eval_steps: int,
        train_noise_level: float,
        eval_noise_level: float,
        train_num_views: int,
        eval_num_views: int,
        train_render_resolution: int,
        eval_render_resolution: int,
        render_bg_color=(0, 0, 0),
        render_r: float = 2.0,
        render_fov: float = 40.0,
        return_render_info: bool = False,
        sigma_min: float = 1e-5,
    ):
        self.stage1_flow_model = stage1_flow_model
        self.stage1_decoder = stage1_decoder
        self.slat_decoder_gs = slat_decoder_gs
        self.image_encoder = image_encoder
        self.stage1_sampler = stage1_sampler
        self.stage2_sampler = stage2_sampler
        self.train_cfg_strength = train_cfg_strength
        self.eval_cfg_strength = eval_cfg_strength
        self.train_steps = train_steps
        self.eval_steps = eval_steps
        self.train_noise_level = train_noise_level
        self.eval_noise_level = eval_noise_level
        self.train_num_views = train_num_views
        self.eval_num_views = eval_num_views
        self.train_render_resolution = train_render_resolution
        self.eval_render_resolution = eval_render_resolution
        self.render_bg_color = render_bg_color
        self.render_r = render_r
        self.render_fov = render_fov
        self.return_render_info = return_render_info
        self.sigma_min = sigma_min
        self._slat_mean = torch.tensor(slat_normalization["mean"]).reshape(1, -1).cuda()
        self._slat_std = torch.tensor(slat_normalization["std"]).reshape(1, -1).cuda()

    @torch.no_grad()
    def encode_cond(self, cond_image: torch.Tensor):
        cond = self.image_encoder.encode_image(cond_image)
        return {"cond": cond, "neg_cond": torch.zeros_like(cond)}

    @torch.no_grad()
    def sample_sparse_structure(self, cond: Dict[str, torch.Tensor]):
        reso = self.stage1_flow_model.resolution
        noise = torch.randn(1, self.stage1_flow_model.in_channels, reso, reso, reso, device=self.stage1_flow_model.device)
        z_s = self.stage1_sampler.sample(
            self.stage1_flow_model,
            noise,
            cond=cond["cond"],
            neg_cond=cond["neg_cond"],
            cfg_strength=self.eval_cfg_strength,
            steps=self.eval_steps,
            verbose=False,
        ).samples
        coords = torch.argwhere(self.stage1_decoder(z_s) > 0)[:, [0, 2, 3, 4]].int()
        return coords

    def _coords_for_group(self, coords: torch.Tensor, group_size: int):
        coords_group = []
        for idx in range(group_size):
            group_coords = coords.clone()
            group_coords[:, 0] = idx
            coords_group.append(group_coords)
        return torch.cat(coords_group, dim=0)

    def _decode_slat(self, slat: sp.SparseTensor):
        batch_mean = self._slat_mean.expand(slat.shape[0], -1)
        batch_std = self._slat_std.expand(slat.shape[0], -1)
        slat = slat.replace(slat.feats * batch_std[slat.coords[:, 0]] + batch_mean[slat.coords[:, 0]])
        reps = self.slat_decoder_gs(slat)
        return reps, slat

    @torch.no_grad()
    def render_gaussians(self, gaussians, num_views: int, resolution: int, return_info: bool = False):
        cams = [render_utils.sphere_hammersley_sequence(i, num_views) for i in range(num_views)]
        yaws = [cam[0] for cam in cams]
        pitchs = [cam[1] for cam in cams]
        extrinsics, intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, self.render_r, self.render_fov)
        rendered = []
        alphas = []
        depths = []
        for gaussian in gaussians:
            frame_info = render_utils.render_frames(
                gaussian,
                extrinsics,
                intrinsics,
                options={"resolution": resolution, "bg_color": self.render_bg_color, "return_aux": return_info},
                verbose=False,
            )
            frames = frame_info["color"]
            frames = torch.stack([torch.tensor(frame).permute(2, 0, 1).float() / 255.0 for frame in frames], dim=0)
            rendered.append(frames.cuda())
            if return_info:
                alpha_frames = frame_info.get("alpha", [])
                depth_frames = frame_info.get("depth", [])
                if alpha_frames and all(frame is not None for frame in alpha_frames):
                    alphas.append(torch.stack([torch.tensor(frame).float() for frame in alpha_frames], dim=0).cuda())
                if depth_frames and all(frame is not None for frame in depth_frames):
                    depths.append(torch.stack([torch.tensor(frame).float() for frame in depth_frames], dim=0).cuda())
        renders = torch.stack(rendered, dim=0)
        info = {}
        if return_info:
            if len(alphas) == len(rendered):
                info["alpha"] = torch.stack(alphas, dim=0)
            if len(depths) == len(rendered):
                info["depth"] = torch.stack(depths, dim=0)
            info["extrinsics"] = torch.stack(extrinsics, dim=0)
            info["intrinsics"] = torch.stack(intrinsics, dim=0)
        return renders, info

    @torch.no_grad()
    def rollout_group(self, model, cond_image: torch.Tensor, group_size: int, train: bool = True):
        cond = self.encode_cond(cond_image.unsqueeze(0))
        coords = self.sample_sparse_structure(cond)
        batch_coords = self._coords_for_group(coords, group_size)
        noise = sp.SparseTensor(
            feats=torch.randn(batch_coords.shape[0], model.in_channels, device=batch_coords.device),
            coords=batch_coords,
        )
        cond_group = {
            "cond": cond["cond"].repeat(group_size, 1, 1),
            "neg_cond": cond["neg_cond"].repeat(group_size, 1, 1),
        }
        sampler_out = self.stage2_sampler.sample(
            model,
            noise,
            cond=cond_group["cond"],
            neg_cond=cond_group["neg_cond"],
            cfg_strength=self.train_cfg_strength if train else self.eval_cfg_strength,
            steps=self.train_steps if train else self.eval_steps,
            noise_level=self.train_noise_level if train else self.eval_noise_level,
            verbose=False,
        )
        gaussians, denorm_slat = self._decode_slat(sampler_out.samples)
        renders, render_info = self.render_gaussians(
            gaussians,
            num_views=self.train_num_views if train else self.eval_num_views,
            resolution=self.train_render_resolution if train else self.eval_render_resolution,
            return_info=self.return_render_info,
        )
        return {
            "coords": coords,
            "batch_coords": batch_coords,
            "cond": cond_group,
            "trajectory": sampler_out,
            "slat": denorm_slat,
            "gaussians": gaussians,
            "renders": renders,
            "render_info": render_info,
        }
