import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HF_MODEL_ROOT = "microsoft/TRELLIS-image-large/ckpts"
LOCAL_DINO_CACHE = "/home/timer/.cache/torch/hub/facebookresearch_dinov2_main"


def setup_rng(seed: int, torch):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def resolve_missing_local_paths(config: dict):
    trainer_args = config["trainer"]["args"]

    dino_path = trainer_args.get("image_cond_repo_or_dir")
    if dino_path and not os.path.exists(dino_path) and os.path.exists(LOCAL_DINO_CACHE):
        trainer_args["image_cond_repo_or_dir"] = LOCAL_DINO_CACHE
        trainer_args["image_cond_source"] = "local"

    model_fallbacks = {
        "stage2_denoiser_path": "slat_flow_img_dit_L_64l8p2_fp16",
        "sparse_structure_flow_model_path": "ss_flow_img_dit_L_16l8_fp16",
        "sparse_structure_decoder_path": "ss_dec_conv3d_16l8_fp16",
        "slat_decoder_gs_path": "slat_dec_gs_swin8_B_64l8gs32_fp16",
    }
    for key, model_name in model_fallbacks.items():
        path = trainer_args.get(key)
        if path and not os.path.exists(path):
            trainer_args[key] = f"{HF_MODEL_ROOT}/{model_name}"


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    resolve_missing_local_paths(config)
    return config


def load_lora(model, ckpt_path: Path):
    from trellis.utils.lora_utils import load_lora_state_dict

    import torch

    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    missing = load_lora_state_dict(model, state)
    if missing:
        raise RuntimeError(f"Missing LoRA keys for {len(missing)} modules in {ckpt_path}")


def to_pil_image(tensor):
    from PIL import Image

    array = (tensor.detach().float().clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
    return Image.fromarray(array)


def make_panel(sample_results, output_path: Path, view_index: int = 0):
    from PIL import Image, ImageDraw

    label_h = 34
    pad = 8
    images = []
    labels = []
    for result in sample_results:
        if result["kind"] == "cond":
            img = to_pil_image(result["image"])
        else:
            img = to_pil_image(result["renders"][view_index])
        images.append(img)
        labels.append(result["label"])

    w = max(img.width for img in images)
    h = max(img.height for img in images)
    canvas = Image.new("RGB", (len(images) * w + (len(images) + 1) * pad, h + label_h + 2 * pad), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (img, label) in enumerate(zip(images, labels)):
        x = pad + i * (w + pad)
        y = pad + label_h
        canvas.paste(img.resize((w, h)), (x, y))
        draw.text((x, pad), label, fill=(0, 0, 0))
    canvas.save(output_path)


def method_label(method):
    if method["name"] == "pretrained":
        return "pretrained"
    return f"{method['name']}@{method['step']}"


def main():
    parser = argparse.ArgumentParser(description="Render fixed visual comparisons for pretrained vs GRPO LoRA checkpoints.")
    parser.add_argument("--config", default="outputs/rl_stage2_abo_2kstep_tuned/config.json")
    parser.add_argument("--data_dir", default="datasets/ABO")
    parser.add_argument("--output_dir", default="outputs/rl_stage2_abo_visual_compare")
    parser.add_argument("--checkpoint_dir", default="outputs/rl_stage2_abo_2kstep_tuned")
    parser.add_argument("--ckpt_steps", type=int, nargs="+", default=[1500, 1800, 1900, 2000])
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--num_variants", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=["train", "eval"], default="eval")
    parser.add_argument("--view_index", type=int, default=0)
    args = parser.parse_args()

    import torch

    from trellis import datasets, models, trainers
    from trellis.utils.data_utils import recursive_to_device
    from trellis.utils.lora_utils import disable_lora

    setup_rng(args.seed, torch)

    output_dir = Path(args.output_dir)
    panel_dir = output_dir / "panels"
    render_dir = output_dir / "renders"
    panel_dir.mkdir(parents=True, exist_ok=True)
    render_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(Path(args.config))
    trainer_args = config["trainer"]["args"]
    trainer_args["max_steps"] = 1
    trainer_args["batch_size_per_gpu"] = 1
    trainer_args["batch_split"] = 1
    trainer_args["i_sample"] = 2
    trainer_args["i_save"] = 2
    trainer_args["enable_model_snapshot"] = False
    trainer_args["dataloader_num_workers"] = 0
    trainer_args["dataloader_persistent_workers"] = False
    trainer_args["prefetch_data"] = False

    command = "python " + " ".join(sys.argv)
    (output_dir / "command.txt").write_text(command + "\n")
    (output_dir / "config.json").write_text(json.dumps(config, indent=4) + "\n")

    dataset = getattr(datasets, config["dataset"]["name"])(args.data_dir, **config["dataset"]["args"])
    model_dict = {
        name: getattr(models, model_cfg["name"])(**model_cfg["args"]).cuda()
        for name, model_cfg in config["models"].items()
    }
    trainer = getattr(trainers, config["trainer"]["name"])(
        model_dict,
        dataset,
        **trainer_args,
        output_dir=str(output_dir),
        load_dir=None,
        step=None,
    )
    model = trainer._unwrap_training_model()
    train_mode = args.mode == "train"

    methods = [{"name": "pretrained", "step": None, "path": None}]
    for step in args.ckpt_steps:
        methods.append(
            {
                "name": Path(args.checkpoint_dir).name,
                "step": step,
                "path": Path(args.checkpoint_dir) / "ckpts" / f"denoiser_lora_step{step:07d}.pt",
            }
        )

    records = []
    start_time = time.time()
    sample_count = min(args.num_samples, len(dataset))

    for sample_idx in range(sample_count):
        setup_rng(args.seed + sample_idx, torch)
        sample = recursive_to_device(dataset[sample_idx], trainer.device)
        cond = sample["cond"]
        sample_results = [{"kind": "cond", "label": f"cond {sample_idx}", "image": cond}]

        for method_idx, method in enumerate(methods):
            label = method_label(method)
            if method["path"] is not None:
                if not method["path"].exists():
                    raise FileNotFoundError(method["path"])
                load_lora(model, method["path"])

            rollout_seed = args.seed * 100000 + sample_idx * 1000
            setup_rng(rollout_seed, torch)
            with torch.no_grad():
                if method["name"] == "pretrained":
                    with disable_lora(model):
                        rollout = trainer.rollout.rollout_group(
                            model, cond, group_size=args.num_variants, train=train_mode
                        )
                else:
                    rollout = trainer.rollout.rollout_group(model, cond, group_size=args.num_variants, train=train_mode)

                score = trainer.reward_evaluator.score(
                    cond.unsqueeze(0).repeat(args.num_variants, 1, 1, 1),
                    rollout["renders"],
                )

            rewards = score["reward"].detach().float().cpu()
            best_idx = int(torch.argmax(rewards).item())
            renders = rollout["renders"][best_idx].detach().cpu()
            reward = float(rewards[best_idx].item())
            image_align = float(score["image_align"][best_idx].detach().float().cpu().item())
            aesthetic = float(score["aesthetic"][best_idx].detach().float().cpu().item())

            method_dir = render_dir / f"sample{sample_idx:03d}" / label.replace("/", "_")
            method_dir.mkdir(parents=True, exist_ok=True)
            for view_idx, view in enumerate(renders):
                to_pil_image(view).save(method_dir / f"view{view_idx:02d}.jpg")

            sample_results.append(
                {
                    "kind": "render",
                    "label": f"{label} r={reward:.3f}",
                    "renders": renders,
                }
            )
            records.append(
                {
                    "sample_idx": sample_idx,
                    "instance": sample.get("instance", ""),
                    "method": label,
                    "step": method["step"],
                    "best_variant": best_idx,
                    "reward": reward,
                    "image_align": image_align,
                    "aesthetic": aesthetic,
                    "variant_rewards": [float(v) for v in rewards.tolist()],
                }
            )

        make_panel(sample_results, panel_dir / f"sample{sample_idx:03d}.jpg", view_index=args.view_index)
        print(f"Rendered sample {sample_idx + 1}/{sample_count}", flush=True)

    by_method = {}
    for record in records:
        by_method.setdefault(record["method"], []).append(record["reward"])

    summary = {
        "num_samples": sample_count,
        "num_variants": args.num_variants,
        "mode": args.mode,
        "elapsed_hours": (time.time() - start_time) / 3600,
        "methods": {
            method: {
                "reward_mean": statistics.fmean(values),
                "reward_min": min(values),
                "reward_max": max(values),
            }
            for method, values in by_method.items()
        },
    }

    (output_dir / "records.json").write_text(json.dumps(records, indent=4) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=4) + "\n")
    print(json.dumps(summary, indent=4))


if __name__ == "__main__":
    main()
