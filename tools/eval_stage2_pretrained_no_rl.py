import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path


HF_MODEL_ROOT = "microsoft/TRELLIS-image-large/ckpts"
LOCAL_DINO_CACHE = "/home/timer/.cache/torch/hub/facebookresearch_dinov2_main"
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def setup_rng(seed: int, torch):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def resolve_missing_local_paths(config: dict):
    """Prefer local paths from the run config, but fall back to portable defaults."""
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


def mean_item(values):
    import torch

    return torch.cat([v.detach().reshape(-1).float().cpu() for v in values]).mean().item()


def main():
    parser = argparse.ArgumentParser(description="Evaluate pretrained stage-2 model without GRPO updates.")
    parser.add_argument("--config", default="outputs/rl_stage2_abo_2kstep_tuned/config.json")
    parser.add_argument("--data_dir", default="datasets/ABO")
    parser.add_argument("--output_dir", default="outputs/rl_stage2_abo_pretrained_no_rl_baseline")
    parser.add_argument("--num_steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--group_size", type=int, default=None)
    parser.add_argument("--batch_size_per_gpu", type=int, default=1)
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--i_log", type=int, default=1)
    args = parser.parse_args()

    import torch

    from trellis import datasets, models, trainers
    from trellis.utils.data_utils import recursive_to_device
    from trellis.utils.lora_utils import disable_lora

    setup_rng(args.seed, torch)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads(Path(args.config).read_text())
    resolve_missing_local_paths(config)

    trainer_args = config["trainer"]["args"]
    trainer_args["max_steps"] = args.num_steps
    trainer_args["batch_size_per_gpu"] = args.batch_size_per_gpu
    trainer_args["batch_split"] = 1
    trainer_args["i_log"] = args.i_log
    trainer_args["i_sample"] = args.num_steps + 1
    trainer_args["i_save"] = args.num_steps + 1
    trainer_args["enable_model_snapshot"] = False
    if args.group_size is not None:
        trainer_args["group_size"] = args.group_size

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
    group_size = trainer.group_size
    train_mode = args.mode == "train"
    elapsed = 0.0
    all_rewards = []
    all_image_align = []
    all_aesthetic = []

    log_path = output_dir / "log.txt"
    if log_path.exists():
        log_path.unlink()

    print(
        f"Evaluating pretrained no-RL baseline for {args.num_steps} steps "
        f"(group_size={group_size}, mode={args.mode})."
    )

    for step in range(1, args.num_steps + 1):
        start = time.time()
        data_list = trainer.load_data()

        rewards = []
        image_aligns = []
        aesthetics = []
        with torch.no_grad(), disable_lora(model):
            for data in data_list:
                data = recursive_to_device(data, trainer.device, non_blocking=True)
                for cond in data["cond"]:
                    rollout = trainer.rollout.rollout_group(model, cond, group_size=group_size, train=train_mode)
                    score = trainer.reward_evaluator.score(
                        cond.unsqueeze(0).repeat(group_size, 1, 1, 1),
                        rollout["renders"],
                    )
                    rewards.append(score["reward"])
                    image_aligns.append(score["image_align"])
                    aesthetics.append(score["aesthetic"])

        elapsed += time.time() - start
        reward = mean_item(rewards)
        image_align = mean_item(image_aligns)
        aesthetic = mean_item(aesthetics)
        all_rewards.append(reward)
        all_image_align.append(image_align)
        all_aesthetic.append(aesthetic)

        row = {
            "time": {"step": time.time() - start, "elapsed": elapsed},
            "reward": reward,
            "image_align": image_align,
            "aesthetic": aesthetic,
            "no_rl_baseline": 1.0,
            "status": {"grad_norm": 0.0},
            "loss": {"loss": 0.0, "policy_loss": 0.0, "kl_loss": 0.0},
            "group_size": group_size,
            "mode": args.mode,
        }
        with log_path.open("a") as fp:
            fp.write(f"{step}: {json.dumps(row)}\n")

        if step % max(args.i_log, 1) == 0:
            print(
                f"Step {step}/{args.num_steps} | reward {reward:.4f} | "
                f"image_align {image_align:.4f} | elapsed {elapsed / 3600:.2f}h",
                flush=True,
            )

    summary = {
        "num_steps": args.num_steps,
        "group_size": group_size,
        "mode": args.mode,
        "reward_mean": statistics.fmean(all_rewards),
        "reward_min": min(all_rewards),
        "reward_max": max(all_rewards),
        "image_align_mean": statistics.fmean(all_image_align),
        "aesthetic_mean": statistics.fmean(all_aesthetic),
        "elapsed_hours": elapsed / 3600,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=4) + "\n")
    print(json.dumps(summary, indent=4))


if __name__ == "__main__":
    main()
