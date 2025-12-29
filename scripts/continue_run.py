import argparse
import os

from omegaconf import OmegaConf

from atlas.runners.dr_hrm_ppo_runner import DRHRMConditionedPPORunner
from atlas.runners.plr_hrm_ppo_runner import PLRHRMConditionedPPORunner
from atlas.utils.logging import download_artifacts


def get_config_dir(run_id: str, run_path: str) -> str:
    return os.path.join(run_path, run_id, "config")


def get_checkpoints_dir(run_id: str, run_path: str) -> str:
    return os.path.join(run_path, run_id, "checkpoints")


def get_continue_cfg(args):
    # Load the training config and make substitutions for evaluation
    cfg = OmegaConf.load(os.path.join(get_config_dir(args.run_id, args.run_path), "config.yaml"))

    # Checkpoint loading
    cfg.logging.run_id = args.run_id
    cfg.logging.checkpoint.training.loading.mode = "continue"
    cfg.logging.checkpoint.training.loading.path = os.path.abspath(
        get_checkpoints_dir(args.run_id, args.run_path)
    )
    cfg.logging.checkpoint.training.loading.step = args.step

    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--project", default="atlas")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--run_path", default="artifacts")
    parser.add_argument("--step", type=int, default=None)  # checkpoint step to load (last in folder by default)

    args = parser.parse_args()

    if args.download:
        download_artifacts(
            args.entity,
            args.project,
            args.run_id,
            get_checkpoints_dir(args.run_id, args.run_path),
            get_config_dir(args.run_id, args.run_path),
        )

    cfg = get_continue_cfg(args)
    if cfg.algorithm.name == "dr":
        runner = DRHRMConditionedPPORunner(cfg)
    elif cfg.algorithm.name == "plr":
        runner = PLRHRMConditionedPPORunner(cfg)

    runner.run()
