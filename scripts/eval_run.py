import argparse
from datetime import datetime
import json
import os
import shutil
from typing import Optional

from omegaconf import OmegaConf
import yaml

from atlas.runners.dr_hrm_ppo_runner import DRHRMConditionedPPORunner
from atlas.runners.plr_hrm_ppo_runner import PLRHRMConditionedPPORunner
from atlas.utils.logging import download_artifacts


def get_config_dir(run_id: str, run_path: str) -> str:
    return os.path.join(run_path, run_id, "config")


def get_checkpoints_dir(run_id: str, run_path: str) -> str:
    return os.path.join(run_path, run_id, "checkpoints")


def get_eval_file_cfg(args):
    file_cfg = []

    for eval_file_path in args.eval_file_paths:
        hrms_path = os.path.join(eval_file_path, "hrms")
        levels_path = os.path.join(eval_file_path, "levels")

        hrm_files = [
            os.path.join(hrms_path, h)
            for h in sorted(os.listdir(hrms_path))
            if h.endswith(".yaml")
        ]

        level_files = [
            os.path.join(levels_path, l)
            for l in sorted(os.listdir(levels_path))
            if l.endswith(".yaml")
        ]

        for l, h in zip(level_files, hrm_files):
            file_cfg.append({
                "hrm_path": h,
                "level_path": l,
                "name": os.path.basename(h)[:-len(".yaml")]
            })

    return file_cfg


def get_eval_loading_cfg(args, hrm_sampler_cfg):
    hrm_args = {
        "max_num_rms": hrm_sampler_cfg.max_num_rms,
        "max_num_states": hrm_sampler_cfg.max_num_states,
        "max_num_edges": hrm_sampler_cfg.max_num_edges,
        "max_num_literals": hrm_sampler_cfg.max_num_literals,
    }
    hrm_args.update(args.hrm_args)

    if args.eval_file_cfg:
        with open(args.eval_file_cfg) as f:
            file_cfg = yaml.safe_load(f)
    else:
        file_cfg = get_eval_file_cfg(args)

    return OmegaConf.create({
        "_target_": "atlas.eval_loaders.file.FileEvaluationSetLoader",
        "files": file_cfg,
        "level_loading_fn": {
            "_partial_": True,
            "_target_": "atlas.envs.xminigrid.level.XMinigridLevel.from_file"
        },
        **hrm_args,
    })


def get_eval_cfg(args, dst_run_id: str, step: int):
    # Load the training config and make substitutions for evaluation
    cfg = OmegaConf.load(
        os.path.join(get_config_dir(args.run_id, args.run_path), "config.yaml")
    )

    cfg.mode = "evaluation"
    cfg.seed = args.seed

    cfg.evaluation.num_rollouts_per_problem = args.num_rollouts
    cfg.evaluation.loader = get_eval_loading_cfg(args, cfg.problem_sampler.hrm_sampler)

    cfg.env_params.update(args.env_params)

    if cfg.algorithm.name == "plr":
        cfg.algorithm.num_high_score_eval_problems = 0
        cfg.algorithm.num_low_score_eval_problems = 0

    cfg.logging.run_id = dst_run_id
    cfg.logging.run_name = args.eval_run_name
    cfg.logging.wandb.group = args.group
    cfg.logging.checkpoint.evaluation.path = os.path.abspath(
        get_checkpoints_dir(args.run_id, args.run_path)
    )
    cfg.logging.checkpoint.evaluation.step = step
    cfg.logging.rollout.num_to_visualize_per_problem = args.num_rendered_rollouts
    cfg.logging.rollout.visualization_length = args.num_rendered_steps

    # Transforming old configuration for interleaved mutations to adapt to the new one
    if cfg.algorithm.name == "plr":
        mutator_cfg = cfg.algorithm.mutator
        if mutator_cfg._target_.endswith("InterleavedLevelHRMMutator") and "use_hindsight" in cfg.algorithm.mutator:
            cfg.algorithm.mutator = OmegaConf.create(dict(
                min_edits=mutator_cfg.min_edits,
                max_edits=mutator_cfg.max_edits,
                level_cfg=dict(
                    enabled=True,
                    use_add_rm_objs=mutator_cfg.use_add_rm_objs,
                    use_add_rm_rooms=mutator_cfg.use_add_rm_rooms,
                    use_move_agent=mutator_cfg.use_move_agent,
                ),
                hrm_cfg=dict(
                    enabled=True,
                    use_add_rm_transitions=mutator_cfg.use_add_rm_transitions
                ),
                hindsight_cfg=dict(enabled=mutator_cfg.use_hindsight),
                use_sparse_reward=mutator_cfg.use_sparse_reward,
                max_num_args=mutator_cfg.max_num_args,
                _target_=mutator_cfg._target_,
            ))

    return cfg


def _run_eval(args, dst_run_id: str, step: Optional[int]):
    if args.download:
        download_artifacts(
            args.entity,
            args.project,
            args.run_id,
            get_checkpoints_dir(args.run_id, args.run_path),
            get_config_dir(args.run_id, args.run_path),
            step
        )

    eval_cfg = get_eval_cfg(args, dst_run_id, step)
    if eval_cfg.algorithm.name == "dr":
        runner = DRHRMConditionedPPORunner(eval_cfg)
    elif eval_cfg.algorithm.name == "plr":
        runner = PLRHRMConditionedPPORunner(eval_cfg)

    runner.run()

    if args.download:
        shutil.rmtree(get_checkpoints_dir(args.run_id, args.run_path))
        shutil.rmtree(get_config_dir(args.run_id, args.run_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--eval_run_name", required=True)
    parser.add_argument("--project", default="atlas")
    parser.add_argument("--eval_file_paths", nargs='+')
    parser.add_argument("--eval_file_cfg")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--run_path", default="artifacts")
    parser.add_argument("--steps", type=int, nargs='+', default=None)
    parser.add_argument("--num_rollouts", type=int, default=1)
    parser.add_argument("--num_rendered_rollouts", type=int, default=0)
    parser.add_argument("--num_rendered_steps", type=int, default=128)
    parser.add_argument("--env_params", type=json.loads, default=dict())
    parser.add_argument("--hrm_args", type=json.loads, default=dict())
    parser.add_argument("--group", default=None)

    args = parser.parse_args()

    if args.eval_file_paths is None and args.eval_file_cfg is None:
        raise RuntimeError(
            "Error: Either a folder containing levels and HRMs OR a file containing the level-HRM pairs"
            "must be provided"
        )

    dst_run_id = f"{args.eval_run_name}--{datetime.now().strftime('%Y%m%d_%H%M')}"
    if args.steps is not None:
        steps = sorted(args.steps)
        for step in steps:
            _run_eval(args, dst_run_id, step)
    else:
        _run_eval(args, dst_run_id, args.steps)
