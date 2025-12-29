import argparse
import os
import tempfile

import jax
from tqdm import tqdm

from atlas.envs.xminigrid.level import XMinigridLevel
from atlas.envs.xminigrid.labeling_function import XMinigridLabelingFunction
from atlas.envs.xminigrid.types import XMinigridEnvParams
from atlas.envs.xminigrid.validation import is_solvable_problem
from atlas.hrm.types import HRM
from atlas.utils.logging import download_artifacts
from atlas.utils.checkpointing import load_runner_state

CHECKPOINT_DIR = os.path.abspath("checkpoints")
CONFIG_DIR = os.path.abspath("config")

ENV_PARAMS = XMinigridEnvParams(height=13, width=19)
LABEL_FN = XMinigridLabelingFunction(ENV_PARAMS)
ALPHABET = LABEL_FN.get_str_alphabet()


def get_num_valid(runner_state):
    buffer_size = runner_state["buffer"]["size"]
    levels = XMinigridLevel(**runner_state["buffer"]["problems"][0])
    hrms = HRM(**runner_state["buffer"]["problems"][1])

    num_valid = 0
    for i in tqdm(range(buffer_size), mininterval=1800):
        level, hrm = jax.tree_util.tree_map(lambda x: x[i], (levels, hrms))

        try:
            if is_solvable_problem(level, hrm, ALPHABET):
                num_valid += 1
        except IndexError as e:
            # Some random walk instances are ill-formed,
            # so we ignore them by now (considering them invalid)
            pass

    return num_valid, buffer_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", required=True)
    parser.add_argument("--algorithm", required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--project", default="atlas")

    args = parser.parse_args()

    with open(f"validity_over_time_{args.algorithm}_{args.run_id}_{args.step}.csv", 'w') as f:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            checkpoint_dir = os.path.join(tmp_dir_name, "checkpoint")
            config_dir = os.path.join(tmp_dir_name, "config")
            download_artifacts(args.entity, args.project, args.run_id, checkpoint_dir, config_dir, step=args.step)
            runner_state = load_runner_state(checkpoint_dir, step=args.step)
            num_valid, num_total = get_num_valid(runner_state)
            f.write(f"{args.algorithm};{args.run_id};{args.step};{num_valid};{num_total};{num_valid / num_total}\n")
        # os.system("wandb artifact cache cleanup --remove-temp 500MB")
