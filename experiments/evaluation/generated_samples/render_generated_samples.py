import os

import jax

from atlas.envs.xminigrid.labeling_function import XMinigridLabelingFunction
from atlas.envs.xminigrid.types import XMinigridEnvParams
from atlas.hrm.visualization import render_to_file
from atlas.envs.xminigrid.renderer import XMinigridRenderer
from atlas.envs.xminigrid.level import XMinigridLevel
from atlas.hrm.types import HRM
from atlas.utils.logging import download_artifacts
from atlas.utils.checkpointing import load_runner_state

ENTITY = None  # TODO: replace by W&B entity
RUNS = {  # TODO: replace None by W&B run ids
    "seq-plr-i": None,
    "seq-plr-c": None,
    "seq-accel_full-i": None,
    "seq-accel_full-c": None,
    "seq-accel_scratch-i": None,
    "seq-accel_scratch-c": None,
    "rw-plr-i": None,
    "rw-plr-c": None
}

NUM_SAMPLES = 10

ENV_PARAMS = XMinigridEnvParams()
RENDERER = XMinigridRenderer()
LABEL_FN = XMinigridLabelingFunction(ENV_PARAMS)
ALPHABET = LABEL_FN.get_str_alphabet()

if __name__ == "__main__":
    for algorithm, run_id in RUNS.items():
        if algorithm.startswith("seq"):
            steps = [100, *range(500, 2001, 500)]
        else:
            steps = [100, *range(500, 3001, 500)]

        for step in steps:
            checkpoint_dir = os.path.abspath(f"artifacts/{run_id}/checkpoints")
            config_dir = os.path.abspath(f"artifacts/{run_id}/config")
            download_artifacts(ENTITY, run_id, checkpoint_dir, config_dir, step)
            runner_state = load_runner_state(checkpoint_dir, step=step)

            if algorithm.startswith("seq-plr") or algorithm.startswith("rw-plr"):  # take some DR generated samples
                hrms = HRM(**runner_state["dr_last_hrm_batch"])
                levels = XMinigridLevel(**runner_state["dr_last_level_batch"])
                for i in range(NUM_SAMPLES):
                    level, hrm = jax.tree_util.tree_map(lambda x: x[i], (levels, hrms))
                    RENDERER.render_level(level).save(f"dr-{algorithm}-{step}-{i}-lvl.png")
                    render_to_file(hrm, f"dr-{algorithm}-{step}-{i}-hrm.png", alphabet=ALPHABET)

            hrms = HRM(**runner_state["replay_last_hrm_batch"])
            levels = XMinigridLevel(**runner_state["replay_last_level_batch"])
            for i in range(NUM_SAMPLES):
                level, hrm = jax.tree_util.tree_map(lambda x: x[i], (levels, hrms))
                RENDERER.render_level(level).save(f"{algorithm}-{step}-{i}-lvl.png")
                render_to_file(hrm, f"{algorithm}-{step}-{i}-hrm.png", alphabet=ALPHABET)
