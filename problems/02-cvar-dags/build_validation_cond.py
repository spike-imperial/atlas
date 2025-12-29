import math
import os

import jax

from atlas.envs.xminigrid.level_sampling.meta import XMinigridMetaLevelSampler
from atlas.envs.xminigrid.labeling_function import XMinigridLabelingFunction
from atlas.envs.xminigrid.level_sampling.single_room import XMinigridSingleRoomLevelSampler
from atlas.envs.xminigrid.level_sampling.two_rooms import XMinigridTwoRoomsLevelSampler
from atlas.envs.xminigrid.level_sampling.four_rooms import XMinigridFourRoomsLevelSampler
from atlas.envs.xminigrid.level_sampling.six_rooms import XMinigridSixRoomsLevelSampler
from atlas.envs.xminigrid.problem_sampling.level_conditioned import XMinigridLevelConditionedProblemSampler
from atlas.envs.xminigrid.renderer import XMinigridRenderer
from atlas.envs.xminigrid.types import XMinigridEnvParams
from atlas.envs.xminigrid.validation import is_solvable_problem
from atlas.hrm.ops import dump
from atlas.hrm.sampling.random_walk import RandomWalkHRMSampler
from atlas.hrm.visualization import render_to_file

HRM_PATH = os.path.join(os.path.abspath("."), "hrms")
LEVEL_PATH = os.path.join(os.path.abspath("."), "levels")

NUM_PROBLEMS_PER_TYPE = 10000


def build():
    env_params = XMinigridEnvParams(width=19, height=13)
    renderer = XMinigridRenderer()
    label_fn = XMinigridLabelingFunction(env_params)
    alphabet = label_fn.get_str_alphabet()

    hrm_sampler = RandomWalkHRMSampler(
        max_num_rms=1,
        max_num_states=6,
        max_num_edges=1,
        max_num_literals=5,
        alphabet_size=label_fn.get_alphabet_size(),
        alphabet=label_fn.get_str_alphabet()
    )

    level_sampler = XMinigridMetaLevelSampler(
        env_params,
        [
            XMinigridSingleRoomLevelSampler(env_params, min_objects=1, max_objects=5),
            XMinigridTwoRoomsLevelSampler(env_params, min_objects=1, max_objects=10),
            XMinigridFourRoomsLevelSampler(env_params, min_objects=4, max_objects=15),
            XMinigridSixRoomsLevelSampler(env_params, min_objects=7, max_objects=20),
        ]
    )

    problem_sampler = jax.jit(XMinigridLevelConditionedProblemSampler(level_sampler, hrm_sampler, label_fn))

    id_sample = 0
    id_valid_sample = 0

    while id_valid_sample < NUM_PROBLEMS_PER_TYPE:
        level, hrm = problem_sampler(jax.random.PRNGKey(id_sample))
        if is_solvable_problem(level, hrm, alphabet):
            idx = str(id_valid_sample).zfill(int(math.log10(NUM_PROBLEMS_PER_TYPE)) + 1)

            level.to_file(os.path.join(LEVEL_PATH, f"{idx}.yaml"))
            renderer.render_level(level).save(os.path.join(LEVEL_PATH, f"{idx}.png"))

            dump(hrm, os.path.join(HRM_PATH, f"{idx}.yaml"), alphabet)
            render_to_file(hrm, os.path.join(HRM_PATH, f"{idx}.png"), alphabet=alphabet)

            id_valid_sample += 1

        id_sample += 1
        print(id_sample, id_valid_sample)


if __name__ == "__main__":
    os.makedirs(HRM_PATH, exist_ok=True)
    os.makedirs(LEVEL_PATH, exist_ok=True)
    build()
