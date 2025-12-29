import math
import os

import jax

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
from atlas.hrm.sampling.single_path_flat import SinglePathFlatHRMSampler
from atlas.hrm.visualization import render_to_file

HRM_PATH = os.path.join(os.path.abspath("."), "hrms")
LEVEL_PATH = os.path.join(os.path.abspath("."), "levels")

MIN_NUM_TRANSITIONS = 1
MAX_NUM_TRANSITIONS = 5
NUM_PROBLEMS_PER_TYPE = 5


def build():
    env_params = XMinigridEnvParams(width=19, height=13)
    renderer = XMinigridRenderer()
    label_fn = XMinigridLabelingFunction(env_params)
    alphabet = label_fn.get_str_alphabet()

    id_sample = 0
    for num_rooms, obj_num, level_sampler in [
        (1, "bot", XMinigridSingleRoomLevelSampler(env_params, min_objects=1, max_objects=2)),
        (1, "mid", XMinigridSingleRoomLevelSampler(env_params, min_objects=3, max_objects=4)),
        (1, "top", XMinigridSingleRoomLevelSampler(env_params, min_objects=5, max_objects=5)),
        (2, "bot", XMinigridTwoRoomsLevelSampler(env_params, min_objects=1, max_objects=3)),
        (2, "mid", XMinigridTwoRoomsLevelSampler(env_params, min_objects=4, max_objects=7)),
        (2, "top", XMinigridTwoRoomsLevelSampler(env_params, min_objects=8, max_objects=10)),
        (4, "bot", XMinigridFourRoomsLevelSampler(env_params, min_objects=4, max_objects=7)),
        (4, "mid", XMinigridFourRoomsLevelSampler(env_params, min_objects=8, max_objects=11)),
        (4, "top", XMinigridFourRoomsLevelSampler(env_params, min_objects=12, max_objects=15)),
        (6, "bot", XMinigridSixRoomsLevelSampler(env_params, min_objects=7, max_objects=10)),
        (6, "mid", XMinigridSixRoomsLevelSampler(env_params, min_objects=11, max_objects=16)),
        (6, "top", XMinigridSixRoomsLevelSampler(env_params, min_objects=17, max_objects=20)),
    ]:
        for type_id, num_transitions in enumerate(range(MIN_NUM_TRANSITIONS, MAX_NUM_TRANSITIONS + 1)):
            hrm_sampler = SinglePathFlatHRMSampler(
                max_num_rms=1,
                max_num_states=num_transitions + 1,
                max_num_edges=1,
                max_num_literals=1,
                alphabet_size=label_fn.get_alphabet_size(),
                num_transitions=num_transitions,
                reward_on_acceptance_only=True,
            )

            problem_sampler = jax.jit(XMinigridLevelConditionedProblemSampler(level_sampler, hrm_sampler, label_fn))

            id_valid_sample = 0
            while id_valid_sample < NUM_PROBLEMS_PER_TYPE:
                level, hrm = problem_sampler(jax.random.PRNGKey(id_sample))
                if is_solvable_problem(level, hrm, alphabet):
                    idx = f"r{num_rooms}_{obj_num}_t{num_transitions}_" + str(id_valid_sample).zfill(int(math.log10(NUM_PROBLEMS_PER_TYPE)) + 1)

                    level.to_file(os.path.join(LEVEL_PATH, f"{idx}.yaml"))
                    renderer.render_level(level).save(os.path.join(LEVEL_PATH, f"{idx}.png"))

                    dump(hrm, os.path.join(HRM_PATH, f"{idx}.yaml"), alphabet)
                    render_to_file(hrm, os.path.join(HRM_PATH, f"{idx}.png"), alphabet=alphabet)

                    id_valid_sample += 1

                id_sample += 1


if __name__ == "__main__":
    os.makedirs(HRM_PATH, exist_ok=True)
    os.makedirs(LEVEL_PATH, exist_ok=True)
    build()
