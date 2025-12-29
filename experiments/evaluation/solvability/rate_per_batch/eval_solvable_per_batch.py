import jax
import numpy as jnp
import pandas as pd
from tqdm import tqdm

from atlas.envs.xminigrid.labeling_function import XMinigridLabelingFunction
from atlas.envs.xminigrid.level_sampling.four_rooms import XMinigridFourRoomsLevelSampler
from atlas.envs.xminigrid.level_sampling.two_rooms import XMinigridTwoRoomsLevelSampler
from atlas.envs.xminigrid.level_sampling.meta import XMinigridMetaLevelSampler
from atlas.envs.xminigrid.level_sampling.single_room import XMinigridSingleRoomLevelSampler
from atlas.envs.xminigrid.level_sampling.six_rooms import XMinigridSixRoomsLevelSampler
from atlas.envs.xminigrid.problem_sampling.level_conditioned import XMinigridLevelConditionedProblemSampler
from atlas.envs.xminigrid.types import XMinigridEnvParams
from atlas.envs.xminigrid.validation import is_solvable_problem
from atlas.hrm.sampling.meta import MetaHRMSampler
from atlas.hrm.sampling.random_walk import RandomWalkHRMSampler
from atlas.hrm.sampling.single_path_flat import SinglePathFlatHRMSampler
from atlas.problem_samplers.independent import IndependentProblemSampler

ENV_PARAMS = XMinigridEnvParams(height=13, width=19)
LABEL_FN = XMinigridLabelingFunction(ENV_PARAMS)
ALPHABET = LABEL_FN.get_str_alphabet()

XMINIGRID_FULL = (
    XMinigridMetaLevelSampler(
        ENV_PARAMS,
        [
            XMinigridSingleRoomLevelSampler(ENV_PARAMS, min_objects=1, max_objects=5),
            XMinigridTwoRoomsLevelSampler(ENV_PARAMS, min_objects=1, max_objects=10),
            XMinigridFourRoomsLevelSampler(ENV_PARAMS, min_objects=4, max_objects=15),
            XMinigridSixRoomsLevelSampler(ENV_PARAMS, min_objects=7, max_objects=20),
        ]
    ),
    "full"
)

XMINIGRID_ONE_L = (
    XMinigridSingleRoomLevelSampler(ENV_PARAMS, min_objects=1, max_objects=2),
    "1l"
)
XMINIGRID_ONE_M = (
    XMinigridSingleRoomLevelSampler(ENV_PARAMS, min_objects=3, max_objects=4),
    "1m"
)
XMINIGRID_ONE_H = (
    XMinigridSingleRoomLevelSampler(ENV_PARAMS, min_objects=5, max_objects=5),
    "1h"
)

XMINIGRID_TWO_L = (
    XMinigridTwoRoomsLevelSampler(ENV_PARAMS, min_objects=1, max_objects=3),
    "2l"
)
XMINIGRID_TWO_M = (
    XMinigridTwoRoomsLevelSampler(ENV_PARAMS, min_objects=4, max_objects=7),
    "2m"
)
XMINIGRID_TWO_H = (
    XMinigridTwoRoomsLevelSampler(ENV_PARAMS, min_objects=8, max_objects=10),
    "2h"
)

XMINIGRID_FOUR_L = (
    XMinigridFourRoomsLevelSampler(ENV_PARAMS, min_objects=4, max_objects=7),
    "4l"
)
XMINIGRID_FOUR_M = (
    XMinigridFourRoomsLevelSampler(ENV_PARAMS, min_objects=8, max_objects=11),
    "4m"
)
XMINIGRID_FOUR_H = (
    XMinigridFourRoomsLevelSampler(ENV_PARAMS, min_objects=12, max_objects=15),
    "4h"
)

XMINIGRID_SIX_L = (
    XMinigridSixRoomsLevelSampler(ENV_PARAMS, min_objects=7, max_objects=10),
    "6l"
)
XMINIGRID_SIX_M = (
    XMinigridSixRoomsLevelSampler(ENV_PARAMS, min_objects=11, max_objects=16),
    "6m"
)
XMINIGRID_SIX_H = (
    XMinigridSixRoomsLevelSampler(ENV_PARAMS, min_objects=17, max_objects=20),
    "6h"
)

HRM_SEQ_ARGS = dict(
    max_num_rms=1,
    max_num_states=6,
    max_num_edges=1,
    max_num_literals=1,
    alphabet_size=LABEL_FN.get_alphabet_size(),
)

HRM_SEQ_ONE = (
    SinglePathFlatHRMSampler(**HRM_SEQ_ARGS, num_transitions=1, reward_on_acceptance_only=True),
    "s1"
)
HRM_SEQ_TWO = (
    SinglePathFlatHRMSampler(**HRM_SEQ_ARGS, num_transitions=2, reward_on_acceptance_only=True),
    "s2"
)
HRM_SEQ_THREE = (
    SinglePathFlatHRMSampler(**HRM_SEQ_ARGS, num_transitions=3, reward_on_acceptance_only=True),
    "s3"
)
HRM_SEQ_FOUR = (
    SinglePathFlatHRMSampler(**HRM_SEQ_ARGS, num_transitions=4, reward_on_acceptance_only=True),
    "s4"
)
HRM_SEQ_FIVE = (
    SinglePathFlatHRMSampler(**HRM_SEQ_ARGS, num_transitions=5, reward_on_acceptance_only=True),
    "s5"
)

HRM_SEQ_FULL = (
    MetaHRMSampler(
        [x[0] for x in [HRM_SEQ_ONE, HRM_SEQ_TWO, HRM_SEQ_THREE, HRM_SEQ_FOUR, HRM_SEQ_FIVE]],
        **HRM_SEQ_ARGS
    ),
    "sfull"
)

RW = (
    RandomWalkHRMSampler(
        max_num_rms=1,
        max_num_states=6,
        max_num_edges=1,
        max_num_literals=5,
        alphabet_size=LABEL_FN.get_alphabet_size(),
        alphabet=LABEL_FN.get_str_alphabet()
    ),
    "rw"
)

COMBINATIONS = [
    (XMINIGRID_FULL, HRM_SEQ_FULL),
    (XMINIGRID_FULL, RW),

    (XMINIGRID_ONE_L, HRM_SEQ_ONE),
    (XMINIGRID_ONE_L, HRM_SEQ_TWO),
    (XMINIGRID_ONE_L, HRM_SEQ_THREE),
    (XMINIGRID_ONE_L, HRM_SEQ_FOUR),
    (XMINIGRID_ONE_L, HRM_SEQ_FIVE),

    (XMINIGRID_ONE_M, HRM_SEQ_ONE),
    (XMINIGRID_ONE_M, HRM_SEQ_TWO),
    (XMINIGRID_ONE_M, HRM_SEQ_THREE),
    (XMINIGRID_ONE_M, HRM_SEQ_FOUR),
    (XMINIGRID_ONE_M, HRM_SEQ_FIVE),

    (XMINIGRID_ONE_H, HRM_SEQ_ONE),
    (XMINIGRID_ONE_H, HRM_SEQ_TWO),
    (XMINIGRID_ONE_H, HRM_SEQ_THREE),
    (XMINIGRID_ONE_H, HRM_SEQ_FOUR),
    (XMINIGRID_ONE_H, HRM_SEQ_FIVE),

    (XMINIGRID_TWO_L, HRM_SEQ_ONE),
    (XMINIGRID_TWO_L, HRM_SEQ_TWO),
    (XMINIGRID_TWO_L, HRM_SEQ_THREE),
    (XMINIGRID_TWO_L, HRM_SEQ_FOUR),
    (XMINIGRID_TWO_L, HRM_SEQ_FIVE),

    (XMINIGRID_TWO_M, HRM_SEQ_ONE),
    (XMINIGRID_TWO_M, HRM_SEQ_TWO),
    (XMINIGRID_TWO_M, HRM_SEQ_THREE),
    (XMINIGRID_TWO_M, HRM_SEQ_FOUR),
    (XMINIGRID_TWO_M, HRM_SEQ_FIVE),

    (XMINIGRID_TWO_H, HRM_SEQ_ONE),
    (XMINIGRID_TWO_H, HRM_SEQ_TWO),
    (XMINIGRID_TWO_H, HRM_SEQ_THREE),
    (XMINIGRID_TWO_H, HRM_SEQ_FOUR),
    (XMINIGRID_TWO_H, HRM_SEQ_FIVE),

    (XMINIGRID_FOUR_L, HRM_SEQ_ONE),
    (XMINIGRID_FOUR_L, HRM_SEQ_TWO),
    (XMINIGRID_FOUR_L, HRM_SEQ_THREE),
    (XMINIGRID_FOUR_L, HRM_SEQ_FOUR),
    (XMINIGRID_FOUR_L, HRM_SEQ_FIVE),

    (XMINIGRID_FOUR_M, HRM_SEQ_ONE),
    (XMINIGRID_FOUR_M, HRM_SEQ_TWO),
    (XMINIGRID_FOUR_M, HRM_SEQ_THREE),
    (XMINIGRID_FOUR_M, HRM_SEQ_FOUR),
    (XMINIGRID_FOUR_M, HRM_SEQ_FIVE),

    (XMINIGRID_FOUR_H, HRM_SEQ_ONE),
    (XMINIGRID_FOUR_H, HRM_SEQ_TWO),
    (XMINIGRID_FOUR_H, HRM_SEQ_THREE),
    (XMINIGRID_FOUR_H, HRM_SEQ_FOUR),
    (XMINIGRID_FOUR_H, HRM_SEQ_FIVE),

    (XMINIGRID_SIX_L, HRM_SEQ_ONE),
    (XMINIGRID_SIX_L, HRM_SEQ_TWO),
    (XMINIGRID_SIX_L, HRM_SEQ_THREE),
    (XMINIGRID_SIX_L, HRM_SEQ_FOUR),
    (XMINIGRID_SIX_L, HRM_SEQ_FIVE),

    (XMINIGRID_SIX_M, HRM_SEQ_ONE),
    (XMINIGRID_SIX_M, HRM_SEQ_TWO),
    (XMINIGRID_SIX_M, HRM_SEQ_THREE),
    (XMINIGRID_SIX_M, HRM_SEQ_FOUR),
    (XMINIGRID_SIX_M, HRM_SEQ_FIVE),

    (XMINIGRID_SIX_H, HRM_SEQ_ONE),
    (XMINIGRID_SIX_H, HRM_SEQ_TWO),
    (XMINIGRID_SIX_H, HRM_SEQ_THREE),
    (XMINIGRID_SIX_H, HRM_SEQ_FOUR),
    (XMINIGRID_SIX_H, HRM_SEQ_FIVE),
]

NUM_BATCHES = 5
BATCH_SIZE = 4096

if __name__ == "__main__":
    df_rows = []

    count = 0
    for (lvl_sampler, lvl_name), (hrm_sampler, hrm_name) in COMBINATIONS:
        for prob_sampler_cls in ["independent", "conditioned"]:
            if prob_sampler_cls == "independent":
                problem_sampler = IndependentProblemSampler(lvl_sampler, hrm_sampler, LABEL_FN)
            elif prob_sampler_cls == "conditioned":
                problem_sampler = XMinigridLevelConditionedProblemSampler(lvl_sampler, hrm_sampler, LABEL_FN)

            problem_sampler_fn = jax.vmap(problem_sampler.sample)

            num_valid = []
            bar = tqdm(total=NUM_BATCHES * BATCH_SIZE, desc=f"{prob_sampler_cls}-{lvl_name}-{hrm_name}")
            for i in range(NUM_BATCHES):
                levels, hrms = problem_sampler_fn(jax.random.split(jax.random.PRNGKey(i), BATCH_SIZE))

                n = 0
                for j in range(BATCH_SIZE):
                    level, hrm = jax.tree_util.tree_map(lambda x: x[j], (levels, hrms))
                    if is_solvable_problem(level, hrm, ALPHABET):
                        n += 1

                    bar.update(1)

                num_valid.append(n / BATCH_SIZE)

            mean, std = jnp.mean(num_valid), jnp.std(num_valid)
            df_rows.append([prob_sampler_cls, lvl_name, hrm_name, *num_valid, mean, std])

            bar.close()

            df = pd.DataFrame(
                df_rows,
                columns=["Problem", "Level", "HRM", *[f"Batch_{i}" for i in range(NUM_BATCHES)], "Mean", "Std"]
            )
            df.to_csv(f"validity_{count}.csv")  # save intermediate result
            count += 1
