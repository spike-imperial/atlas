from enum import IntEnum

import jax.numpy as jnp


class Mutations(IntEnum):
    ADD_NON_DOOR_OBJ = 0
    RM_NON_DOOR_OBJ = 1
    MOVE_NON_DOOR_OBJ = 2
    MOVE_AGENT = 3
    REPLACE_DOOR = 4
    REPLACE_NON_DOOR = 5
    ADD_ROOMS = 6
    RM_ROOMS = 7
    SWITCH_PROP = 8
    ADD_TRANSITION = 9
    RM_TRANSITION = 10
    HINDSIGHT_LVL_ONLY = 11
    HINDSIGHT_PRED = 12
    HINDSIGHT_SUCC = 13


class MutationCategories(IntEnum):
    LVL = 0
    HRM = 1
    HINDSIGHT = 2


MUTATION_TO_CATEGORY = jnp.array([
    MutationCategories.LVL,
    MutationCategories.LVL,
    MutationCategories.LVL,
    MutationCategories.LVL,
    MutationCategories.LVL,
    MutationCategories.LVL,
    MutationCategories.LVL,
    MutationCategories.LVL,
    MutationCategories.HRM,
    MutationCategories.HRM,
    MutationCategories.HRM,
    MutationCategories.HINDSIGHT,
    MutationCategories.HINDSIGHT,
    MutationCategories.HINDSIGHT,
])


def mutation_to_str(mutation_id: Mutations) -> str:
    match mutation_id:
        case Mutations.ADD_NON_DOOR_OBJ:
            return "add_obj"
        case Mutations.RM_NON_DOOR_OBJ:
            return "rm_obj"
        case Mutations.MOVE_NON_DOOR_OBJ:
            return "move_obj"
        case Mutations.MOVE_AGENT:
            return "move_agent"
        case Mutations.REPLACE_DOOR:
            return "replace_door"
        case Mutations.REPLACE_NON_DOOR:
            return "replace_non_door"
        case Mutations.ADD_ROOMS:
            return "add_rooms"
        case Mutations.RM_ROOMS:
            return "rm_rooms"
        case Mutations.SWITCH_PROP:
            return "switch_prop"
        case Mutations.ADD_TRANSITION:
            return "add_transition"
        case Mutations.RM_TRANSITION:
            return "rm_transition"
        case Mutations.HINDSIGHT_LVL_ONLY:
            return "hindsight_lvl"
        case Mutations.HINDSIGHT_PRED:
            return "hindsight_pred"
        case Mutations.HINDSIGHT_SUCC:
            return "hindsight_succ"


def mutation_to_category(mutation_id: Mutations) -> MutationCategories:
    return MUTATION_TO_CATEGORY[mutation_id]


def category_to_str(category_id: MutationCategories) -> str:
    match category_id:
        case MutationCategories.LVL:
            return "lvl"
        case MutationCategories.HRM:
            return "hrm"
        case MutationCategories.HINDSIGHT:
            return "hindsight"
