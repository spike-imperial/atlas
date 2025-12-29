from typing import Dict, Optional

import chex
from chex import dataclass
import jax

from .common import HRMSampler
from .. import ops
from ..types import HRM


@dataclass
class SinglePathFlatHRMSamplerExtras:
    rng: chex.PRNGKey


class SinglePathFlatHRMSampler(HRMSampler):
    """
    Samples an HRM consisting of a single RM with a specifiable number of
    transitions from the initial to the accepting state. Each transition
    is labeled with a random proposition between 0 (inclusive) and the size
    of the alphabet (exclusive). The reward can be either (i) 0 everywhere
    except for the transition to the accepting state, for which it is 1, or
    (ii) 1 for every transition to a different state (self-transitions are
    still 0-rewarded).
    """

    ROOT_ID = 0
    EDGE_ID = 0

    def __init__(
        self,
        max_num_rms: int,
        max_num_states: int,
        max_num_edges: int,
        max_num_literals: int,
        alphabet_size: int,
        num_transitions: int,
        reward_on_acceptance_only: bool,
        **kwargs: dict,
    ):
        super().__init__(max_num_rms, max_num_states, max_num_edges, max_num_literals, alphabet_size)
        self._num_transitions = num_transitions
        self.reward_on_acceptance_only = reward_on_acceptance_only

    def sample(self, key: chex.PRNGKey, extras: Optional[Dict] = None) -> HRM:
        hrm = self._init_hrm(self.ROOT_ID, SinglePathFlatHRMSamplerExtras(rng=key))

        propositions = jax.random.choice(
            key,
            self._alphabet_size,
            shape=(self._num_transitions,),
            p=extras["prop_probs"] if extras is not None else None
        )

        src_id = ops.get_initial_state_id()

        for i in range(self._num_transitions):
            dst_id = (
                src_id + 1
                if i < self._num_transitions - 1
                else ops.get_accepting_state_id(hrm)
            )
            transition = dict(
                hrm=hrm,
                rm_id=self.ROOT_ID,
                src_id=src_id,
                dst_id=dst_id,
            )

            ops.add_leaf_call(**transition, edge_id=self.EDGE_ID)
            ops.add_condition(
                **transition,
                edge_id=self.EDGE_ID,
                proposition=propositions[i],
                is_positive=True,
            )

            if not self.reward_on_acceptance_only or ops.is_accepting_state(hrm, dst_id):
                ops.add_reward(**transition, reward=1.0)

            src_id = dst_id

        return hrm
