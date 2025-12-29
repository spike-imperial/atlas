import chex
import jax
from flax import linen as nn

from .conditioned_rnn_agent import ConditionedRNNAgent
from .types import ConditionedAgentState
from ..conditioners.types import Conditioner
from ..hrm.types import HRM, HRMState


class HRMConditionedAgent(nn.Module):
    hrm_conditioner: Conditioner
    agent: ConditionedRNNAgent

    def __call__(
        self,
        observation: chex.Array,
        done: chex.Array,
        hrm: HRM,
        hrm_state: HRMState,
        prev_action: chex.Array,
        prev_reward: chex.Array,
        cond_agent_state: ConditionedAgentState,
    ):
        # Update the conditioner state and obtain the output of the conditioner
        c_state, c_out = self.hrm_conditioner(
            c_state=cond_agent_state.c_state,
            hrm=hrm,
            hrm_state=hrm_state,
        )

        # Concatenate the output of the conditioner with the observation
        a_state, a_out = self.agent(
            observation,
            done,
            prev_action,
            prev_reward,
            cond_agent_state.a_state,
            c_out.conditioning_vector,
        )

        return ConditionedAgentState(c_state=c_state, a_state=a_state), (c_out, a_out)

    def initialize_state(self, batch_size: int, rng: chex.PRNGKey) -> ConditionedAgentState:
        c_rng, a_rng = jax.random.split(rng)
        return ConditionedAgentState(
            c_state=self.hrm_conditioner.initialize_state(batch_size, c_rng),
            a_state=self.agent.initialize_state(batch_size, a_rng),
        )
