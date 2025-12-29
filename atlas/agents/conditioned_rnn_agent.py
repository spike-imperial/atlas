import chex
from flax import linen as nn

from ..networks.cond_actor_critic_rnn import ActorCriticInput, ConditionedActorCriticRNN


class ConditionedRNNAgent(nn.Module):
    network: ConditionedActorCriticRNN

    def __call__(
        self, 
        observation: chex.Array, 
        done: chex.Array,
        prev_action: chex.Array,
        prev_reward: chex.Array,
        a_state: chex.Array,  # state of the RNN 
        conditioning_vector: chex.Array,
    ):
        dist, values, a_state = self.network(
            ActorCriticInput(
                observation=observation,
                done=done,
                prev_action=prev_action,
                prev_reward=prev_reward,
            ),
            a_state,
            conditioning_vector,
        )

        return a_state, (dist, values)

    def initialize_state(self, batch_size: int, rng: chex.PRNGKey):
        return self.network.initialize_hidden_state(batch_size)
