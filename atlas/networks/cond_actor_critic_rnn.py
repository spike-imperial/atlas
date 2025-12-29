from typing import Any, Dict, Tuple, TypedDict, Union

import chex
import distrax
from flax import linen as nn
import jax.numpy as jnp

from .rnn import RNNModel


class ActorCriticInput(TypedDict):
    observation: Any
    done: chex.Array
    prev_action: chex.Array
    prev_reward: chex.Array


class ConditionedActorCriticRNN(nn.Module):
    num_actions: int
    use_actions: bool = True
    action_emb_dim: int = 16
    rnn_cell_type: str = "gru"
    rnn_hidden_dim: int = 512

    @nn.compact
    def __call__(
        self, inputs: ActorCriticInput, h_state: chex.Array, c_vector: chex.Array
    ) -> Tuple[distrax.Categorical, Dict, chex.Array]:
        """
        Args:
            inputs: data coming from the environment and agent's decisions.
            h_state: the hidden state of the RNN.
            c_vector: the vector to which the actor-critic are conditioned.

        Returns:
            - action probability distribution
            - the critic values (a dictionary since there might be multiple
              critics, e.g. in the asymmetric XMinigrid architecture)
            - the next hidden state of the RNN
        """
        raise NotImplementedError

    def initialize_hidden_state(self, batch_size: int) -> Union[chex.Array, Tuple[chex.Array, chex.Array]]:
        """
        Initializes the hidden state of the RNN.
        """
        # Originally tried to do:
        #   return self.cell_type(self.rnn_hidden_dim).initialize_carry(
        #       jax.random.PRNGKey(0), (batch_size, self.rnn_hidden_dim)
        #   )
        # but `flax` does not like doing this inside class instances.
        # A possible solution is to make the method static, but then the
        # conditioners need to be static too and required a few mods.
        # The following lines just perform the default initialization of
        # the different RNN cells.
        if self.rnn_cell_type == "lstm":
            return (jnp.zeros((batch_size, self.rnn_hidden_dim)), jnp.zeros((batch_size, self.rnn_hidden_dim)))
        elif self.rnn_cell_type == "gru":
            return jnp.zeros((batch_size, self.rnn_hidden_dim))

    def _get_rnn_model(self) -> RNNModel:
        if self.rnn_cell_type == "lstm":
            rnn_cell = nn.OptimizedLSTMCell(self.rnn_hidden_dim)
        elif self.rnn_cell_type == "gru":
            rnn_cell = nn.GRUCell(self.rnn_hidden_dim)
        return RNNModel(rnn_cell)

    def _get_encoded_action(self, inputs: ActorCriticInput) -> chex.Array:
        # If `done`, set the previous action to 0 (there is no such action)
        # Note that when done=True, we have started a new episode because of
        # the autoreset.
        prev_action = inputs["prev_action"] * jnp.logical_not(inputs["done"])
        return nn.Embed(self.num_actions, self.action_emb_dim)(prev_action)

    def _get_encoded_reward(self, inputs: ActorCriticInput) -> chex.Array:
        # If `done`, set the previous reward to 0 (there is no such reward)
        prev_reward = inputs["prev_reward"] * jnp.logical_not(inputs["done"])
        return prev_reward[..., jnp.newaxis]
