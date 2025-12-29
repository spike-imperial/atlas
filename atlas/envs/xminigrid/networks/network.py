import math
from typing import Dict, List, Tuple

import chex
import distrax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import orthogonal

from .embedding_encoder import EmbeddingEncoder
from ....networks.cond_actor_critic_rnn import ActorCriticInput, ConditionedActorCriticRNN


class XMinigridNetwork(ConditionedActorCriticRNN):
    """
    Actor-critic network with shared observation encodings. By default,
    it uses the egocentric observations if present; otherwise, it uses
    the full observations.
    """
    head_hidden_dim: int = 256         # Dimensionality of the actor-critic heads
    pre_ac_hidden_dim: int = -1        # Layer before the actor-critic heads
    use_actions: bool = True           # Whether to concatenate embedded actions to the embedded obs.
    use_rewards: bool = False          # Whether to concatenate embedded rewards to the embedded obs.
    use_obj_encoder: bool = True       # Whether to transform the tiles and colors into embeddings before the CNN
    use_rnn_cond: bool = True          # Whether to apply the RNN over the conditioning vector
    num_ac_layers: int = 2             # Number of hidden layers for the actor and the critic
    use_relu_ac: bool = True           # Use ReLUs in the actor-critic layers (old: False)
    cnn_features: Tuple = (32, 64, 64) # Features used for each CNN layer (old: 16, 32, 64)
    cnn_padding: str = "SAME"

    @nn.compact
    def __call__(
        self, inputs: ActorCriticInput, h_state: chex.Array, c_vector: chex.Array
    ) -> Tuple[distrax.Categorical, Dict, chex.Array]:
        B, S = inputs["done"].shape[:2]

        # Check conditioning vector shape is correct
        assert c_vector.shape[:2] == (B, S), (
            f"Expected conditioning vector to have shape {(B, S)}, got {c_vector.shape}"
        )

        # Encode the observation, [batch_size, seq_len, ...] and append action and/or reward
        # information
        out = self._get_encoded_obs(inputs["observation"]).reshape(B, S, -1)
        if self.use_actions:
            out = jnp.concatenate([out, self._get_encoded_action(inputs)], axis=-1)
        if self.use_rewards:
            out = jnp.concatenate([out, self._get_encoded_reward(inputs)], axis=-1)

        # Concatenate the conditioning vector
        if self.use_rnn_cond:
            out = jnp.concatenate([out, c_vector], axis=-1)

        # Pass observations through the RNN (history of obs, rewards and actions)
        new_hidden, out = self._get_rnn_model()((out, inputs["done"]), h_state)

        # Concatenate the conditioning vector
        if not self.use_rnn_cond:
            out = jnp.concatenate([out, c_vector], axis=-1)

        # An extra layer before the actor-critic heads
        if self.pre_ac_hidden_dim > 0:
            out = nn.Sequential([nn.Dense(self.pre_ac_hidden_dim), nn.relu])(out)

        # Core networks
        actor = nn.Sequential([
            self._get_ac_hidden_layers(),
            nn.Dense(self.num_actions, kernel_init=orthogonal(0.1)),
        ])

        critic = nn.Sequential([
            self._get_ac_hidden_layers(),
            nn.Dense(1, kernel_init=orthogonal(1.0)),
        ])

        return (
            distrax.Categorical(logits=actor(out)),
            {"critic": jnp.squeeze(critic(out), axis=-1)},
            new_hidden
        )

    def _get_ac_hidden_layers(self):
        return nn.Sequential([
            nn.Sequential([
                nn.Dense(self.head_hidden_dim, kernel_init=orthogonal(2.0)),
                nn.relu if self.use_relu_ac else nn.tanh,
            ])
            for _ in range(self.num_ac_layers)
        ])

    def _get_encoded_obs(self, observation: Dict) -> chex.Array:
        obj_encoder = [EmbeddingEncoder(16)] if self.use_obj_encoder else []
        obs_encoder = nn.Sequential([
            *obj_encoder,
            *[
                nn.Sequential([
                    nn.Conv(feats, (2, 2), padding=self.cnn_padding, kernel_init=orthogonal(math.sqrt(2))),
                    nn.relu,
                ])
                for feats in self.cnn_features
            ]
        ])

        return obs_encoder(observation["ego"]) if "ego" in observation else obs_encoder(observation["full"])
