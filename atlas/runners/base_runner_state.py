import chex
from flax import struct
from flax.training.train_state import TrainState


class RunnerState(struct.PyTreeNode):
    rng: chex.PRNGKey
    train_state: TrainState
