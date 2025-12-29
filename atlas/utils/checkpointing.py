import absl
absl.logging.set_verbosity(absl.logging.ERROR)  # Reduce verbosity of orbax checkpointing
import os
from typing import Optional

import jax
import jax.numpy as jnp
from omegaconf import DictConfig
import orbax.checkpoint as ocp

from ..runners.base_runner_state import RunnerState


def setup_checkpointing(config: DictConfig) -> ocp.CheckpointManager:
    """
    This takes in the train state and config, and returns an orbax checkpoint manager.

    Based on: https://github.com/DramaCow/jaxued/blob/main/examples/maze_dr.py
    """
    overall_save_dir = get_checkpoints_save_dir(config)
    os.makedirs(overall_save_dir, exist_ok=True)
    return ocp.CheckpointManager(overall_save_dir, options=ocp.CheckpointManagerOptions(max_to_keep=2))


def get_checkpoints_save_dir(config: DictConfig) -> str:
    """
    Returns the path of the checkpoint saved during training.
    """
    run_name = config.logging.run_id if config.logging.run_id else config.logging.run_name
    return os.path.join(os.getcwd(), "checkpoints", run_name)


def get_checkpoint_save_dir(config: DictConfig, step: int) -> str:
    """
    Returns the path of a specific checkpoint based on the given step.
    """
    return os.path.join(get_checkpoints_save_dir(config), str(step))


def save_checkpoint(checkpoint_manager: ocp.CheckpointManager, runner_state: RunnerState, step: int):
    try:
        checkpoint_manager.save(step, runner_state, force=True, args=ocp.args.StandardSave(runner_state))
        checkpoint_manager.wait_until_finished()
    except ocp.checkpoint_manager.StepAlreadyExistsError as e:
        print(f"Caught exception: {e}")


def load_runner_state(path: str, target: Optional[RunnerState] = None, step: Optional[int] = None) -> RunnerState:
    """
    Returns the RunnerState in a checkpoint (in dictionary form!).
    """
    checkpoint_manager = ocp.CheckpointManager(path, ocp.StandardCheckpointer())
    step = checkpoint_manager.latest_step() if step is None else step
    step_metadata = checkpoint_manager.item_metadata(step)
    if target is None:
        target = jax.tree_util.tree_map(
            lambda x: jnp.zeros(x.shape, x.dtype),
            step_metadata
        )
    elif "buffer" in step_metadata:
        # A small fix to ensure checkpoint reusability for older PLR runs
        mutation_ids = step_metadata["buffer"]["extra"]["mutation_ids"]
        mutation_args = step_metadata["buffer"]["extra"]["mutation_args"]
        target.buffer["extra"]["mutation_ids"] = jnp.zeros(mutation_ids.shape, dtype=mutation_ids.dtype)
        target.buffer["extra"]["mutation_args"] = jnp.zeros(mutation_args.shape, dtype=mutation_args.dtype)
    return checkpoint_manager.restore(step, target)
