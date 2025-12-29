from typing import List

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageOps

from .evaluation import Rollout
from ..envs.common.renderer import EnvironmentRenderer
from ..envs.common.types import EnvParams
from ..hrm.visualization import render_to_img


class RolloutRenderer:
    """
    Renders rollouts of HRM-conditioned policies.
    """

    def __init__(self, env_renderer: EnvironmentRenderer):
        self._env_renderer = env_renderer

    def render(
        self,
        rollout: Rollout,
        env_params: EnvParams,
        alphabet: List[str],
        max_rollout_length: int,
        macro_block_size: int = 16,
    ) -> np.array:
        num_steps = min(rollout.length + 1, max_rollout_length)

        env_frames, hrm_frames = [], []
        last_hrm_frame = None
        env_w, env_h = None, None
        hrm_w = None

        for i in range(num_steps):
            p_hrm_state = jax.tree_util.tree_map(lambda x: x[i - 1], rollout.hrm_states)
            env_state, hrm_state = jax.tree_util.tree_map(lambda x: x[i], (rollout.states, rollout.hrm_states))

            # Render the environment step and resize it to be divisible
            # by macro_block_size
            env_frame = self._env_renderer.render(env_state, env_params)
            env_frame = self._resize_frame(env_frame, macro_block_size)
            env_w, env_h = env_frame.size
            env_frames.append(env_frame)

            # Render HRM
            is_same_hrm_state = jax.tree_util.tree_all(jax.tree_util.tree_map(
                lambda x, y: jnp.allclose(x, y), p_hrm_state, hrm_state
            ))
            if last_hrm_frame is None or jnp.logical_not(is_same_hrm_state):
                last_hrm_frame = render_to_img(rollout.hrm, hrm_state, alphabet)

                # Resize the HRM frames to match the height of environment frames while
                # preserving aspect ratio
                hrm_w_original, hrm_h_original = last_hrm_frame.size
                aspect_ratio = hrm_w_original / hrm_h_original
                hrm_w = int(env_h * aspect_ratio)
                last_hrm_frame = last_hrm_frame.resize((hrm_w, env_h), resample=Image.BILINEAR)

            hrm_frames.append(last_hrm_frame)

        # Add padding between hrm_ims and env_ims, and ensure divisible by macro_block_size
        excess_left_margin = macro_block_size - (hrm_frames[0].width % macro_block_size)
        padding = macro_block_size * 4
        img_arr = 255 * np.ones(
            (len(hrm_frames), env_h, excess_left_margin + hrm_w + padding + env_w, 3), dtype=np.uint8
        )

        for i, (hrm_frame, env_frame) in enumerate(zip(hrm_frames, env_frames)):
            # Add the resized HRM frame
            img_arr[i, :, excess_left_margin:excess_left_margin + hrm_w, :] = np.array(hrm_frame)[..., :3]

            # Add the environment frame
            img_arr[i, :, excess_left_margin + hrm_w + padding:excess_left_margin + hrm_w + padding + env_w, :] = np.array(env_frame)

        return img_arr

    def _resize_frame(self, frame, macro_block_size):
        frame_w, frame_h = frame.size
        aspect_ratio = frame_w / frame_h

        # Calculate new dimensions while maintaining aspect ratio
        new_w = (frame_w + macro_block_size - 1) // macro_block_size * macro_block_size
        new_h = int(new_w / aspect_ratio)

        # If new_h is not divisible by macro_block_size, adjust new_w and pad
        if new_h % macro_block_size != 0:
            new_h = (
                (new_h + macro_block_size - 1) // macro_block_size * macro_block_size
            )
            new_w = int(new_h * aspect_ratio)
            resized_frame = frame.resize((new_w, new_h))

            # Calculate padding to make the width divisible by macro_block_size
            pad_w = (
                (new_w + macro_block_size - 1) // macro_block_size * macro_block_size
            )
            pad_h = new_h

            # Pad the resized frame
            padded_frame = ImageOps.pad(
                resized_frame,
                (pad_w, pad_h),
                color=(255, 255, 255),
                centering=(0.5, 0.5),
            )
            return padded_frame

        return frame.resize((new_w, new_h))
