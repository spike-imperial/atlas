"""
Modified from XLand-Minigrid to print the outcome of the labeling function at each step.
"""

import argparse

import jax
import numpy as np
from PIL import Image
import pygame
import pygame.freetype
from pygame.event import Event

from atlas.envs.common.env import Environment, EnvParams
from atlas.envs.common.labeling_function import LabelingFunction
from atlas.envs.common.wrappers import (
    AutoReplayHRMWrapper,
    AutoResetHRMWrapper,
    HRMWrapper,
)
from atlas.envs.xminigrid.env import XMinigridEnv
from atlas.envs.xminigrid.labeling_function import XMinigridLabelingFunction
from atlas.envs.xminigrid.level_sampling.single_room import XMinigridSingleRoomLevelSampler
from atlas.envs.xminigrid.level_sampling.two_rooms import XMinigridTwoRoomsLevelSampler
from atlas.envs.xminigrid.level_sampling.four_rooms import XMinigridFourRoomsLevelSampler
from atlas.envs.xminigrid.level_sampling.six_rooms import XMinigridSixRoomsLevelSampler
from atlas.envs.xminigrid.level_sampling.nine_rooms import XMinigridNineRoomsLevelSampler
from atlas.envs.xminigrid.level_sampling.meta import XMinigridMetaLevelSampler
from atlas.envs.xminigrid.problem_sampling.level_conditioned import XMinigridLevelConditionedProblemSampler
from atlas.hrm.sampling.random_walk import RandomWalkHRMSampler
from atlas.hrm.sampling.single_path_flat import SinglePathFlatHRMSampler
from atlas.hrm.visualization import render_to_img
from atlas.problem_samplers.base import ProblemSampler


class ManualControl:
    def __init__(
        self,
        env: Environment,
        env_params: EnvParams,
        labeling_function: LabelingFunction,
        problem_sampler: ProblemSampler,
        use_hrm_completion: bool,
    ):
        self.env = env
        self.env_params = env_params
        self.labeling_function = labeling_function
        self.problem_sampler = problem_sampler
        self.use_hrm_completion = use_hrm_completion

        self._reset = jax.jit(self.env.reset)
        self._step = jax.jit(self.env.step)
        self._get_label = jax.jit(self.labeling_function.get_label)
        self._sample_problem = jax.jit(self.problem_sampler.sample)
        self._alphabet = self.labeling_function.get_str_alphabet()

        self._key = jax.random.PRNGKey(0)

        self.timestep = None

        self.render_size = None
        self.window = None
        self.clock = None
        self.closed = False

    def render(self) -> None:
        assert self.timestep is not None

        # [h, w, c] -> [w, h, c]
        env_img = self.env.render(self.env_params, self.timestep)
        env_img = np.transpose(env_img, axes=(1, 0, 2))

        # Get HRM image and resize
        hrm_img = render_to_img(
            self.timestep.extras.hrm,
            self.timestep.extras.hrm_state,
            self._alphabet,
            title="",
        )
        hrm_w, hrm_h = hrm_img.size
        aspect_ratio = hrm_w / hrm_h
        new_hrm_w = int(env_img.shape[1] * aspect_ratio)
        hrm_img = np.asarray(
            hrm_img.resize((new_hrm_w, env_img.shape[0]), resample=Image.BILINEAR)
        )
        if hrm_img.shape[2] > 3:
            hrm_img = hrm_img[:, :, :-1]
        hrm_img = np.transpose(hrm_img, axes=(1, 0, 2))

        # Concatenate environment and HRM image
        final_img = np.concatenate((env_img, hrm_img), axis=0)

        if self.render_size is None:
            self.render_size = final_img.shape[:2]

        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.render_size)
            pygame.display.set_caption("xland-minigrid")
        if self.clock is None:
            self.clock = pygame.time.Clock()
        surf = pygame.surfarray.make_surface(final_img)

        self.window.blit(surf, (0, 0))
        pygame.event.pump()
        self.clock.tick(10)
        pygame.display.flip()

    def start(self) -> None:
        self.reset()

        """Start the window display with blocking event loop"""
        while not self.closed:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    break
                if event.type == pygame.KEYDOWN:
                    event.key = pygame.key.name(int(event.key))
                    self.key_handler(event)

    def step(self, action: int) -> None:
        self.timestep = self._step(self.env_params, self.timestep, action)
        self.render()
        self.print_info()

    def reset(self) -> None:
        print("Reset!")
        self._key, problem_key, reset_key = jax.random.split(self._key, 3)

        level, hrm = self._sample_problem(problem_key)
        self.timestep = self._reset(
            key=reset_key, env_params=self.env_params, level=level, hrm=hrm
        )

        self.render()
        self.print_info()

    def print_info(self):
        print(
            "StepType:", self.timestep.step_type,
            "Discount:", self.timestep.discount,
            "Reward:", self.timestep.reward,
            "Task completed:", self.timestep.extras.task_completed,
            "Label:", self.labeling_function.label_to_prop_list(self._get_label(self.timestep)),
            end=""
        )

        if self.use_hrm_completion:
            print("HRM completion:", self.timestep.extras.hrm_completion)
        else:
            print()

    def key_handler(self, event: Event) -> None:
        key: str = event.key
        print("pressed", key)

        if key == "escape":
            self.close()
            return
        elif key == "n":
            self.reset()
            return

        key_to_action = {
            "up": 0,
            "right": 1,
            "left": 2,
            "tab": 3,
            "left shift": 4,
            "space": 5,
        }
        if key in key_to_action.keys():
            action = key_to_action[key]
            self.step(action)
        else:
            print(key)

    def close(self) -> None:
        if self.window:
            pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--room_type",
        type=str,
        default="all",
        choices=["single_room", "two_rooms", "four_rooms", "six_rooms", "nine_rooms", "all"],
    )
    parser.add_argument(
        "--wrapper",
        type=str,
        default="autoreset",
        choices=["vanilla", "autoreset", "autoreplay"],
    )
    parser.add_argument(
        "--hrm_sampler",
        type=str,
        default="single_path_flat",
        choices=["random_walk", "single_path_flat"],
    )
    parser.add_argument(
        "--disable_hrm_reward",
        action="store_true",
    )
    parser.add_argument(
        "--enable_hrm_completion",
        action="store_true",
    )

    args = parser.parse_args()

    env = XMinigridEnv()
    env_params = env.default_params().replace(max_steps=100, height=19, width=19)
    labeling_function = XMinigridLabelingFunction(
        env_params,
        use_front_props=True,
        use_carrying_props=True,
        use_next_to_props=True,
    )

    if args.room_type == "single_room":
        level_sampler = XMinigridSingleRoomLevelSampler(env_params)
    elif args.room_type == "two_rooms":
        level_sampler = XMinigridTwoRoomsLevelSampler(env_params)
    elif args.room_type == "four_rooms":
        level_sampler = XMinigridFourRoomsLevelSampler(env_params)
    elif args.room_type == "six_rooms":
        level_sampler = XMinigridSixRoomsLevelSampler(env_params)
    elif args.room_type == "nine_rooms":
        level_sampler = XMinigridNineRoomsLevelSampler(env_params)
    elif args.room_type == "all":
        level_sampler = XMinigridMetaLevelSampler(env_params, [
            XMinigridSingleRoomLevelSampler(env_params),
            XMinigridTwoRoomsLevelSampler(env_params),
            XMinigridFourRoomsLevelSampler(env_params),
            XMinigridSixRoomsLevelSampler(env_params),
            XMinigridNineRoomsLevelSampler(env_params)
        ])
    else:
        raise RuntimeError(f"Error: Unknown room type '{args.room_type}'.")

    if args.hrm_sampler == "random_walk":
        hrm_sampler = RandomWalkHRMSampler(
            max_num_rms=3,
            max_num_states=4,
            max_num_edges=1,
            max_num_literals=5,
            alphabet_size=labeling_function.get_alphabet_size(),
            alphabet=labeling_function.get_str_alphabet(),
        )
    elif args.hrm_sampler == "single_path_flat":
        hrm_sampler = SinglePathFlatHRMSampler(
            max_num_rms=1,
            max_num_states=5,
            max_num_edges=1,
            max_num_literals=5,
            alphabet_size=labeling_function.get_alphabet_size(),
            num_transitions=2,
            reward_on_acceptance_only=True,
        )

    problem_sampler = XMinigridLevelConditionedProblemSampler(level_sampler, hrm_sampler, labeling_function)

    if args.wrapper == "vanilla":
        env = HRMWrapper(env, labeling_function, not args.disable_hrm_reward, args.enable_hrm_completion)
    elif args.wrapper == "autoreset":
        env = AutoResetHRMWrapper(env, labeling_function, problem_sampler, not args.disable_hrm_reward, args.enable_hrm_completion)
    elif args.wrapper == "autoreplay":
        env = AutoReplayHRMWrapper(env, labeling_function, not args.disable_hrm_reward, args.enable_hrm_completion)

    control = ManualControl(env, env_params, labeling_function, problem_sampler, args.enable_hrm_completion)
    control.start()
