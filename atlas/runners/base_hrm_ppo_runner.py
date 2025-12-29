from abc import abstractmethod
from collections import defaultdict
import functools
import math
import time
from typing import Callable, List, Optional, Union

import chex
from flax.training.train_state import TrainState
from graphviz.backend.execute import CalledProcessError
from hydra.utils import instantiate
import jax
import jax.numpy as jnp
from omegaconf import DictConfig
import optax
from tqdm.auto import tqdm
import wandb

from .base_runner import Runner
from .base_runner_state import RunnerState
from ..agents.conditioned_rnn_agent import ConditionedRNNAgent
from ..agents.hrm_conditioned_agent import HRMConditionedAgent
from ..envs.common.env import Environment
from ..envs.common.labeling_function import LabelingFunction
from ..envs.common.level import Level
from ..envs.common.wrappers import HRMWrapper
from ..eval_loaders.types import EvaluationProblem, EvaluationSetLoader
from ..hrm.types import HRM
from ..problem_samplers.base import ProblemSampler
from ..utils.checkpointing import get_checkpoint_save_dir, load_runner_state, save_checkpoint, setup_checkpointing
from ..utils.evaluation import RolloutStats, Rollout
from ..utils.evaluation import evaluate_agent
from ..utils.plotting_utils import plotly_xminigrid_prob_distrib_solving
from ..utils.rollout_renderer import RolloutRenderer


class BaseHRMConditionedPPORunner(Runner):

    # Size of each batch of problems used for evaluation
    EVAL_BATCH_SIZE = 100

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # Labeling function
        self.label_fn: LabelingFunction = instantiate(cfg.label_fn, env_params=self.env_params)

        # Problem sampler
        self.problem_sampler: ProblemSampler = instantiate(
            cfg.problem_sampler,
            env_params=self.env_params,
            label_fn=self.label_fn,
            gamma=cfg.training.gamma,
        )

        # Agent
        self.hrm_agent = HRMConditionedAgent(
            hrm_conditioner=instantiate(cfg.conditioner, label_fn=self.label_fn),
            agent=ConditionedRNNAgent(
                network=instantiate(cfg.network, num_actions=self.env.num_actions(self.env_params)),
                name="agent",
            )
        )

        # Evaluation environment and instance loader
        # Warning: the training environment depends on the subclass
        self.eval_env = HRMWrapper(self.env, self.label_fn)
        self.eval_loader: EvaluationSetLoader = instantiate(
            cfg.evaluation.loader,
            problem_sampler=self.problem_sampler,       # sampler loader only
            alphabet=self.label_fn.get_str_alphabet(),  # file loader only
            env_params=self.env_params,                 # file loader ony
        )

        # Rollout renderer
        self.renderer = RolloutRenderer(instantiate(cfg.renderer))

    def train(self):
        # Initialize the runner state
        runner_state = self._init_train_runner_state()

        # Initialize the evaluation set
        eval_problems = self.eval_loader.load()

        # Compile the training function
        print("Compiling training function...", end=" ", flush=True)
        t = time.time()
        train_and_eval_step_fn = jax.jit(self._make_train_and_eval_step_fn()).lower(
            runner_state, eval_problems,
        ).compile()
        print(f"Completed [{time.time() - t:.2f}s].", flush=True)

        # Checkpointing
        if self.cfg.logging.checkpoint.training.storing.enabled:
            checkpoint_manager = setup_checkpointing(self.cfg)

        # Profiling
        if self.cfg.profile_code:
            jax.profiler.start_trace("/tmp/tensorboard")

        # Training and logging loop
        init = self._get_num_updates(runner_state) // self.num_updates_per_iteration
        pbar = tqdm(initial=init, total=self.num_updates // self.num_updates_per_iteration, leave=False,)
        for step in range(init, pbar.total):
            # Run a training-evaluation step
            pbar.set_description("Training...")
            start_time = time.time()
            runner_state, train_metrics, eval_metrics, eval_rollouts = jax.block_until_ready(
                train_and_eval_step_fn(runner_state, eval_problems)
            )
            time_delta = time.time() - start_time

            # Checkpointing
            if self.cfg.logging.checkpoint.training.storing.enabled:
                pbar.set_description("Saving checkpoint...")
                checkpoint_step = self._get_num_updates(runner_state)
                save_checkpoint(checkpoint_manager, runner_state, checkpoint_step)

                if self.cfg.logging.checkpoint.training.storing.upload:
                    pbar.set_description("Uploading checkpoint...")
                    self.logger.log_checkpoint(get_checkpoint_save_dir(self.cfg, checkpoint_step), checkpoint_step)

            # Log the metrics
            pbar.set_description("Logging...")
            is_last_log = step == pbar.total - 1
            self._log_metrics(train_metrics, runner_state, eval_metrics, eval_rollouts, time_delta, is_last_log)

            # Update progress bar
            pbar.update(1)

        # Profiling
        if self.cfg.profile_code:
            jax.profiler.stop_trace()

    def eval(self):
        # Load the runner state from the checkpoint
        runner_state = load_runner_state(
            self.cfg.logging.checkpoint.evaluation.path,
            self._init_runner_state(),
            self.cfg.logging.checkpoint.evaluation.step,
        )

        @functools.partial(jax.jit, static_argnames="num_problems")
        def _eval_aux(problems: EvaluationProblem, num_problems: int):
            eval_rng, agent_init_rng = jax.random.split(self.rng)
            return evaluate_agent(
                self.cfg.evaluation,
                eval_rng,
                runner_state.train_state,
                self.eval_env,
                self.env_params,
                problems,
                num_problems,
                self.hrm_agent.initialize_state(num_problems, agent_init_rng),
            )

        # Load problems
        eval_problems = self._get_eval_problems(runner_state)
        num_eval_problems = eval_problems.hrm.root_id.shape[0]  # use a permanent field to determine the value
        num_batches = jnp.ceil(num_eval_problems / self.EVAL_BATCH_SIZE).astype(int)
        num_saved_rollouts_per_prob = max(1, self.cfg.logging.rollout.num_to_visualize_per_problem)
        rollout_length = self.cfg.logging.rollout.visualization_length if self.cfg.logging.rollout.num_to_visualize_per_problem > 0 else 1

        # Run evaluation
        eval_metrics, eval_rollouts = [], []
        for i in tqdm(range(num_batches), desc="Evaluating..."):
            start_idx = i * self.EVAL_BATCH_SIZE
            end_idx = min(num_eval_problems, start_idx + self.EVAL_BATCH_SIZE)
            
            eval_metrics_iter, eval_rollouts_iter = _eval_aux(
                jax.tree_util.tree_map(lambda x: x[start_idx:end_idx], eval_problems),
                end_idx - start_idx
            )

            # Saving all rollouts is memory-intensive, so only keep as many as we want to visualize
            # (normally 0-1) and the minimum rollout length possible (1).
            eval_rollouts_iter = jax.tree_util.tree_map(
                lambda x: x[:, :num_saved_rollouts_per_prob, ...],
                eval_rollouts_iter
            )
            states, hrm_states = jax.tree_util.tree_map(lambda x: x[:, :, :rollout_length, ...], (eval_rollouts_iter.states, eval_rollouts_iter.hrm_states))
            eval_rollouts_iter = eval_rollouts_iter.replace(
                states=states,
                hrm_states=hrm_states,
                length=jnp.minimum(eval_rollouts_iter.length, rollout_length)
            )

            eval_metrics.append(eval_metrics_iter)
            eval_rollouts.append(eval_rollouts_iter)

        eval_metrics = jax.tree_util.tree_map(lambda *x: jnp.concat(x), *eval_metrics)
        eval_rollouts = jax.tree_util.tree_map(lambda *x: jnp.concat(x), *eval_rollouts)

        # Log the metrics
        print("Logging...", flush=True)
        metrics = self._get_log_eval_metrics(eval_metrics, eval_rollouts, runner_state, step=0, is_last_log=True)[0]
        step = self.cfg.logging.checkpoint.evaluation.step if self.cfg.logging.checkpoint.evaluation.step else 0
        metrics = {
            **metrics,
            "num_env_steps": step * self.num_envs * self.cfg.training.num_steps * self.cfg.training.num_outer_steps,
            "num_updates": step
        }
        self.logger.log_metrics(metrics, step=step)
        self.ms_logger.log_loss(loss=metrics["eval/episode_solve_rate/mean"], mode="val", step=step)

    def _get_eval_problems(self, runner_state: RunnerState) -> EvaluationProblem:
        return self.eval_loader.load()

    def _init_train_runner_state(self) -> RunnerState:
        init_runner_state = self._init_runner_state()
        path = self.cfg.logging.checkpoint.training.loading.path
        if path:
            mode = self.cfg.logging.checkpoint.training.loading.mode
            step = self.cfg.logging.checkpoint.training.loading.step

            if mode == "reset":
                return init_runner_state.replace(
                    train_state=TrainState.create(
                        apply_fn=self.hrm_agent.apply,
                        params=load_runner_state(path, step=step)["train_state"]["params"],
                        tx=self._init_optimizer()
                    )
                )
            elif mode == "continue":
                return load_runner_state(path, init_runner_state, step)
            else:
                raise RuntimeError(f"Error: Unknown checkpoint loading mode '{mode}'.")

        return init_runner_state

    @abstractmethod
    def _init_runner_state(self) -> RunnerState:
        raise NotImplementedError

    def _init_train_state(self, rng: chex.PRNGKey) -> TrainState:
        env_rng, problem_rng, params_rng, agent_rng = jax.random.split(rng, 4)
        
        level, hrm = self.problem_sampler(problem_rng)
        init_timestep = self.training_env.reset(env_rng, self.env_params, level, hrm=hrm)

        # The fields are of shape [B, Timesteps, ...]
        print("Initializing agent...", end=" ", flush=True)
        conditioned_agent_params = jax.jit(self.hrm_agent.init)(
            params_rng,
            jax.tree_util.tree_map(lambda x: x[None, None, ...], init_timestep.observation),
            init_timestep.last()[None, None, ...],
            jax.tree_util.tree_map(lambda x: x[None, None, ...], init_timestep.extras.hrm),
            jax.tree_util.tree_map(lambda x: x[None, None, ...], init_timestep.extras.hrm_state),
            jnp.zeros((1, 1), dtype=jnp.int32),
            jnp.zeros((1, 1)),
            self.hrm_agent.initialize_state(batch_size=1, rng=agent_rng),
        )
        print("Completed.")

        return TrainState.create(
            apply_fn=self.hrm_agent.apply,
            params=conditioned_agent_params,
            tx=self._init_optimizer(),
        )

    def _init_optimizer(self):
        return optax.chain(
            optax.clip_by_global_norm(self.cfg.training.max_grad_norm),
            optax.adam(learning_rate=self._get_learning_rate(), eps=self.cfg.training.adam_eps),
        )

    def _get_learning_rate(self) -> Union[Callable, float]:
        if self.cfg.training.lr_schedule:
            # Linear schedule (not performed for UED in `minimax`)
            # Based on: https://github.com/DramaCow/jaxued/blob/41b693fcf7fa09899ed120b8b6731f3047d2b2cc/examples/craftax/craftax_plr.py#L598
            def _linear_schedule(count):
                frac = (
                    1.0
                    - (count // (self.cfg.training.num_minibatches * self.cfg.training.update_epochs))
                    / (self.num_updates * self.cfg.training.num_outer_steps)
                )
                return self.cfg.training.lr * frac
            return _linear_schedule
        return self.cfg.training.lr

    @abstractmethod
    def _make_train_and_eval_step_fn(self):
        raise NotImplementedError

    def _log_metrics(
        self,
        train_metrics: dict,
        runner_state: RunnerState,
        eval_metrics: RolloutStats,
        rollouts: Rollout,
        time_delta: float,
        is_last_log: bool,
    ):
        # Collect logging information
        num_updates = self._get_num_updates(runner_state)
        log_train = self._get_log_train_metrics(train_metrics, runner_state, num_updates, time_delta)
        log_eval = self._get_log_eval_metrics(eval_metrics, rollouts, runner_state, num_updates, is_last_log)

        # Log into W&B
        for step in sorted(log_train.keys()):
            self.logger.log_metrics({**log_train[step], **log_eval[step]}, step)

        # If using MeinSweeper, we log only the last step of the interval
        self.ms_logger.log_loss(loss=log_train[num_updates]["training/total_loss"], mode="train", step=num_updates)
        self.ms_logger.log_loss(loss=log_eval[num_updates]["eval/episode_solve_rate/mean"], mode="val", step=num_updates)

    @abstractmethod
    def _get_num_updates(self, runner_state: RunnerState) -> int:
        raise NotImplementedError

    def _get_log_train_metrics(self, train_metrics: dict, runner_state: RunnerState, step: int, time_delta: float):
        log_metrics: dict[int, dict] = defaultdict(dict)
        training_dict = {f"training/{k}": v for k, v in train_metrics.items()}

        for update_step in range(self.num_updates_per_iteration - 1):
            log_metrics[step - self.num_updates_per_iteration + update_step + 1] = jax.tree_util.tree_map(lambda x: x[update_step], training_dict)

        # Log number of updates and environment steps
        log_metrics[step] = {
            **jax.tree_util.tree_map(lambda x: x[self.num_updates_per_iteration - 1], training_dict),
            "num_updates": step,
            "num_env_steps": step * self.num_envs * self.cfg.training.num_steps * self.cfg.training.num_outer_steps,
            "num_env_steps_per_sec": self.num_updates_per_iteration * self.num_envs * self.cfg.training.num_steps * self.cfg.training.num_outer_steps / time_delta,
            "update_step_time": time_delta,
        }

        return log_metrics

    def _get_log_eval_metrics(
        self,
        eval_metrics: RolloutStats,
        rollouts: Rollout,
        runner_state: RunnerState,
        step: int,
        is_last_log: bool,
    ):
        num_problems = eval_metrics.length.shape[0]
        return self._get_log_eval_metrics_aux(
            num_problems,
            eval_metrics,
            rollouts,
            step,
            is_last_log,
            problem_names=self.eval_loader.get_problem_names()
        )

    def _get_log_eval_metrics_aux(
        self,
        num_problems: int,
        eval_metrics: RolloutStats,
        rollouts: Rollout,
        step: int,
        is_last_log: bool,
        prefix: str = "eval",
        problem_names: Optional[List] = None,
    ):
        log_metrics: dict[int, dict] = defaultdict(dict)

        # The transformation of rollouts into frame sequences is currently done outside the function
        # because the rendering of the HRMs cannot be jitted (need to export into file then parse)
        levels, hrms = jax.tree_util.tree_map(lambda x: x[:, 0], (rollouts.level, rollouts.hrm))
        log_metrics[step] = {
            f"{prefix}/episode_returns/mean": eval_metrics.reward.mean(axis=1).mean(),
            f"{prefix}/episode_returns_disc/mean": eval_metrics.disc_reward.mean(axis=1).mean(),
            f"{prefix}/episode_lengths/mean": eval_metrics.length.mean(axis=1).mean(),
            f"{prefix}/episode_solve_rate/mean": eval_metrics.is_task_completed.mean(axis=1).mean(),
            f"{prefix}/episode_solve_rate/problem_distribution": plotly_xminigrid_prob_distrib_solving(
                levels,
                hrms,
                eval_metrics.is_task_completed.max(axis=1),  # in any of the rollouts
            ),
        }

        alphabet = self.label_fn.get_str_alphabet()  # costly, best outside the loop
        for eval_problem_id in range(num_problems):
            name = problem_names[eval_problem_id] if problem_names else f"{eval_problem_id}"
            avg_metrics = jax.tree_util.tree_map(
                lambda x: x[eval_problem_id].mean(), eval_metrics
            )
            log_metrics[step].update({
                f"{prefix}/episode_returns/{name}": avg_metrics.reward,
                f"{prefix}/episode_returns_disc/{name}": avg_metrics.disc_reward,
                f"{prefix}/episode_lengths/{name}": avg_metrics.length,
                f"{prefix}/episode_solve_rate/{name}": avg_metrics.is_task_completed,
            })

            show_rollouts = not self.cfg.logging.rollout.show_on_termination_only or is_last_log
            num_rollouts_to_visualize = show_rollouts * min(
                self.cfg.logging.rollout.num_to_visualize_per_problem, self.cfg.evaluation.num_rollouts_per_problem
            )
            for rollout_id in range(num_rollouts_to_visualize):
                rollout_id_str = str(rollout_id).zfill(int(math.log10(num_rollouts_to_visualize)) + 1)
                rollout_name = f"{name}_{rollout_id_str}"
                try:
                    frames = self.renderer.render(
                        jax.tree_util.tree_map(lambda x: x[eval_problem_id, rollout_id], rollouts),
                        self.env_params,
                        alphabet,
                        self.cfg.logging.rollout.visualization_length,
                    )
                    log_metrics[step].update({
                        f"{prefix}/rollouts/{rollout_name}": self.logger.make_video(
                            f"{prefix.replace('/', '')}_{rollout_name}",
                            frames
                        )
                    })
                except CalledProcessError as e:
                    print(f"Error: Graphviz rendering failed [{e}].", flush=True)

        return log_metrics

    def _get_rendered_problem_batch(self, env: Environment, levels: Level, hrms: HRM, num_rendered_problems: int):
        problem_imgs = []
        for i in range(num_rendered_problems):
            # Get the initial state
            level, hrm = jax.tree_util.tree_map(lambda x: x[i], (levels, hrms))
            init_timestep = env.reset(
                jax.random.PRNGKey(0), self.env_params, level, hrm=hrm
            )

            # Simulate the init timestep is a rollout and render it
            problem_img = self.renderer.render(
                Rollout(
                    jax.tree_util.tree_map(lambda x: jnp.asarray(x)[None, ...], init_timestep.state),
                    level,
                    init_timestep.extras.hrm,
                    jax.tree_util.tree_map(lambda x: jnp.asarray(x)[None, ...], init_timestep.extras.hrm_state),
                    length=1
                ),
                self.env_params,
                self.label_fn.get_str_alphabet(),
                max_rollout_length=1
            )
            problem_imgs.append(wandb.Image(problem_img))

        return problem_imgs

    @property
    def num_updates_per_iteration(self):
        return self.cfg.logging.interval
