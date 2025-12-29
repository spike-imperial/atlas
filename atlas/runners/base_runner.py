from abc import ABC, abstractmethod
import math

from hydra.utils import instantiate
import jax
from meinsweeper import MSLogger
from omegaconf import DictConfig

from ..utils.logging import WandbLogger


class Runner(ABC):
    def __init__(self, cfg: DictConfig):
        assert jax.local_device_count() == 1, "Error: Only one device is currently supported"

        self.cfg = cfg

        self.rng = jax.random.PRNGKey(cfg.seed)

        self.env = instantiate(cfg.env)
        self.env_params = instantiate(cfg.env_params)

        self.logger = WandbLogger()
        self.ms_logger = MSLogger()

    def run(self):
        self.logger.init(self.cfg)
        self.ms_logger.log_loss(loss=0.0, mode='train', step=0)

        if self.cfg.mode == "training":
            self.train()
        elif self.cfg.mode == "evaluation":
            self.eval()
        else:
            print(f"Error: Unknown mode `{self.cfg.mode}`.")
        
        self.logger.close()

    @abstractmethod
    def train(self):
        raise NotImplementedError
    
    @abstractmethod
    def eval(self):
        raise NotImplementedError

    @property
    def num_envs(self) -> int:
        return self.cfg.training.num_envs

    @property
    def num_updates(self) -> int:
        train_cfg = self.cfg.training
        if train_cfg.num_updates is not None:
            return train_cfg.num_updates
        return math.ceil(
            train_cfg.total_timesteps /
            (train_cfg.num_steps * train_cfg.num_outer_steps * train_cfg.num_envs)
        )

    @property
    def total_timesteps(self) -> int:
        train_cfg = self.cfg.training
        if self.cfg.training.total_timesteps is not None:
            return self.cfg.training.total_timesteps
        return train_cfg.num_updates * train_cfg.num_steps * train_cfg.num_outer_steps * train_cfg.num_envs
