import hydra
import jax
from omegaconf import DictConfig

from atlas.runners.dr_hrm_ppo_runner import DRHRMConditionedPPORunner
from atlas.runners.plr_hrm_ppo_runner import PLRHRMConditionedPPORunner
from atlas.utils.resolvers import register_resolvers


@hydra.main(version_base=None, config_path="../config", config_name="config")
def run(cfg: DictConfig) -> None:
    if cfg.algorithm.name == "dr":
        runner = DRHRMConditionedPPORunner(cfg)
    elif cfg.algorithm.name == "plr":
        runner = PLRHRMConditionedPPORunner(cfg)
    else:
        raise RuntimeError(f"Error: Unknown runner `{cfg.runner}`.")

    print(f"Num. Devices: {jax.local_device_count()}")
    if cfg.mode == "training":
        print(f"Num. Training Updates: {runner.num_updates}")

    runner.run()


if __name__ == "__main__":
    register_resolvers()
    run()
