import atlas
from pathlib import Path
import sys
atlas_path = Path(atlas.__file__).parent / 'scripts/sweeping'
sys.path.append(str(atlas_path))

import argparse
import jax
import wandb
from omegaconf import DictConfig
from hydra import initialize, compose
from pathlib import Path
import os
import time
from collections import Counter
from functools import reduce
import operator
from typing import Dict, List, Optional
import logging
from datetime import datetime
from jax.lib import xla_bridge
import gc

from atlas.runners.dr_hrm_ppo_runner import DRHRMConditionedPPORunner
from atlas.runners.plr_hrm_ppo_runner import PLRHRMConditionedPPORunner
from atlas.utils.resolvers import register_resolvers


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_short_name(sweep_config, max_length=128):
    # Dictionary of common abbreviations
    abbreviations = {
        'False': 'f',
        'True': 't',
        'capacity': 'cp',
        'cond_aggregation': 'ca',
        'conditioner': 'c',
        'conditioner/literal_embedder': 'lt',
        'd_feat': 'f',
        'd_hidden': 'h',
        'enforce_sequentiality': 's',
        'ent_coef': 'ent',
        'exploratory_grad_updates': 'g',
        'gae_lambda': 'gae',
        'gamma': 'gm',
        'head_hidden_dim': 'hhd',
        'hrm_sampler_wrapper': 'hw',
        'level_id': 'lv',
        'lr': 'lr',
        'max_num_rms': 'rm',
        'min_eps_log': 'me',
        'num_envs': 'ev',
        'num_layers': 'ly',
        'num_outer_steps': 'o',
        'num_steps': 'st',
        'replay_prob': 'r',
        'reward_agg': 'a',
        'reward_shaping': 'r',
        'rgcn': 'rgcn',
        'rnn_cell_type': 'rct',
        'rnn_hidden_dim': 'rhd',
        'score_function': 's',
        'seed': 's',
        'shortest_path_reward': 'spr',
        'staleness_coeff': 'st',
        'temperature': 't',
        'update_epochs': 'ep',
        'use_call_compat_matrix': 'c',
        'use_layer_norm': 'l',
        'use_next_to_props': 'p',
        'use_transition_compat_matrix': 't',
        'xminigrid_hrm_cond': 'hrm',
        'xminigrid_level_cond': 'lvl',
        'xminigrid_ortho': 'ortho',
    }

    # Function to abbreviate a key
    def abbreviate_key(key):
        parts = key.split('.')
        return abbreviations.get(parts[-1], parts[-1][:3])

    def abbreviate_val(val):
        return abbreviations.get(str(val), val)

    # Create name components
    components = [f"{abbreviate_key(k)}_{abbreviate_val(v)}" for k, v in sweep_config.items()]

    # Join components
    name = "-".join(components)

    # If name is too long, start truncating the least important parts
    if len(name) > max_length:
        while len(name) > max_length and components:
            # Remove the middle component (least important)
            middle = len(components) // 2
            components.pop(middle)
            name = "-".join(components)

    return name.replace("[", "").replace("]", "")


def format_box(title: str, content: List[str], width: int = 80) -> str:
    """Create a boxed format for logging messages"""
    horizontal_line = "─" * width
    result = [
        f"\n┌{horizontal_line}┐",
        f"│ {title.center(width-2)} │"
    ]
    if content:
        result.append(f"├{horizontal_line}┤")
        for line in content:
            result.append(f"│ {line:<{width-2}} │")
    result.append(f"└{horizontal_line}┘")
    return "\n".join(result)


class WandBSweepAgent:
    """A class to manage and execute W&B sweeps with monitoring capabilities."""
    
    def __init__(
        self,
        project: str,
        entity: str,
        check_interval: int = 300,
        max_runs_per_sweep: Optional[int] = None,
        gpu_device: Optional[int] = None
    ):
        """
        Initialize the W&B sweep agent.
        
        Args:
            project: W&B project name
            entity: W&B entity name
            check_interval: Time in seconds between sweep checks
            max_runs_per_sweep: Maximum number of runs to execute per sweep
            gpu_device: GPU device index to use. If None, will use device 0.
        """
        self.project = project
        self.entity = entity
        self.check_interval = check_interval
        self.max_runs_per_sweep = max_runs_per_sweep
        self.api = wandb.Api()
        
        # GPU device management
        self.gpu_device = gpu_device if gpu_device is not None else 0
        if self.gpu_device >= jax.local_device_count():
            raise ValueError(f"GPU device {self.gpu_device} not available. Only {jax.local_device_count()} devices found.")
        
        # Job tracking
        self.job_counter = 0

    @staticmethod
    def get_project_root():
        """Get the root directory of the project using the package location"""
        return Path(atlas.__file__).parent.parent

    def _get_parameter_combinations(self, sweep_config: Dict) -> int:
        """Calculate total number of parameter combinations in a sweep"""
        parameters = sweep_config.get('parameters', {})
        
        # Count combinations for each parameter
        param_counts = []
        for param, config in parameters.items():
            if 'values' in config:
                param_counts.append(len(config['values']))
            elif all(k in config for k in ['min', 'max', 'distribution']):
                param_counts.append(config.get('count', 1))
        
        return 1 if not param_counts else reduce(operator.mul, param_counts)

    def get_sweep_stats(self, sweep) -> Dict:
        """Get statistics about runs in a sweep"""
        runs = sweep.runs
        
        # Count runs by status
        status_counts = Counter(run.state for run in runs)
        
        # Calculate total parameter combinations
        total_combinations = self._get_parameter_combinations(sweep.config)
        
        # Get actual run counts
        finished = status_counts.get('finished', 0)
        running = status_counts.get('running', 0)
        failed = status_counts.get('failed', 0)
        crashed = status_counts.get('crashed', 0)
        
        # Calculate pending as total combinations minus completed runs
        pending = max(0, total_combinations - (finished + failed + crashed))
        
        return {
            'total_combinations': total_combinations,
            'total_runs': len(runs),
            'finished': finished,
            'running': running,
            'failed': failed,
            'crashed': crashed,
            'pending': pending
        }

    def find_active_sweeps(self) -> List[Dict]:
        """Find sweeps that still have pending runs"""
        sweeps = self.api.project(self.project, entity=self.entity).sweeps()
        
        active_sweeps = []
        for sweep in sweeps:
            # Get stats first to check for pending runs
            stats = self.get_sweep_stats(sweep)
            
            # Consider a sweep active if:
            # 1. It's in "running" state AND
            # 2. Has pending runs AND
            # 3. Not all runs are failed/crashed
            print(sweep.state, stats)
            if (sweep.state == "RUNNING" and 
                (stats['pending'] + stats['failed'] + stats['crashed']) > 0):
                active_sweeps.append({
                    'id': sweep.id,
                    'name': sweep.name,
                    'state': sweep.state,
                    'stats': stats
                })
        
        return active_sweeps

    def execute_run(self, cfg: DictConfig) -> None:
        """Execute a single run with the given configuration"""
        if cfg.algorithm.name == "dr":
            runner = DRHRMConditionedPPORunner(cfg)
        elif cfg.algorithm.name == "plr":
            runner = PLRHRMConditionedPPORunner(cfg)
        else:
            raise RuntimeError(f"Error: Unknown runner `{cfg.runner}`.")

        logger.info(f"Num. Devices: {jax.local_device_count()}")
        if cfg.mode == "training":
            logger.info(f"Num. Training Updates: {runner.num_updates}")

        runner.run()

    def run_sweep_function(self):
        """Main run function that integrates with W&B"""
        with wandb.init() as run:
            project_root = self.get_project_root()
            
            # Clean and format the config path for Hydra
            config_path = run.config.get('config-path', '').replace('../', '')
            config_name = run.config.get('config-name', 'config')
            
            # Format overrides properly
            overrides = [
                f"hydra.searchpath=[{project_root}/config]",
                f"+experiments={config_path}/{config_name}"
            ]
            
            # w&b config contains the overrides:
            for k, v in run.config.items():
                if k not in ["config-path", "config-name"]:
                    overrides.extend([f"{k}={v}"])

            with initialize(version_base=None, config_path=os.path.join("../../", config_path)):
                cfg = compose(config_name="config", overrides=overrides)

            # Create alert message with run details
            alert_msg = (
                f"Starting sweep run with:\n"
                f"Name: {run.name}\n"
                f"Overrides:\n" + "\n".join(f"  {override}" for override in overrides 
                                          if not override.startswith("hydra.searchpath") 
                                          and not override.startswith("+experiments"))
            )
            
            # Log alert to wandb
            run.alert(
                title="Starting Sweep Run",
                text=alert_msg,
                level=wandb.AlertLevel.INFO
            )
            
            # Override the run names
            naming_params = {k:v for k,v in run.config.items() if k not in ["config-path", "config-name"]}
            cfg.logging.run_name = create_short_name(naming_params)
            cfg.logging.run_id = run.id
            cfg.logging.wandb.group = f"sweep_{run.sweep_id}"
            run.name = cfg.logging.run_name

            # Execute the run
            self.execute_run(cfg)

    def _get_next_job_id(self) -> int:
        self.job_counter += 1
        return self.job_counter

    def _clear_gpu_memory(self):
        """Clear JAX GPU memory"""
        backend = xla_bridge.get_backend()
        for buf in backend.live_buffers():
            buf.delete()
        gc.collect()

    def _format_sweep_config(self, sweep) -> List[str]:
        """Format sweep configuration for logging"""
        config = sweep.config
        formatted_lines = []
        
        # Add method and metric info
        formatted_lines.extend([
            f"Method: {config.get('method', 'N/A')}",
            f"Metric: {config.get('metric', {}).get('name', 'N/A')} ({config.get('metric', {}).get('goal', 'N/A')})"
        ])
        
        # Add parameters
        params = config.get('parameters', {})
        if params:
            formatted_lines.append("")
            formatted_lines.append("Parameters:")
            for param_name, param_config in params.items():
                if 'values' in param_config:
                    values = param_config['values']
                    if len(values) > 4:
                        values = values[:3] + ['...']
                    formatted_lines.append(f"  {param_name}: {values}")
                elif all(k in param_config for k in ['min', 'max']):
                    formatted_lines.append(
                        f"  {param_name}: range({param_config['min']}, {param_config['max']}, "
                        f"{param_config.get('distribution', 'uniform')})"
                    )
                else:
                    formatted_lines.append(f"  {param_name}: {param_config}")
        
        return formatted_lines

    def _format_job_start(self, sweep, job_id: int, count: Optional[int]) -> str:
        """Format job start information"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        content = [
            f"Job ID: {job_id}",
            f"Started at: {timestamp}",
            f"GPU: {self.gpu_device}",
            "",
            f"Sweep Name: {sweep.name}",
            f"Sweep ID: {sweep.id}",
            f"Runs to execute: {'all' if count is None else count}",
            ""
        ]
        content.extend(self._format_sweep_config(sweep))
        return format_box(f"Starting New Job", content)

    def run_sweep(self, sweep_id: str, count: Optional[int] = None):
        """Run a sweep with optional count limit"""
        try:
            # Get sweep information
            sweep = self.api.sweep(f"{self.entity}/{self.project}/{sweep_id}")
            job_id = self._get_next_job_id()
            
            # Log job start
            logger.info(self._format_job_start(sweep, job_id, count))
            
            # Set environment variable for JAX to use specific GPU
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_device)
            
            # Run the sweep
            wandb.agent(
                sweep_id,
                function=self.run_sweep_function,
                count=count,
                entity=self.entity,
                project=self.project
            )
            
            # Clear GPU memory
            self._clear_gpu_memory()
            
            # Log job completion
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info(format_box(
                f"Job {job_id} Completed",
                [
                    f"Completed at: {timestamp}",
                    f"Sweep ID: {sweep_id}",
                    f"GPU: {self.gpu_device} released and memory cleared"
                ]
            ))
            
        except Exception as e:
            # Log job failure
            logger.error(format_box(
                f"Job {job_id} Failed",
                [
                    f"Error: {str(e)}",
                    f"Sweep ID: {sweep_id}",
                    f"GPU: {self.gpu_device}"
                ]
            ))
            # Try to clear GPU memory even on failure
            try:
                self._clear_gpu_memory()
            except:
                pass
            raise

    def print_sweep_stats(self, sweep: Dict):
        """Print statistics for a sweep in a formatted way"""
        stats = sweep['stats']
        content = [
            f"Name: {sweep['name']}",
            f"State: {sweep['state']}",
            "",
            "Statistics:",
            f"  Total Combinations: {stats['total_combinations']}",
            f"  Finished: {stats['finished']}",
            f"  Running: {stats['running']}",
            f"  Failed: {stats['failed']}",
            f"  Crashed: {stats['crashed']}",
            f"  Pending: {stats['pending']}"
        ]
        logger.info(format_box(f"Sweep {sweep['id']}", content))

    def monitor_and_run_sweeps(self, continuous: bool = False):
        """
        Monitor and run sweeps continuously or once.
        
        Args:
            continuous: Whether to run continuously or exit after all current sweeps are done
        """
        # Log agent start
        logger.info(format_box(
            "W&B Sweep Agent Started",
            [
                f"Project: {self.project}",
                f"Entity: {self.entity}",
                f"GPU Device: {self.gpu_device}",
                f"Check Interval: {self.check_interval}s",
                f"Mode: {'Continuous' if continuous else 'One-time'}"
            ]
        ))
        
        while True:
            active_sweeps = self.find_active_sweeps()
            
            if not active_sweeps:
                if continuous:
                    logger.info(format_box(
                        "No Active Sweeps",
                        [f"Checking again in {self.check_interval} seconds..."]
                    ))
                    time.sleep(self.check_interval)
                    continue
                else:
                    logger.info(format_box("Agent Stopping", ["No active sweeps found"]))
                    break
            
            logger.info(format_box(f"Active Sweeps Found", [f"Count: {len(active_sweeps)}"]))
            for sweep in active_sweeps:
                self.print_sweep_stats(sweep)
                
                # Start agent for this sweep
                count = min(sweep['stats']['pending'], self.max_runs_per_sweep) if self.max_runs_per_sweep else None
                if count != 0:
                    self.run_sweep(sweep['id'], count)
            
            if not continuous:
                logger.info(format_box("Agent Completed", ["All sweeps processed"]))
                break
            
            logger.info(format_box(
                "Waiting for Next Check",
                [f"Interval: {self.check_interval} seconds"]
            ))
            time.sleep(self.check_interval)


def main():
    # Register custom resolvers
    register_resolvers()
    
    parser = argparse.ArgumentParser(description="Run W&B sweeps with monitoring capabilities")
    parser.add_argument("--project", type=str, required=True, help="W&B project name")
    parser.add_argument("--entity", type=str, required=True, help="W&B entity name")
    parser.add_argument("--sweep_id", type=str, help="Specific W&B sweep ID to run")
    parser.add_argument("--count", type=int, default=None, help="Number of runs to execute for a specific sweep")
    parser.add_argument("--monitor", action="store_true", help="Monitor and run all active sweeps")
    parser.add_argument("--continuous", action="store_true", help="Continuously monitor for new sweeps")
    parser.add_argument("--check_interval", type=int, default=300, help="Seconds between sweep checks in monitor mode")
    parser.add_argument("--max_runs_per_sweep", type=int, default=None, help="Maximum runs to execute per sweep")
    parser.add_argument("--gpu_device", type=int, default=0, help="GPU device index to use")
    
    args = parser.parse_args()
    
    # Create the sweep agent
    agent = WandBSweepAgent(
        project=args.project,
        entity=args.entity,
        check_interval=args.check_interval,
        max_runs_per_sweep=args.max_runs_per_sweep,
        gpu_device=args.gpu_device
    )
    
    if args.sweep_id:
        # Run a specific sweep
        agent.run_sweep(args.sweep_id, args.count)
    elif args.monitor or args.continuous:
        # Monitor and run sweeps
        agent.monitor_and_run_sweeps(continuous=args.continuous)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 