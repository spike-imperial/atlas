from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, Optional

import imageio.v3 as iio
import numpy as np
from omegaconf import DictConfig, OmegaConf
import tqdm
import wandb
import yaml

WANDB_GQL_PATH = Path(wandb.__file__).parent / "vendor/gql-0.2.0"
sys.path.append(str(WANDB_GQL_PATH))

from wandb_gql import gql


def _download_checkpoint(checkpoint_artifact, checkpoint_dir: str):
    checkpoint_artifact.download(root=os.path.join(checkpoint_dir, str(checkpoint_artifact.metadata["step"])))


def download_artifacts(entity: str, project: str, wandb_run_id: str, checkpoint_dir: str, config_dir: str, step: Optional[int] = None):
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{wandb_run_id}")

    tgt_checkpoint = None
    checkpoint_done = False
    config_done = False

    for artifact in run.logged_artifacts():
        if artifact.type == "checkpoint":
            if step is None:
                if tgt_checkpoint is None or tgt_checkpoint.metadata["step"] < artifact.metadata["step"]:
                    tgt_checkpoint = artifact
            elif step == artifact.metadata["step"]:
                tgt_checkpoint = artifact
        elif artifact.type == "config":
            artifact.download(root=config_dir)
            config_done = True

    if tgt_checkpoint:
        _download_checkpoint(tgt_checkpoint, checkpoint_dir)
        checkpoint_done = True

    if not (checkpoint_done and config_done):
        raise RuntimeError("Error: Couldn't download artifacts for config and checkpoints.")


def download_checkpoints(entity, project, run_id, steps, dst):
    nodes = _get_run_checkpoint_artifacts(entity, project, run_id, steps)
    _download_nodes(entity, project, nodes, dst, max_workers=20)


class WandbLogger:
    VIDEO_FPS = 6

    def __init__(self):
        self._run = None

    def init(self, cfg: DictConfig):
        if cfg.logging.run_id is None:
            run_id = f"{cfg.logging.run_name}--s{cfg.seed}--{datetime.now().strftime('%Y%m%d_%H%M')}"
        else:
            run_id = cfg.logging.run_id

        config = OmegaConf.to_container(cfg, resolve=True)

        # Configure wandb settings for non-interactive environments
        self._run = wandb.init(
            name=cfg.logging.run_name,
            id=run_id,
            resume="allow",
            config=config,
            settings=wandb.Settings(
                _disable_stats=True,
                console="off",
                show_info=False
            ),
            **cfg.logging.wandb,
        )

        # Log the resolved configuration as an artifact
        # For W&B sweeps, the shown config is only shown for the swept
        # hyperparameters, so we need to upload the full config separately
        if cfg.logging.wandb.mode == "online":
            self._log_config_artifact(config)

    def close(self):
        try:
            self._run.finish()
        except AttributeError as e:
            # Specifically catch the isatty attribute error
            if "'LoggerWriter' object has no attribute 'isatty'" in str(e):
                # Try the quiet finish as a fallback
                self._run.finish(quiet=True)
            else:
                # Re-raise if it's a different attribute error
                raise

    def log_metrics(self, data: Dict[str, Any], step: Optional[int] = None):
        self._run.log(data, step=step, commit=True)

    def log_checkpoint(self, path: str, step: int):
        artifact = wandb.Artifact(name=f"checkpoint-{step}", type="checkpoint", metadata={"step": step})
        artifact.add_dir(local_path=path)
        self._run.log_artifact(artifact)

    def _log_config_artifact(self, cfg: DictConfig):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=True) as temp_file:
            yaml.dump(cfg, temp_file)

            artifact = wandb.Artifact(name="config", type="config")
            artifact.add_file(temp_file.name, name="config.yaml")
            self._run.log_artifact(artifact).wait()

    @staticmethod
    def make_video(id: str, frames: np.array) -> wandb.Video:
        """
        Creates a high-quality video file.

        Args:
            frames: a video in `numpy` with shape [time, height, width, channels]
        
        Returns:
            wandb.Video object
        """
        # Create temporary file
        temp_path = os.path.join(tempfile.gettempdir(), f"wandb_video-{os.getpid()}-{id}.mp4")

        try:
            # Convert uint8 if needed
            if frames.dtype != np.uint8:
                frames = (frames * 255).astype(np.uint8)
            
            # Write high quality MP4
            iio.imwrite(temp_path, frames, fps=WandbLogger.VIDEO_FPS)
            return wandb.Video(temp_path, format="mp4")
            
        except Exception as e:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise e


def _get_run_artifacts(entity: str, project: str, wandb_run_id: str):
    api = wandb.Api()
    client = api.client
    query = gql("""
        query ListArtifacts($entity: String!, $project: String!, $run: String!, $get_metadata: Boolean!) {
            project(name: $project, entityName: $entity) {
                run(name: $run) {
                    outputArtifacts {
                        edges {
                            node {
                                id
                                digest
                                versionIndex
                                aliases {
                                    artifactCollectionName
                                    alias
                                }
                                metadata @include(if: $get_metadata)
                            }
                        }
                    }
                }
            }
        }
        """)

    variables = {
        "entity": entity,
        "project": project,
        "run": wandb_run_id,
        "get_metadata": False,
    }

    response = client.execute(query, variable_values=variables)

    if response == {'project': None}:
        print("Unable to find project or entity")
        return []
    elif response == {'project': {'run': None}}:
        print("Unable to find run (or run has no logged artifacts")
        return []

    artifacts = response["project"]["run"]["outputArtifacts"]["edges"]
    return artifacts


def _get_run_checkpoint_artifacts(entity, project, run_id, steps):
    relevant_nodes = []
    for node in _get_run_artifacts(entity, project, run_id):
        artifact = node['node']
        artifact_name = artifact['aliases'][0]['artifactCollectionName']
        if 'checkpoint' in artifact_name:
            if int(artifact_name.split('-')[1]) in steps:
                relevant_nodes.append(node['node'])
    return relevant_nodes


def _node_to_artifact_path(entity, project, node):
    """
    Convert the `node` object returned by your GraphQL query to the
    string spec understood by `wandb.Api().artifact()`, e.g.
    "entity/project/collection:alias"  or  "...:v123".
    """
    # Try the first alias if it exists, otherwise fall back to an explicit version
    if node["aliases"]:
        collection = node["aliases"][0]["artifactCollectionName"]
        alias      = node["aliases"][0]["alias"]
        suffix     = alias
    else:
        collection = "artifacts"                       # <- never used if aliases exist
        suffix     = f"v{node['versionIndex']}"
    return f"{entity}/{project}/{collection}:{suffix}"


def _download_nodes(entity, project, nodes, dst="./downloads", max_workers=8, partial_file=None):
    api = wandb.Api()
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)

    def _fetch(node):
        spec = _node_to_artifact_path(entity, project, node)
        art = api.artifact(spec)

        # Get collection name for subfolder
        if node["aliases"]:
            # save with step as name
            collection = node["aliases"][0]["artifactCollectionName"].split('-')[1]
        else:
            collection = "artifacts"

        # Create subfolder for this collection
        collection_dst = dst / collection
        collection_dst.mkdir(parents=True, exist_ok=True)

        if partial_file is None:
            return art.download(root=collection_dst)  # full download
        else:
            return art.get_path(partial_file).download(root=collection_dst)  # single file

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(
            tqdm.tqdm(
                pool.map(_fetch, nodes),
                total=len(nodes),
                desc="Artifacts"
            )
        )
