# ATLAS (Aligning Tasks and Levels for Autocurricula of Specifications)

Code for the paper [_"Beyond Fixed Tasks: Unsupervised Environment Design for Task-Level Pairs"_](https://arxiv.org/abs/2511.12706) 
by Daniel Furelos-Blanco, Charles Pert, Frederik Kelbel, Alex F. Spies, Alessandra Russo, and Michael Dennis.  
Published at the *AAAI Conference on Artificial Intelligence (AAAI), 2026*.

## Table of Contents
1. [Installation](#installation)
2. [Source Code](#source-code)
3. [Problem Sets](#problem-sets)
4. [Experiments](#experiments)
5. [Notebooks and Scripts](#notebooks-and-scripts)
6. [Tests](#tests)
7. [Citation](#citation)

## Installation

### Prerequisites
- [Miniforge3](https://github.com/conda-forge/miniforge) (or any Conda distribution).
- For GPU support: CUDA 12.

### Setup
1. Install Miniforge (if not already installed): 
```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
```
2. Activate it:
```bash
source miniforge3/bin/activate
```
3. Create the environment (Python 3.10) and activate it:
```bash
conda create --name atlas python=3.10 ffmpeg graphviz -c conda-forge
conda activate atlas
```
4. Install the package and requirements:
```bash
cd atlas

# For GPU (CUDA 12)
pip install -e . --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# For CPU only:
# Edit requirements.txt first: change jax[cuda12] to jax[cpu]
# Then run: pip install -e .
```
5. Configure [Weights & Biases](https://docs.wandb.ai/quickstart/):
```bash
wandb login
```

## Source Code
The source code is contained within the `atlas` folder. We describe the different main folders below.

### `agents`
High-level implementation of the agents used in the experiments, see `hrm_conditioned_agent.py` specifically:
the input HRM(s) are embedded and passed as an input to an RNN-based agent.

### `conditioners`
Implements different HRM-embedding strategies:
* `dummy` does nothing, outputs an empty embedding.
* `vanilla` outputs a one-hot embedding indicating the current RM state.
* `rgcn` outputs the embedding for the current RM state using a graph convolutional network over the RM graph.

### `envs`
Implements the environment-related components, including:
* **Levels and level sampling.** Specify the parameters of each level and methods for sampling them (mainly focusing
on sampling levels with different numbers of rooms).
* **Literal embedding.** Implements domain-specific methods for embedding the literals labeling the formulas of an HRM.
* **Mutators.** Implements domain-specific mutation operators (edits).
* **Networks.** Implements the network for the Minigrid environment.
* **Problem Sampling.** Implements the level-conditioned and HRM-conditioned problem sampling strategies. 
* **Labeling Function.** Implements the mapping from environment observations to sets of propositions.
* **Renderer.** Implements operations for rendering levels and environment observations.

Additional details can be found in the comments and in the submitted paper.
The Minigrid implementation wraps the XLand-Minigrid one by [Nikulin et al. (2024)](https://arxiv.org/pdf/2312.12044).

### `eval_loaders`
Implements different methods for loading a validation set for training.

### `hrm`
Implements the HRM formalism as well as different samplers. Here, `single_path_flat` stands for the sequential sampler
in the paper.

### `networks`
Implements the RNN and a generic actor-critic architecture used in the implemented environments.

### `problem_samplers`
Implements the base problem sampler, which samples tasks and levels independently.

### `runners`
Implements the UED algorithms used in the paper: DR and PLR. ACCEL is implicitly implemented within PLR.

### `ued`
Implements some UED-related functions: the replay buffer and different scoring functions.

The buffer includes some modifications from the JaxUED implementation by [Coward et al. (2024)](https://github.com/DramaCow/jaxued),
including sampling without replacement, and tie-breaking and the use of _lower or equal_ 
comparisons to decide the problem to substitute upon insertion. Further, we experimented
with different scoring functions, and interpolations of them. 

The default configuration (with replacement, no tie-breaking, _strictly lower_ comparison,
single scoring with MaxMC and PVL) was used in all paper experiments, as the variants did not 
yield significant improvements. However, we retain these alternative implementations as they may 
be useful for future work or different domains.

### `utils`
Implements some auxiliary functions for checkpointing, evaluation (rollouts), logging (to Weights and Biases),
math operations, plotting (diagnostics uploaded to Weights and Biases), rendering rollouts and training (e.g. PPO).

## Problem Sets
The `problems` directory contains the different problem sets involved in the paper:
- `00-validation-set` is the validation set, the only one used at training time.
- `01-cvar-sequential` is the set for computing the CVaR for problems generated with the sequential sampler.
- `02-cvar-dags` is the set for computing the CVaR for problems generated with the random walk-based sampler.
- `03-hand-designed` is the set containing the 150 hand-designed problems. The `_rendered` directory contains the illustrations for the levels and the HRMs.

The validation and CVaR sets are automatically generated and filtered to ensure they 
contain solvable problems (i.e., problems where the task can be realized in the level). 
See Appendix E.3 for details on the solvability checking approach.

## Experiments
We describe how to run the experiments to reproduce the results in the paper, run the evaluations from the training
runs and produce the final plots.

### Description
The `experiments` directory contains a folder for each set of experiments in the paper.
All experiments are determined using configuration files building on the structure determined in the `config` folder.
The experiments are the following:
- `00-sweep` are the initial sweeps to refine some of the hyperparameters (see Appendix E.1).
  - `from-full` correspond to experiments using sampling from the full training distribution (PLR, ACCEL).
  - `from-scratch` correspond to experiments using sampling from the simple problem distribution (ACCEL-0).
- `01-core` are the experiments for the main results (Section 5.2, Appendix E.4) and the problem sampling ablations
  (Section 5.3, Appendix E.5).
- `02-vanilla-conditioning` are the ablation experiments using the vanilla conditioner, i.e. conditioning on the RM state id
  (see Appendix E.8).
- `03-myopic` are the ablation experiments using the graph neural network with a single layer (see Appendix E.8).
- `04-domain-independent-literal-embeddings` are the ablation experiments using domain independent literal embeddings, i.e.
  not exploiting the proposition structure (see Appendix E.8).
- `05-num-mutations` are the ablation experiments analyzing shorter and longer edit sequences (Section 5.5, Appendix E.7).
- `06-mutation-types` are the ablation experiments where some edit types are removed (Section 5.5, Appendix E.7).
- `07-dag-sampling` are the ablation experiments on the task sampling, where the default sequential sampler is substituted
  with a random walk-based sampler that produces RMs as directed acyclic graphs (see Section 5.4, Appendix E.6).
- `08-pvl` are the ablation experiments on the scoring function, where PVL is used instead of MaxMC (Appendix E.9).

### Execution
To run any of the experiments above, follow these steps:
1. Find a file starting with `sweep` corresponding to the experiment to run, e.g. `experiments/training/01-core/plr/sweep.yaml`.
2. Open the file and fill the `entity` field with the W&B entity where you want to log the results. Do the same with the corresponding `config.yaml`.
3. Run the command `wandb sweep experiments/training/01-core/plr/sweep.yaml` (using the path to the sweep path you chose). This
will create a new sweep in your W&B. The output should be something like the following (the sweep ID will be different):
```bash
wandb: Creating sweep from: experiments/training/01-core/plr/sweep.yaml
wandb: Creating sweep with ID: 8h61u6kz
wandb: View sweep at: https://wandb.ai/YOUR_ENTITY/atlas/sweeps/8h61u6kz
wandb: Run sweep agent with: wandb agent YOUR_ENTITY/atlas/8h61u6kz
```
4. The next step is to launch an experiment picked from the sweep using the following command (you can queue a sequence by setting `count` higher than 1):
```bash
python scripts/sweeping/run_wandb_sweep.py --sweep_id YOUR_SWEEP_ID --count 1 --project atlas --entity YOUR_ENTITY
```

### Evaluation on the CVaR and Hand-Designed Sets
Once the training runs have been completed in the step above, it is time to run the evaluations
on the CVaR sets and the hand-designed set. The evaluations will also be logged into W&B.
**WARNING:** Note that you will need to edit the W&B run identifiers from all files to yours.

#### CVaR Sets
Run the following command for _sequential_ and _random walk-based_ sampling. 
```bash
python experiments/evaluation/cvar/data_collection/run_eval_seq_cond_set.py
python experiments/evaluation/cvar/data_collection/run_eval_rw_cond_set.py
```

Once the evaluation is complete, the results from W&B can be dumped into `.csv` files using:
```bash
python experiments/evaluation/cvar/data_collection/dump_eval_cond_set.py
```

The results are currently dumped in the file `experiments/evaluation/cvar/data_collection/cvar_seq.csv`
and `experiments/evaluation/cvar/data_collection/cvar_rw.csv`.

#### Hand-Designed Evaluation Set
To evaluate performance only at the end of training:
```bash
python experiments/evaluation/handcrafted/data_collection/run_eval_last_checkpoint.py
```

To evaluate different checkpoints through training (to later produce the learning curve):
```bash
python experiments/evaluation/handcrafted/data_collection/run_eval_checkpoint_seq.py
```

### Producing the Plots from Evaluation and Training Data
#### Curriculum Analysis
The notebook `experiments/evaluation/curriculum/curriculum.ipynb` produces the plots for the curriculum analysis
shown in Figures 5, 19 and 22. **WARNING:** the buffer data is already dumped into `.csv` files (`seq_buffer_data.csv`
and `rw_buffer_data.csv`), so there is no need to dump the checkpoints, which occupy a lot of space.

#### CVaR Evaluation
The script `experiments/evaluation/cvar/plot_cvar_cond.py` produces the CVaR plots shown in Figures 3a, 18a and 21a.

#### Generated Samples
The script `experiments/evaluation/generated_samples/render_generated_samples.py` dumps some samples generated
samples by the different algorithms at different times during training. This is done directly from artifacts in
W&B, so existing runs are needed.

#### Hand-Designed Evaluation
To produce the _learning curve_ plots (Figures 3b and 18b), run the following command:
```bash
python experiments/evaluation/handcrafted/aggregate/iqm_curve.py
```

To produce the _IQM solve rate at the end of training_ plots (Figures 18c, 21b, 26-28), run the following command:
```bash
python experiments/evaluation/handcrafted/aggregate/iqm.py
```

To produce the _solve rate per problem_ plots (Figure 4), run the following command:
```bash
python experiments/evaluation/handcrafted/per_problem/per_instance.py
```

To produce the _solve rate per problem_ tables (Tables 5-7), run the following command:
```bash
python experiments/evaluation/handcrafted/per_problem/latex_table.py
```

#### Mutations
To obtain how the presence of mutations evolves in the buffer over time (Figure 17),
run the notebook `experiments/evaluation/mutations/mutations.ipynb`. It requires
substituting the W&B run identifiers.

#### Solvability
To obtain the _solvability over time_ (Figures 12, 20, 25), run the following command:
```bash
python experiments/evaluation/solvability/buffer_over_time/plot_solvability_over_time.py
```
The input `.csv`, which is already provided, can be obtained using the `gen_solvability_over_time_jobs.py` script that
generates jobs in a PBS cluster. Alternatively, you can run `eval_solvability_over_time.py` for each instant run and
desired timestep.

To obtain the _percent of solvable problems per batch_ (Tables 3-4), run the following command:
```bash
python experiments/evaluation/solvability/rate_per_batch/eval_solvable_per_batch.py
```

## Notebooks and Scripts
The `notebooks` directory contains several Jupyter notebooks that exemplify how to use
environments and HRMs:
- `env/01 Environment Interaction` shows how to run a rollout for a level-HRM pair.
- `hrms/01 HRM Example` shows several steps across a complex HRM.
- `hrms/02 HRM Sampling Example` shows the use of sequential and random walk-based samplers.
- `hrms/03 HRM Rendering Example` shows how to render HRMs.
- `hrms/04 HRM Traversal Speed` performs some tests on the speed traversal of different HRMs.
- `problems/01 Problem Sampling` exemplifies different types of task-level sampling strategies (independent, level-conditioned, HRM-conditioned).

The `scripts/xminigrid/manual_control.py` script enables interacting with the environment
via keyboard, moving an agent in randomly sampled task-level pairs. **WARNING:** The first
step will take a bit long since the JAX-compilation of the function will be happening at that
time.

## Tests
The `tests` directory contains some automated tests:
- `hrm/test` verifies that HRM traversals are correctly done across diverse HRMs.
- `test_conditioner` verifies the correctness of different conditioning strategies. **WARNING:** The
check for RM embedding correctness employs a prototype implementation of HRM embeddings involving more
than one RM. Note that in the paper we examine HRMs with a single RM.
- `test_wrappers` verifies the HRM wrappers work correctly.
- `test_xminigrid_labeling_function` verifies the correctness of the labeling function for Minigrid.

## Citation
If you use this code in your research, please cite our paper:
```bibtex
@inproceedings{FurelosBlancoPKSRD26,
  author = {Furelos-Blanco, Daniel and Pert, Charles and Kelbel, Frederik and Spies, Alex F. and Russo, Alessandra and Dennis, Michael},
  title = {{Beyond Fixed Tasks: Unsupervised Environment Design for Task-Level Pairs}},
  booktitle = {{AAAI} Conference on Artificial Intelligence (AAAI)},
  year = {2026},
}
```
