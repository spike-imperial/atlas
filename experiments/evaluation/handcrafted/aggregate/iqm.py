import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils
import seaborn as sns
import yaml


def _make_plot(eval_data_df, experiment_names, labels, xlabel_y_coordinate, path, out_filename, colors=None):
    # Reverse to make items appear in order
    experiment_names.reverse()
    labels.reverse()

    # Compile results using formatting required by the rliable library
    results = {}
    for e, l in zip(experiment_names, labels):
        cols_to_keep = eval_data_df.columns.difference(["experiment", "training_run_id", "mean"])
        df_exp = eval_data_df[eval_data_df["experiment"] == e]
        df_exp = df_exp[cols_to_keep]
        results[l] = df_exp.to_numpy()

    # Compute IQM and export results
    aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
        results, lambda x: np.array([metrics.aggregate_iqm(x)])
    )
    with open(os.path.join(path, f"{out_filename}.yaml"), 'w') as f:
        yaml.safe_dump({k: float(v[0]) for k, v in aggregate_scores.items()}, f)

    # Make the figure
    fig, axes = plot_utils.plot_interval_estimates(
        aggregate_scores,
        aggregate_score_cis,
        metric_names=['IQM'],
        algorithms=labels,
        colors=colors,
        max_ticks=5,
        xlabel_y_coordinate=xlabel_y_coordinate,
        xlabel='IQM Solve Rate',
    )
    axes.set_title("")
    axes.set_xlim(-0.01, 1.01)
    plt.savefig(os.path.join(path, f"{out_filename}.pdf"), bbox_inches='tight')


if __name__ == "__main__":
    dir_name = os.path.join(os.path.dirname(sys.argv[0]), "results")
    os.makedirs(dir_name, exist_ok=True)
    eval_data_df = pd.read_csv(os.path.join(dir_name, "..", "..", "data_collection", "eval_last_checkpoint.csv"))

    # Classic colors used for DR, PLR, ACCEL, ACCEL-0
    set2 = sns.color_palette("Set2", 8)
    base_colors = {"DR": set2[3], "PLR$^\\bot$": set2[0], "ACCEL": set2[1], "ACCEL-0": set2[2]}

    # Sequential sampler + Independent sampling
    _make_plot(
        eval_data_df,
        experiment_names=["seq-dr-i", "seq-plr-i", "seq-accel_full-i", "seq-accel_scratch-i"],
        labels=["DR", "PLR$^\\bot$", "ACCEL", "ACCEL-0"],
        xlabel_y_coordinate=-0.3,
        colors=base_colors,
        path=dir_name,
        out_filename="iqm_seq_indep"
    )

    # Sequential sampler + Conditioned sampling
    _make_plot(
        eval_data_df,
        experiment_names=["seq-dr-c", "seq-plr-c", "seq-accel_full-c", "seq-accel_scratch-c"],
        labels=["DR", "PLR$^\\bot$", "ACCEL", "ACCEL-0"],
        xlabel_y_coordinate=-0.3,
        colors=base_colors,
        path=dir_name,
        out_filename="iqm_seq_cond"
    )

    # Sequential sampler + Independent and Conditioned sampling
    _make_plot(
        eval_data_df,
        experiment_names=[
            "seq-dr-i", "seq-plr-i", "seq-accel_full-i", "seq-accel_scratch-i",
            "seq-dr-c", "seq-plr-c", "seq-accel_full-c", "seq-accel_scratch-c"
        ],
        labels=[
            "DR$_\\text{indep}$", "PLR$^\\bot_\\text{indep}$", "ACCEL$_\\text{indep}$", "ACCEL-0$_\\text{indep}$",
            "DR$_\\text{cond}$", "PLR$^\\bot_\\text{cond}$", "ACCEL$_\\text{cond}$", "ACCEL-0$_\\text{cond}$"
        ],
        xlabel_y_coordinate=-0.1,
        path=dir_name,
        out_filename="iqm_seq_indep_cond"
    )

    # Random walk sampler + Independent and Conditioned sampling
    _make_plot(
        eval_data_df,
        experiment_names=["rw-dr-i", "rw-plr-i", "rw-dr-c", "rw-plr-c", "seq-dr-i", "seq-plr-i", "seq-dr-c", "seq-plr-c"],
        labels=[
            "DR$_\\text{dag,indep}$",
            "PLR$^\\bot_\\text{dag,indep}$",
            "DR$_\\text{dag,cond}$",
            "PLR$^\\bot_\\text{dag,cond}$",
            "DR$_\\text{seq,indep}$",
            "PLR$^\\bot_\\text{seq,indep}$",
            "DR$_\\text{seq,cond}$",
            "PLR$^\\bot_\\text{seq,cond}$",
        ],
        path=dir_name,
        xlabel_y_coordinate=-0.1,
        out_filename="iqm_rw"
    )

    # ACCEL ablations - Number of edits
    _make_plot(
        eval_data_df,
        experiment_names=[
            "seq-accel_full-i",
            "seq-accel_scratch-i",
            "seq-accel_full-n1-i",
            "seq-accel_full-n3-i",
            "seq-accel_full-n20-i",
            "seq-accel_scratch-n1-i",
            "seq-accel_scratch-n3-i",
            "seq-accel_scratch-n20-i",
        ],
        labels=[
            "ACCEL",
            "ACCEL-0",
            "ACCEL$_1$",
            "ACCEL$_3$",
            "ACCEL$_{20}$",
            "ACCEL-0$_1$",
            "ACCEL-0$_3$",
            "ACCEL-0$_{20}$",
        ],
        path=dir_name,
        xlabel_y_coordinate=-0.1,
        out_filename="iqm_accel_num_mutations"
    )

    # PLR Ablations - Conditioning
    _make_plot(
        eval_data_df,
        experiment_names=["seq-plr-i", "seq-plr-vanilla-i", "seq-plr-myopic-i", "seq-plr-domain_indep-i"],
        labels=["Default", "Vanilla", "Myopic", "D.I. Embed."],
        path=dir_name,
        xlabel_y_coordinate=-0.3,
        out_filename="iqm_conditioning"
    )

    # ACCEL-0 Ablation - Hindsight
    _make_plot(
        eval_data_df,
        experiment_names=["seq-accel_scratch-i", "seq-accel_scratch-no_hind-i"],
        labels=["Hindsight", "No Hindsight"],
        path=dir_name,
        xlabel_y_coordinate=-0.7,
        out_filename="iqm_accel_scratch_no_hindsight"
    )

    # ACCEL Ablations - Mutation combos
    _make_plot(
        eval_data_df,
        experiment_names=[
            "seq-accel_full-tff-i",
            "seq-accel_full-ftf-i",
            "seq-accel_full-tft-i",
            "seq-accel_full-ftt-i",
            "seq-accel_full-ttf-i",
            "seq-accel_full-i",
        ],
        labels=[
            "L",
            "T",
            "L+H",
            "T+H",
            "L+T",
            "L+T+H",
        ],
        path=dir_name,
        xlabel_y_coordinate=-0.15,
        out_filename="iqm_accel_mutation_types"
    )

    # PVL ablations - Independent and Conditioned Sampling
    for sampling, sampling_all in [("i", "indep"), ("c", "cond")]:
        _make_plot(
            eval_data_df,
            experiment_names=[
                f"seq-plr-{sampling}", f"seq-accel_full-{sampling}", f"seq-accel_scratch-{sampling}",
                f"seq-plr-pvl-{sampling}", f"seq-accel_full-pvl-{sampling}", f"seq-accel_scratch-pvl-{sampling}"
            ],
            labels=[
                "PLR$^\\bot_\\text{MaxMC}$", "ACCEL$_\\text{MaxMC}$", "ACCEL-0$_\\text{MaxMC}$",
                "PLR$^\\bot_\\text{PVL}$", "ACCEL$_\\text{PVL}$", "ACCEL-0$_\\text{PVL}$",
            ],
            path=dir_name,
            xlabel_y_coordinate=-0.15,
            out_filename=f"iqm_pvl_seq_{sampling_all}"
        )
