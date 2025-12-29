"""
Produces a learning curve over the test set by computing the aggregated IQM of the
solve rate at different checkpoints.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils
import seaborn as sns


def _make_plot(eval_data_df, experiment_names, labels, colors, filename):
    results = {}
    for e, l in zip(experiment_names, labels):
        df_exp = eval_data_df[eval_data_df["experiment"] == e]
        problem_names = df_exp.columns.difference(["experiment", "training_run_id", "step"])

        data = []
        for run_id in df_exp["training_run_id"].unique():
            df_run = df_exp[df_exp["training_run_id"] == run_id]
            run_data = []
            steps = sorted(df_run["step"].unique())
            for problem in problem_names:
                problem_data = []
                for step in steps:
                    problem_data.append(
                        df_run[df_run["step"] == step][problem].iloc[0]
                    )
                run_data.append(problem_data)
            data.append(run_data)

        results[l] = np.array(data)

    frames = [10, *range(100, 2001, 100)]

    iqm = lambda scores: np.array([
        metrics.aggregate_iqm(scores[..., frame])
        for frame in range(len(frames))
    ])

    iqm_scores, iqm_cis = rly.get_interval_estimates(
        results, iqm, reps=50000
    )

    plot_utils.plot_sample_efficiency_curve(
        frames,
        iqm_scores,
        iqm_cis,
        xlabel=r'Number of Environment Steps (in millions)',
        ylabel="IQM Solve Rate",
        colors=colors,
        xticks=[0, 500, 1000, 1500, 2000],
        xticklabels=[0, 1000, 2000, 3000, 4000],
        legend=True
    )

    plt.ylim(-0.04331604051589966, 1.0451226906776427)
    plt.savefig(filename, bbox_inches="tight", pad_inches=0.0)


if __name__ == "__main__":
    dir_name = os.path.join(os.path.dirname(sys.argv[0]), "results")
    os.makedirs(dir_name, exist_ok=True)
    eval_data_df = pd.read_csv(os.path.join(dir_name, "..", "..", "data_collection", "eval_checkpoint_seq.csv"))
    set2 = sns.color_palette("Set2", 8)

    _make_plot(
        eval_data_df,
        experiment_names=["seq-dr-i", "seq-plr-i", "seq-accel_full-i", "seq-accel_scratch-i"],
        labels=["DR", "PLR$^\\bot$", "ACCEL", "ACCEL-0"],
        colors={"DR": set2[3], "PLR$^\\bot$": set2[0], "ACCEL": set2[1], "ACCEL-0": set2[2]},
        filename=os.path.join(dir_name, "iqm_seq_indep_curve.pdf")
    )

    _make_plot(
        eval_data_df,
        experiment_names=["seq-dr-c", "seq-plr-c", "seq-accel_full-c", "seq-accel_scratch-c"],
        labels=["DR", "PLR$^\\bot$", "ACCEL", "ACCEL-0"],
        colors={"DR": set2[3], "PLR$^\\bot$": set2[0], "ACCEL": set2[1], "ACCEL-0": set2[2]},
        filename=os.path.join(dir_name, "iqm_seq_cond_curve.pdf")
    )
