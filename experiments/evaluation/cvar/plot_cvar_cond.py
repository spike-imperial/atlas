"""
Plots the CVaR curves for the sequential and random-walk RM sampling
trained in the independent sampling setting. The CVaR set is formed
using the level-conditioned sampler.

See `run_eval_seq_cond_set`, `run_eval_rw_cond_set` and `dump_eval_cond_set`.
"""
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
pd.options.mode.copy_on_write = True
import seaborn as sns

from atlas.utils.evaluation import cvar


def _make_plot(cvar_df, algorithms, names, colors, out_filename):
    _, ax = plt.subplots(figsize=(7, 5))

    for algorithm, name, color in zip(algorithms, names, colors):
        df_algorithm = cvar_df[cvar_df["algorithm"] == algorithm]
        df_algorithm["name"] = name
        sns.lineplot(
            df_algorithm,
            x="percent",
            y="solve_rate",
            marker='o',
            hue="name",
            palette=[color],
            markeredgecolor='auto',
            linewidth=2,
            ax=ax,
            err_kws={"alpha": 0.2}
        )

    ax.set_xlabel('$\\alpha$% Worst-Case Problems', fontsize="xx-large")
    ax.set_xscale('log')
    ax.set_ylabel('Average Solve Rate', fontsize="xx-large")
    ax.set_xticks([1, 10, 100])
    ax.set_xticklabels(['1', '10', '100'])
    ax.grid(True, alpha=0.2)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params(length=0.1, width=0.1, labelsize="xx-large")
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.legend(title=None, fontsize="x-large")
    ax.set_ylim(-0.04331604051589966, 1.0451226906776427)

    plt.savefig(out_filename, bbox_inches="tight", pad_inches=0.0)


def _make_plot_df(filename, algorithms):
    df = pd.read_csv(filename)
    problem_cols = df.columns.difference({"experiment", "training_run_id", "seed"})

    rows = []
    for algorithm in algorithms:
        df_algo = df[df["experiment"] == algorithm]
        run_ids = df_algo["training_run_id"].unique()

        for run_id in run_ids:
            df_run = df_algo[df_algo["training_run_id"] == run_id]
            biased = df_run[df_run["seed"] == "s0"][problem_cols].to_numpy().squeeze()
            unbiased = df_run[df_run["seed"] == "s1"][problem_cols].to_numpy().squeeze()
            _, cvar_solve_rates = cvar(biased, unbiased)
            for percent, solve_rate in cvar_solve_rates.items():
                rows.append({
                    "algorithm": algorithm,
                    "percent": percent,
                    "solve_rate": float(solve_rate)
                })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    dir_name = os.path.join(os.path.dirname(sys.argv[0]), "results")
    os.makedirs(dir_name, exist_ok=True)

    set2 = sns.color_palette("Set2", 8)

    # Sequential + Independent
    algorithms = ["dr-i", "plr-i", "accel_full-i", "accel_scratch-i"]
    _make_plot(
        _make_plot_df(os.path.join(dir_name, "..", "data_collection", "cvar_seq.csv"), algorithms),
        algorithms,
        names=["DR", "PLR$^\\bot$", "ACCEL", "ACCEL-0"],
        colors=[set2[3], set2[0], set2[1], set2[2]],
        out_filename=os.path.join(dir_name, "cvar_seq_indep.pdf")
    )

    # Sequential + Conditioned
    algorithms = ["dr-c", "plr-c", "accel_full-c", "accel_scratch-c"]
    _make_plot(
        _make_plot_df(os.path.join(dir_name, "..", "data_collection", "cvar_seq.csv"), algorithms),
        algorithms,
        names=["DR", "PLR$^\\bot$", "ACCEL", "ACCEL-0"],
        colors=[set2[3], set2[0], set2[1], set2[2]],
        out_filename=os.path.join(dir_name, "cvar_seq_cond.pdf")
    )

    # Sequential + Independent + Conditioned
    algorithms = ["dr-i", "dr-c", "plr-i", "plr-c"]
    _make_plot(
        _make_plot_df(os.path.join(dir_name, "..", "data_collection", "cvar_rw.csv"), algorithms),
        algorithms,
        names=["DR$_\\text{indep}$", "DR$_\\text{cond}$", "PLR$^\\bot_\\text{indep}$", "PLR$^\\bot_\\text{cond}$"],
        colors=[set2[3], set2[0], set2[1], set2[2]],
        out_filename=os.path.join(dir_name, "cvar_rw.pdf")
    )
