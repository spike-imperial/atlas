import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
pd.options.mode.copy_on_write = True
import seaborn as sns


def _make_plot(algorithms, names, colors, xticks, xticklabels, filename):
    _, ax = plt.subplots(figsize=(7, 5))

    for algorithm, name, color in zip(algorithms, names, colors):
        df_algorithm = df[df["algorithm"] == algorithm]
        df_algorithm["name"] = name
        sns.lineplot(
            df_algorithm,
            x="step",
            y="percent",
            marker='o',
            hue="name",
            palette=[color],
            markeredgecolor='auto',
            linewidth=2,
            ax=ax,
            err_kws={"alpha": 0.2}
        )

    ax.set_xlabel('Number of Environment Steps (in millions)', fontsize="xx-large")
    ax.set_ylabel('% Solvable Buffer Problems', fontsize="xx-large")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.grid(True, alpha=0.2)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params(length=0.1, width=0.1, labelsize="xx-large")
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))

    ax.legend(title=None, fontsize="x-large", loc='lower right')

    ax.set_ylim(-0.04331604051589966, 1.0451226906776427)
    plt.savefig(filename, bbox_inches="tight")


if __name__ == "__main__":
    dir_name = os.path.join(os.path.dirname(sys.argv[0]), "results")
    os.makedirs(dir_name, exist_ok=True)

    df = pd.read_csv(os.path.join(dir_name, "..", "solvability_over_time.csv"), sep=";")
    set2 = sns.color_palette("Set2", 8)
    for sampling, sampling_full in [("i", "indep"), ("c", "cond")]:
        _make_plot(
            algorithms=[f"seq-plr-{sampling}", f"seq-accel_full-{sampling}", f"seq-accel_scratch-{sampling}"],
            names=["PLR$^\\bot$", "ACCEL", "ACCEL-0"],
            colors=[set2[0], set2[1], set2[2]],
            xticks=[0, 500, 1000, 1500, 2000],
            xticklabels=[0, 1000, 2000, 3000, 4000],
            filename=os.path.join(dir_name, f"solvability_seq-{sampling}.pdf")
        )

    _make_plot(
        algorithms=[f"rw-plr-i", "rw-plr-c"],
        names=["PLR$^\\bot_\\text{indep}$", "PLR$^\\bot_\\text{cond}$"],
        colors=[set2[0], set2[1]],
        xticks=[0, 1000, 2000, 3000],
        xticklabels=[0, 2000, 4000, 6000],
        filename=os.path.join(dir_name, f"solvability_rw.pdf")
    )
