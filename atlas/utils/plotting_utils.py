from typing import Dict, List, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from ..envs.common.labeling_function import LabelingFunction
from ..envs.xminigrid.level import XMinigridLevel, get_num_objects
from ..envs.xminigrid.mutators.common import Mutations, mutation_to_str, mutation_to_category, category_to_str, \
    MutationCategories
from ..hrm import ops as hrm_ops
from ..hrm.types import HRM

XMINIGRID_ROOM_SIZES = [(7, 7), (7, 13), (13, 13), (13, 19)]
XMINIGRID_NUM_OBJ_INTERVALS = [
    [1, 3, 5, float('inf')],
    [1, 4, 8, float('inf')],
    [4, 8, 12, float('inf')],
    [7, 11, 17, float('inf')]
]


def plotly_num_rms(hrms: HRM, prob_dict: Optional[Dict]) -> go.Figure:
    """
    Plots a histogram of the probability of sampling an HRM with
    a certain number of constituent RMs.
    """
    return go.Figure(_get_num_rms_bars(len(hrms.root_id), hrms, prob_dict))


def plotly_xminigrid_num_rms(
    levels: XMinigridLevel,
    hrms: HRM,
    prob_dict: Optional[Dict],
    room_sizes: List[Tuple[int, int]]
) -> go.Figure:
    """
    Plots (for each room size) a histogram of the probability of sampling an HRM with
    a certain number of constituent RMs.
    """
    fig = make_subplots(
        rows=len(room_sizes),
        cols=1,
        subplot_titles=[f"Room Size {h}x{w}" for h, w in room_sizes]
    )

    for i, (h, w) in enumerate(room_sizes):
        mask = (levels.height == h) & (levels.width == w)
        if mask.sum() > 0:
            bars = _get_num_rms_bars(
                len(hrms.root_id),
                jax.tree_util.tree_map(lambda x: x[mask], hrms),
                jax.tree_util.tree_map(lambda x: x[mask], prob_dict),
            )
            for bar in bars:
                fig.add_trace(bar, row=i + 1, col=1)
            fig.update_xaxes(title_text="num RMs per hierarchy", row=i + 1, col=1)
            fig.update_yaxes(range=[0, 1], row=i + 1, col=1)

    fig.update_layout(barmode="group", showlegend=True)
    return fig


def _get_num_rms_bars(num_problems: int, hrms: HRM, prob_dict: Optional[Dict]) -> List[go.Bar]:
    """
    Returns the `plotly` histogram of the probability of sampling an HRM with
    a certain number of constituent RMs.
    """
    max_num_rms = hrm_ops.get_max_num_machines(
        jax.tree_util.tree_map(lambda x: x[0], hrms)
    )
    num_rms = jax.vmap(hrm_ops.get_num_machines)(hrms)
    bins = jnp.arange(max_num_rms + 2)

    bars = [
        go.Bar(
            x=bins[:-1],
            y=jnp.histogram(num_rms, bins=bins)[0] / num_problems,
            name="fraction",
        )
    ]

    if prob_dict is not None:
        for prob_name, probs in prob_dict.items():
            bars.append(go.Bar(
                x=bins[:-1],
                y=jnp.histogram(num_rms, bins=bins, weights=probs)[0],
                name=prob_name,
            ))

    return bars


def plotly_avg_states_per_rm_sampling(hrms: HRM, prob_dict: Optional[Dict] = None) -> go.Figure:
    """
    Plots a histogram of the percentage of RMs with a certain
    average number of states per RM and (optionally) the probability of
    sampling an RM with a certain average number of states per RM.
    """
    return go.Figure(
        data=_get_hrm_sampling_bars(len(hrms.root_id), hrms, prob_dict),
        layout=dict(
            barmode="group",
            xaxis_title="average states per RM",
            yaxis=dict(range=[0.0, 1.0]),
        )
    )


def plotly_xminigrid_prob_distrib_sampling(
    levels: XMinigridLevel,
    hrms: HRM,
    prob_dict: Optional[Dict],
) -> go.Figure:
    """
    Plots (for each room size, and different object intervals) a histogram of the
    percentage of RMs with a certain average number of states per RM and (optionally)
    the probability of sampling an RM with a certain average number of states per RM.
    """
    fig = make_subplots(
        rows=len(XMINIGRID_ROOM_SIZES),
        cols=max([len(x) - 1 for x in XMINIGRID_NUM_OBJ_INTERVALS]),
        subplot_titles=[
            f"{h}x{w} / [{XMINIGRID_NUM_OBJ_INTERVALS[i][j]},{XMINIGRID_NUM_OBJ_INTERVALS[i][j + 1]})"
            for i, (h, w) in enumerate(XMINIGRID_ROOM_SIZES)
            for j in range(len(XMINIGRID_NUM_OBJ_INTERVALS[i]) - 1)
        ]
    )

    num_objects = get_num_objects(levels)
    for i, (h, w) in enumerate(XMINIGRID_ROOM_SIZES):
        intervals = XMINIGRID_NUM_OBJ_INTERVALS[i]
        for j in range(len(intervals) - 1):
            start, end = intervals[j], intervals[j + 1]
            mask = (levels.height == h) & (levels.width == w) & (num_objects >= start) & (num_objects < end)
            if mask.sum() > 0:
                bars = _get_hrm_sampling_bars(
                    len(hrms.root_id),
                    jax.tree_util.tree_map(lambda x: x[mask], hrms),
                    jax.tree_util.tree_map(lambda x: x[mask], prob_dict),
                )
                for bar in bars:
                    fig.add_trace(bar, row=i + 1, col=j + 1)
                fig.update_xaxes(title_text="average states per RM", row=i + 1, col=j + 1)
                fig.update_yaxes(range=[0, 1], row=i + 1, col=j + 1)

    fig.update_layout(barmode="group", showlegend=True)
    return fig


def _get_hrm_sampling_bars(num_problems: int, hrms: HRM, prob_dict: Optional[Dict]) -> List[go.Bar]:
    """
    Returns the `plotly` histogram of the percentage of RMs with a certain
    average number of states per RM and (optionally) the probability
    of sampling an RM with a certain average number of states per RM.

    TODO: error bars.
    """
    max_num_states = hrm_ops.get_max_num_states_per_machine(
        jax.tree_util.tree_map(lambda x: x[0], hrms)
    )
    avg_states_per_rm = jax.vmap(
        lambda hrm: jnp.sum(hrm_ops.get_num_states(hrm)) / hrm_ops.get_num_machines(hrm)
    )(hrms)
    bins = jnp.arange(max_num_states + 2)

    bars = [
        go.Bar(
            x=bins[:-1],
            y=jnp.histogram(avg_states_per_rm, bins=bins)[0] / num_problems,
            name="fraction",
        )
    ]

    if prob_dict is not None:
        for prob_name, probs in prob_dict.items():
            bars.append(go.Bar(
                x=bins[:-1],
                y=jnp.histogram(avg_states_per_rm, bins=bins, weights=probs)[0],
                name=prob_name,
            ))

    return bars


def plotly_avg_states_per_rm_solving(hrms: HRM, solved: chex.Array) -> go.Figure:
    """
    Plots a histogram of the percentage of solved and unsolved associated with
    an HRM containing a certain average number of states.
    """
    return go.Figure(
        data=_get_hrm_solved_bars(len(hrms.root_id), hrms, solved, showlegend=True),
        layout=dict(
            barmode="stack",
            xaxis_title="average states per RM",
            yaxis=dict(range=[0.0, 1.0]),
        )
    )


def plotly_xminigrid_prob_distrib_solving(
    levels: XMinigridLevel,
    hrms: HRM,
    solved: chex.Array
) -> go.Figure:
    """
    Plots (for each room size) a histogram of the percentage of solved and unsolved
    associated with an HRM containing a certain average number of states.
    """
    fig = make_subplots(
        rows=len(XMINIGRID_ROOM_SIZES),
        cols=max([len(x) - 1 for x in XMINIGRID_NUM_OBJ_INTERVALS]),
        subplot_titles=[
            f"{h}x{w} / [{XMINIGRID_NUM_OBJ_INTERVALS[i][j]},{XMINIGRID_NUM_OBJ_INTERVALS[i][j + 1]})"
            for i, (h, w) in enumerate(XMINIGRID_ROOM_SIZES)
            for j in range(len(XMINIGRID_NUM_OBJ_INTERVALS[i]) - 1)
        ]
    )

    first = True
    num_objects = get_num_objects(levels)
    for i, (h, w) in enumerate(XMINIGRID_ROOM_SIZES):
        intervals = XMINIGRID_NUM_OBJ_INTERVALS[i]
        for j in range(len(intervals) - 1):
            start, end = intervals[j], intervals[j + 1]
            mask = (levels.height == h) & (levels.width == w) & (num_objects >= start) & (num_objects < end)
            if mask.sum() > 0:
                unsolved_bar, solved_bar = _get_hrm_solved_bars(
                    len(hrms.root_id),
                    jax.tree_util.tree_map(lambda x: x[mask], hrms),
                    solved[mask],
                    first
                )
                fig.add_trace(unsolved_bar, row=i + 1, col=j + 1)
                fig.add_trace(solved_bar, row=i + 1, col=j + 1)
                fig.update_xaxes(title_text="average states per RM", row=i + 1, col=j + 1)
                fig.update_yaxes(range=[0, 1], row=i + 1, col=j + 1)
                first = False

    fig.update_layout(barmode="stack", showlegend=True)
    return fig


def _get_hrm_solved_bars(num_problems: int, hrms: HRM, solved: chex.Array, showlegend: bool) -> Tuple[go.Bar, go.Bar]:
    """
    Returns the `plotly` histogram for the percentage of solved and unsolved
    associated with an HRM containing a certain average number of states.

    TODO: error bars.
    """
    max_num_states = hrm_ops.get_max_num_states_per_machine(
        jax.tree_util.tree_map(lambda x: x[0], hrms)
    )
    avg_states_per_rm = jax.vmap(
        lambda hrm: jnp.sum(hrm_ops.get_num_states(hrm)) / hrm_ops.get_num_machines(hrm)
    )(hrms)
    bins = jnp.arange(max_num_states + 2)

    # Get the bin to which each average point belongs to
    bin_idxs = jnp.digitize(avg_states_per_rm, bins, right=True)

    # Compute the number of HRMs that belong to each bin
    num_hrms_per_type = jnp.histogram(avg_states_per_rm, bins=bins)[0]

    # Compute the number of solved problems for each bin
    num_solved_per_type = jax.ops.segment_sum(
        solved.astype(int), bin_idxs, num_segments=max_num_states + 1
    )

    return (
        go.Bar(
            x=bins[:-1],
            y=(num_hrms_per_type - num_solved_per_type) / num_problems,
            name="unsolved",
            marker_color="#1f77b4",
            showlegend=showlegend,
        ),
        go.Bar(
            x=bins[:-1],
            y=num_solved_per_type / num_problems,
            name="solved",
            marker_color="#ff7f0e",
            showlegend=showlegend,
        )
    )


def plotly_prop_frequency(hrms: HRM, label_fn: LabelingFunction) -> go.Figure:
    """
    Plots the frequency of each proposition within the list of HRMs.
    """
    frequencies = jax.vmap(hrm_ops.get_proposition_frequency, in_axes=(0, None))(
        hrms, label_fn.get_alphabet_size()
    )
    return go.Figure(
        data=go.Bar(
            x=label_fn.get_str_alphabet(),
            y=jnp.mean(frequencies, axis=0)
        ),
        layout=dict(
            xaxis_title="proposition",
            yaxis_title="average frequency",
            xaxis=dict(tickangle=90),
            yaxis=dict(range=[0.0, 1.0]),
        )
    )


def plotly_epsilon(hrms: HRM, probs: chex.Array, num_bins: int = 5) -> Optional[go.Figure]:
    """
    Plots a histogram showing the probability of sampling an HRM
    with a certain average epsilon value. The epsilon is the average
    across the constituent RMs of the hierarchy. The x-axis error
    is the average std across samples in each bin.
    """
    hrm = jax.tree_util.tree_map(lambda x: x[0], hrms)
    if not hrm.extras.get("eps_log") is not None:
        return None

    epsilons = hrm.extras["eps_log"]
    valid_rms = jax.vmap(hrm_ops.get_machine_mask)(hrms)
    num_valid_rms = jnp.sum(valid_rms, axis=1)

    # Compute the average epsilon for each HRM (there is an epsilon for each
    # constituent RM) and the standard deviation
    avg_epsilon = jnp.sum(epsilons * valid_rms, axis=1) / num_valid_rms
    sum_squared_diff = jnp.sum((valid_rms * (epsilons - avg_epsilon[:, jnp.newaxis])) ** 2, axis=1)
    std_epsilon = jnp.sqrt(sum_squared_diff / num_valid_rms)

    bins = jnp.linspace(
        start=hrm.extras["min_eps_log"],
        stop=hrm.extras["max_eps_log"],
        num=num_bins
    )
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Compute accumulated probability of sampling an HRM in each bin
    bin_probs, _ = jnp.histogram(avg_epsilon, bins=bins, weights=probs)

    # Compute the x-axis error for each bin as the average std
    # of the HRM samples in the bin
    bin_indices = jnp.digitize(avg_epsilon, bins) - 1
    bin_std_sum = jax.ops.segment_sum(std_epsilon, bin_indices, len(bins) - 1)
    num_samples_per_bin = jnp.clip(
        jax.ops.segment_sum(jnp.ones_like(bin_indices), bin_indices, len(bins) - 1),
        min=1
    )
    bin_error = bin_std_sum / num_samples_per_bin

    return go.Figure(
        data=[
            go.Bar(x=bin_centers, y=bin_probs),
            go.Scatter(
                x=bin_centers,
                y=bin_probs,
                mode="markers",
                error_x=dict(type='data', array=bin_error, visible=True),
                marker=dict(color="red", size=10)
            )
        ],
        layout=dict(
            xaxis_title="epsilon",
            yaxis_title="sampling probability",
            yaxis=dict(range=[0.0, 1.0]),
            bargap=0.2,  # Gap between bars
            showlegend=False,
        )
    )


def plotly_buffer_scores(
    scores: chex.Array,
    prob_dict: Dict,
    num_bins: int = 10,
    mask: Optional[chex.Array] = None,
) -> go.Figure:
    """
    Plots a histogram of the scores in the buffer.
    """
    if mask is not None:
        scores, prob_dict = jax.tree_util.tree_map(
            lambda x: x[mask], (scores, prob_dict)
        )

    return go.Figure(
        data=[
            go.Histogram(x=scores, nbinsx=num_bins, histnorm="probability", name="Fraction"),
            *[
                go.Histogram(x=scores, y=weights, nbinsx=num_bins, histfunc="sum", name=name)
                for name, weights in prob_dict.items()
            ]
        ],
        layout=dict(
            xaxis_title="Scores",
            yaxis=dict(range=[0.0, 1.0]),
        )
    )


def plotly_buffer_staleness(
    staleness: chex.Array, staleness_weights: chex.Array, problem_weights: chex.Array, num_bins: int = 10
) -> go.Figure:
    """
    Plots a histogram of the staleness in the buffer.
    """
    return go.Figure(
        data=[
            go.Histogram(x=staleness, nbinsx=num_bins, histnorm="probability", name="Fraction"),
            go.Histogram(x=staleness, y=staleness_weights, nbinsx=num_bins, histfunc="sum", name="Sampling Prob"),
            go.Histogram(x=staleness, y=problem_weights, nbinsx=num_bins, histfunc="sum", name="Sampling Prob w/ Scores"),
        ],
        layout=dict(
            xaxis_title="Staleness",
            yaxis=dict(range=[0.0, 1.0]),
        )
    )


def plotly_mutation_count(mutation_ids: chex.Array, prob_dict: Optional[dict] = None) -> go.Figure:
    """
    Plots a histogram indicating how many problems contain
    a certain mutation (or no mutation at all).
    """
    id_range = jnp.arange(len(Mutations))
    num_problems = mutation_ids.shape[0]

    x_labels = ["no mutation"] + [mutation_to_str(mutation_id) for mutation_id in id_range]
    mask = jnp.concat((
        jnp.all(mutation_ids == -1, axis=1)[jnp.newaxis, ...],
        jax.vmap(lambda x: jnp.any(mutation_ids == x, axis=1))(id_range)
    ))

    bars = [go.Bar(x=x_labels, y=jnp.sum(mask, axis=1) / num_problems, name="fraction")]
    if prob_dict is not None:
        for prob_name, probs in prob_dict.items():
            bars.append(go.Bar(x=x_labels, y=jnp.sum(mask * probs, axis=1), name=prob_name))

    return go.Figure(
        data=bars,
        layout=dict(
            xaxis_title="Mutation Name",
            bargap=0.05,
            bargroupgap=0.2,
            yaxis=dict(range=[0.0, 1.0]),
        )
    )


def plotly_mutation_category_count(mutation_ids: chex.Array, prob_dict: Optional[Dict] = None) -> go.Figure:
    """
    Plots a histogram indicating how many problems perform
    mutations of a certain category.
    """
    mutation_cats = mutation_to_category(mutation_ids)
    mask = mutation_ids >= 0
    mutation_cats = mask * mutation_cats - jnp.logical_not(mask)

    id_range = jnp.arange(len(MutationCategories))

    x_labels = ["no mutation"] + [category_to_str(category_id) for category_id in id_range]
    cat_mask = jnp.concat((
        jnp.all(mutation_cats == -1, axis=1)[jnp.newaxis, ...],
        jax.vmap(lambda x: jnp.any(mutation_cats == x, axis=1))(id_range)
    ))

    bars = [go.Bar(x=x_labels, y=jnp.sum(cat_mask, axis=1) / mutation_cats.shape[0], name="fraction")]
    if prob_dict is not None:
        for prob_name, probs in prob_dict.items():
            bars.append(go.Bar(x=x_labels, y=jnp.sum(cat_mask * probs, axis=1), name=prob_name))

    return go.Figure(
        data=bars,
        layout=dict(
            xaxis_title="Mutation Category Name",
            bargap=0.05,
            bargroupgap=0.2,
            yaxis=dict(range=[0.0, 1.0]),
        )
    )


def plotly_mutation_fraction(mutation_counts: chex.Array, prob_dict: Optional[Dict] = None) -> go.Figure:
    """
    Plots a histogram showing how many of the problems are the outcome
    of a mutation.
    """
    mutated_mask = mutation_counts > 0
    num_problems = mutation_counts.shape[0]
    num_mutated_problems = jnp.sum(mutated_mask)

    x_labels = ["Non-Mutated/DR", "Mutated"]
    bars = [go.Bar(
        x=x_labels,
        y=[(num_problems - num_mutated_problems) / num_problems, num_mutated_problems / num_problems],
        name="fraction"
    )]

    if prob_dict is not None:
        for prob_name, probs in prob_dict.items():
            bars.append(go.Bar(
                x=x_labels,
                y=[jnp.sum(jnp.logical_not(mutated_mask) * probs), jnp.sum(mutated_mask * probs)],
                name=prob_name
            ))

    return go.Figure(
        data=bars,
        layout=dict(
            yaxis_title="Problem Count Fraction",
            bargap=0.05,
            bargroupgap=0.2,
            yaxis=dict(range=[0.0, 1.0]),
        )
    )


def plotly_mutation_hindsight_lvl(
    hrms: HRM, mutation_ids: chex.Array, mutation_args: chex.Array, prob_dict: Optional[Dict] = None
) -> go.Figure:
    """
    A similar plot to `plotly_mutation_count` but based on the arguments of the hindsight-level
    mutations, i.e. what caused such a mutation to be performed: getting stuck in the initial
    state, or reaching the accepting state.
    """
    num_problems = mutation_ids.shape[0]
    arg_id = 0
    hindsight_lvl_mask = mutation_ids == Mutations.HINDSIGHT_LVL_ONLY
    partition_states = hindsight_lvl_mask * mutation_args[..., arg_id] - jnp.logical_not(hindsight_lvl_mask)

    mask_init_states = partition_states == hrm_ops.get_initial_state_id()
    mask_acc_states = partition_states == hrm_ops.get_accepting_state_id(
        jax.tree_util.tree_map(lambda x: x[0], hrms)
    )
    masks = jnp.concat((
        jnp.any(mask_init_states, axis=1)[jnp.newaxis, ...],
        jnp.any(mask_acc_states, axis=1)[jnp.newaxis, ...])
    )

    x_labels = ["Initial", "Accepting"]
    bars = [go.Bar(x=x_labels, y=jnp.sum(masks, axis=1) / num_problems, name="fraction")]
    if prob_dict is not None:
        for prob_name, probs in prob_dict.items():
            bars.append(go.Bar(x=x_labels, y=jnp.sum(masks * probs, axis=1), name=prob_name))

    return go.Figure(
        data=bars,
        layout=dict(
            xaxis_title="Level Hindsight Mutation Cause",
            bargap=0.05,
            bargroupgap=0.2,
            yaxis=dict(range=[0.0, 1.0]),
        )
    )


def plotly_num_mutations(mutation_ids: chex.Array) -> go.Figure:
    """
    Plots the average number of mutations across the input and the standard deviation as an error bar.
    """
    num_mutations = jnp.sum(mutation_ids > -1, axis=1)

    return go.Figure(
        data=go.Bar(
            y=[num_mutations.mean()],
            error_y=dict(
                type='data',
                array=[num_mutations.std()],
                visible=True,
                color='black',
                thickness=2,
                width=3
            ),
        ),
        layout=dict(
            yaxis_title='Number of mutations',
            xaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                title=''
            )
        )
    )

