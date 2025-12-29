import io
import tempfile
from typing import List, Optional, Set

import chex
import graphviz
import jax.numpy as jnp
from PIL import Image

from .types import HRM, HRMState, StackFields
from .ops import (
    is_accepting_state,
    is_empty_rm,
    is_leaf_rm,
    is_null_rm,
    is_root_rm,
    is_terminal_state,
    get_max_num_machines,
    get_max_num_states_per_machine,
    get_max_num_edges_per_state_pair,
)


def render_to_file(
    hrm: HRM,
    output_path: str,
    hrm_state: Optional[HRMState] = None,
    alphabet: Optional[List[str]] = None,
    title: Optional[str] = None,
) -> None:
    """
    Renders a hierarchy of reward machines to a file.

    Args:
        hrm: the HRM to render.
        output_path: where the render is saved.
        hrm_state: the HRM state whose items to highlight (if specified).
        alphabet: the alphabet on which the HRM is defined - if specified,
            integers are replaced by the string in their positions.
        title: the title of the plot.

    Source:
        https://stackoverflow.com/questions/19280229/graphviz-putting-a-caption-on-a-node-in-addition-to-a-label
    """

    def _get_rm_label(rm_id: chex.Numeric) -> str:
        return f"M<SUB>{rm_id}</SUB>"

    def _get_cluster_label(rm_id: chex.Numeric) -> str:
        suffix = " (root)" if is_root_rm(hrm, rm_id) else ""
        return f"<{_get_rm_label(rm_id)}{suffix}>"

    def _get_edge_label(
        called_rm: chex.Numeric,
        formula: chex.Array,
        num_literals: chex.Numeric,
        reward: chex.Numeric,
    ) -> str:
        called_rm_str = (
            f"<{_get_rm_label(called_rm)}|" if not is_leaf_rm(hrm, called_rm) else "<"
        )

        formula_str = []
        for i in range(num_literals):
            literal = formula[i]
            prop = jnp.abs(literal) - 1
            prop_str = alphabet[prop] if alphabet else str(prop)
            if literal > 0:
                formula_str.append(prop_str)
            elif literal < 0:
                formula_str.append(f"-{prop_str}")

        formula_str = "&amp;".join(formula_str) if len(formula_str) > 0 else "T"
        return f"{called_rm_str}{formula_str},{reward:.2f}>"

    def _get_global_state_id(rm_id: chex.Numeric, state_id: chex.Numeric) -> str:
        return f"{rm_id}_{state_id}"

    def _add_node(
        g: graphviz.Digraph, rm_id: chex.Numeric, state_id: chex.Numeric
    ) -> None:
        global_state_id = _get_global_state_id(rm_id, state_id)
        if global_state_id in graph_nodes:
            return
        graph_nodes.add(global_state_id)

        state_label = "A" if is_accepting_state(hrm, state_id) else state_id
        attrs = {
            "label": f"u<SUB>{state_label}</SUB><BR /><FONT POINT-SIZE='10'> </FONT>"
        }

        # Placeholder label
        if hrm_state:
            if hrm_state.rm_id == rm_id and hrm_state.state_id == state_id:
                attrs["color"] = "#FF0000"
                attrs["fontcolor"] = "#FF0000"
                attrs["style"] = "filled"
                attrs["fillcolor"] = "#FFCCCC"

            stack_level = _get_rm_state_stack_level(rm_id, state_id)
            if stack_level >= 0:
                attrs["color"] = "#0000FF"
                attrs["fontcolor"] = "#0000FF"
                attrs["style"] = "filled"
                attrs["fillcolor"] = "#CCCCFF"
                attrs["label"] = attrs["label"].replace(
                    " </FONT>", f"({stack_level})</FONT>"
                )

        attrs["label"] = f"<{attrs['label']}>"
        g.node(global_state_id, **attrs)

    def _add_edge(
        g: graphviz.Digraph,
        rm_id: chex.Numeric,
        src_id: chex.Numeric,
        dst_id: chex.Numeric,
        edge_id: chex.Numeric,
    ) -> None:
        called_rm = hrm.calls[rm_id, src_id, dst_id, edge_id]
        reward = hrm.rewards[rm_id, src_id, dst_id]
        if not is_null_rm(hrm, called_rm):
            _add_node(g, rm_id, dst_id)
            g.edge(
                _get_global_state_id(rm_id, src_id),
                _get_global_state_id(rm_id, dst_id),
                label=_get_edge_label(
                    called_rm,
                    hrm.formulas[rm_id, src_id, dst_id, edge_id],
                    hrm.num_literals[rm_id, src_id, dst_id, edge_id],
                    reward,
                ),
            )
        elif src_id == dst_id and edge_id == 0 and reward != 0:
            # Print reward for self-edges if any (only for one edge)
            _add_node(g, rm_id, dst_id)
            g.edge(
                _get_global_state_id(rm_id, src_id),
                _get_global_state_id(rm_id, dst_id),
                label=f"{reward:.2f}"
            )

    def _is_rm_active(rm_id: chex.Numeric) -> bool:
        if hrm_state.rm_id == rm_id:
            return True
        for i in range(hrm_state.stack_size):
            stack_transition = hrm_state.stack[i]
            if rm_id == stack_transition[StackFields.CALLING_RM]:
                return True
        return False

    def _get_rm_state_stack_level(rm_id: chex.Numeric, state_id: chex.Numeric) -> int:
        for i in range(hrm_state.stack_size):
            stack_transition = hrm_state.stack[i]
            if (
                rm_id == stack_transition[StackFields.CALLING_RM]
                and state_id == stack_transition[StackFields.DST_STATE_CALLING_RM]
            ):
                return i
        return -1

    def _color_subgraph(subgraph: graphviz.Digraph, rm_id: chex.Numeric) -> None:
        if hrm_state and not _is_rm_active(rm_id):
            subgraph.attr(color="gray", fontcolor="gray")
            subgraph.node_attr["color"] = "gray"
            subgraph.node_attr["fontcolor"] = "gray"
            subgraph.edge_attr["color"] = "gray"
            subgraph.edge_attr["fontcolor"] = "gray"

    g = graphviz.Digraph("G")  # The 'neato' engine is faster but HRMs look worse
    g.node_attr["shape"] = "circle"
    graph_nodes: Set[str] = set()

    # Add this block
    if title:
        g.attr(label=title, labelloc="tl", labeljust="l")

    for rm_id in jnp.arange(get_max_num_machines(hrm)):
        if is_empty_rm(hrm, rm_id):
            continue
        with g.subgraph(name=f"cluster_{rm_id}") as subgraph:
            subgraph.attr(label=_get_cluster_label(rm_id))
            _color_subgraph(subgraph, rm_id)

            for src_id in jnp.arange(get_max_num_states_per_machine(hrm)):
                if is_terminal_state(hrm, rm_id, src_id):
                    continue
                _add_node(subgraph, rm_id, src_id)
                for dst_id in jnp.arange(get_max_num_states_per_machine(hrm)):
                    for edge_id in jnp.arange(get_max_num_edges_per_state_pair(hrm)):
                        _add_edge(subgraph, rm_id, src_id, dst_id, edge_id)

    g.render(outfile=output_path, format="png", cleanup=True)


def render_to_img(
    hrm: HRM,
    hrm_state: Optional[HRMState] = None,
    alphabet: Optional[List[str]] = None,
    title: Optional[str] = None,
) -> Image:
    """
    Renders a hierarchy of reward machines to an image object.

    Args:
        hrm: the HRM to render.
        hrm_state: the HRM state whose items to highlight (if specified).
        alphabet: the alphabet on which the HRM is defined - if specified,
            integers are replaced by the string in their positions.
        title: the title of the plot.
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        render_to_file(hrm, temp_file.name, hrm_state, alphabet, title)
        temp_file.seek(0)
        image_data = temp_file.read()
    return Image.open(io.BytesIO(image_data))
