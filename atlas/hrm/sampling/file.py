from typing import Dict, List, Optional

import chex
from chex import dataclass
import jax
import jax.numpy as jnp

from .common import HRMSampler
from .. import ops
from ..types import HRM


@dataclass
class FileHRMSamplerExtras:
    file_id: chex.Array  # the index of the file in the input


class FileHRMSampler(HRMSampler):
    """
    Samples an HRM initially loaded from a list of file paths. The actual root ids
    are determined by the loaded files.
    """

    def __init__(
        self,
        files: List[str],
        max_num_rms: int,
        max_num_states: int,
        max_num_edges: int,
        max_num_literals: int,
        alphabet_size: int,
        alphabet: List[str],
        **kwargs: dict,
    ):
        assert len(files) > 0, "The number of files to sample HRMs from must be > 0"

        super().__init__(max_num_rms, max_num_states, max_num_edges, max_num_literals, alphabet_size)

        hrms = []
        for i, f in enumerate(files):
            hrm = self._init_hrm(root_id=0, extras=FileHRMSamplerExtras(file_id=i))
            ops.load(hrm, f, alphabet)
            hrms.append(hrm)

        self._hrms = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *hrms)
        self._num_hrms = len(files)

    def sample(self, key: chex.PRNGKey, extras: Optional[Dict] = None) -> HRM:
        idx = jax.random.randint(key, shape=(), minval=0, maxval=self._num_hrms)
        return jax.tree_map(lambda x: x[idx], self._hrms)
