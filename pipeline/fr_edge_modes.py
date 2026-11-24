"""Edge-feature modes the library modules lack, added in memory for two ablation arms.

The flat library modules stay unmodified (CLAUDE.md rule 3, and the regression
contract): as s0_generate_population.py does for the generator, the two modes are
patched into the imported modules, by fr_train.py before it builds a dataset and by
s4_indomain.py before it evaluates.  Added 2026-09-28, with the author:

  GraphSAGE  "cpsd_full"    the full-spectrum complex coherency EdgeGATConv reads
                            (edge_attr_cpsd_full, 800-d), so the two architectures
                            are compared on identical inputs.  GraphSAGE's edge
                            encoder maps it to one positive weight per edge
                            (sigmoid, then EdgeWeightNorm), as for every other mode.
  EdgeGAT    "geom_orient"  edge orientation only: (cos theta, sin theta) of the
                            stored edge angle (edge_attr_geom[:, 1]).  No phase and
                            no length - raw lengths in mm (up to ~20 000) are what
                            made the GraphSAGE geometry arm unrunnable.
Added 2026-09-29, completing a two-layer x three-feature grid (none | orientation |
full coherency):
  GraphSAGE  "geom_orient"  the same orientation features; bounded in [-1, 1], so the
                            sigmoid edge weight cannot underflow to zero
  EdgeGAT    "zero"         no edge information: zeros of GraphSAGE's "zero" shape
                            (E, 2), so every edge gets the same learned embedding

Every existing mode is delegated to the library's own function, unchanged.
"""

from __future__ import annotations

import numpy as np

GS_CPSD_FULL = "cpsd_full"
GS_GEOM_ORIENT = "geom_orient"
EG_GEOM_ORIENT = "geom_orient"
EG_ZERO = "zero"
CPSD_FULL_DIM = 800
ORIENT_DIM = 2


def orientation(data) -> np.ndarray:
    """(cos theta, sin theta) per directed edge, float32, from the stored edge angle."""
    theta = np.asarray(data["edge_attr_geom"], dtype=np.float64)[:, 1]
    return np.column_stack([np.cos(theta), np.sin(theta)]).astype(np.float32)


def install() -> None:
    """Patch both modes into the imported library modules; safe to call more than once."""
    import EdgeGATConv_Modal_GNN as eg
    import GraphSAGE_Modal_GNN as gs

    if getattr(gs, "_fr_edge_modes_installed", False):
        return
    gs_resolve, gs_dim, eg_resolve = gs._resolve_edge_attr_array, gs._edge_feature_dim_for_mode, eg._resolve_edge_attr_array

    def gs_resolve_ext(data, edge_feature_mode: str) -> np.ndarray:
        if edge_feature_mode == GS_CPSD_FULL:
            a = np.asarray(data["edge_attr_cpsd_full"], dtype=np.float32)
            n_edges = np.asarray(data["edge_index"]).shape[1]
            if a.shape != (n_edges, CPSD_FULL_DIM):
                raise ValueError(f"edge_attr_cpsd_full must have shape ({n_edges}, {CPSD_FULL_DIM}), got {a.shape}")
            return a
        if edge_feature_mode == GS_GEOM_ORIENT:
            return orientation(data)
        return gs_resolve(data, edge_feature_mode)

    def gs_dim_ext(edge_feature_mode: str) -> int:
        if edge_feature_mode == GS_CPSD_FULL:
            return CPSD_FULL_DIM
        if edge_feature_mode == GS_GEOM_ORIENT:
            return ORIENT_DIM
        return gs_dim(edge_feature_mode)

    def eg_resolve_ext(data, edge_feature_mode: str) -> np.ndarray:
        if edge_feature_mode == EG_GEOM_ORIENT:
            return orientation(data)
        if edge_feature_mode == EG_ZERO:
            return np.zeros((np.asarray(data["edge_index"]).shape[1], ORIENT_DIM), dtype=np.float32)
        return eg_resolve(data, edge_feature_mode)

    gs._resolve_edge_attr_array = gs_resolve_ext
    gs._edge_feature_dim_for_mode = gs_dim_ext
    gs.EDGE_FEATURE_MODES = (*gs.EDGE_FEATURE_MODES, GS_CPSD_FULL, GS_GEOM_ORIENT)
    eg._resolve_edge_attr_array = eg_resolve_ext
    eg.EDGE_FEATURE_MODES = (*eg.EDGE_FEATURE_MODES, EG_GEOM_ORIENT, EG_ZERO)
    gs._fr_edge_modes_installed = True
