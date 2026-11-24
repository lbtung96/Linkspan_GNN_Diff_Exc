"""Edge-feature modes - library code since the 2026-11-24 restructure; this module is kept for its callers.

In the archive, this module patched four modes into the imported library modules in memory
(2026-09-28/29): GraphSAGE "cpsd_full" (800-d coherency) and "geom_orient"; EdgeGAT "geom_orient" and
"zero".  EC T16 then patched in EdgeGAT's 802-d "cpsd_full_orient" (p_common.install_cohorient).  All
five are now ordinary modes of GraphSAGE_Modal_GNN / EdgeGATConv_Modal_GNN, with the same arithmetic:

  geom_orient        (cos theta, sin theta) of the stored edge angle edge_attr_geom[:, 1], float32
  zero (EdgeGAT)     zeros of shape (E, 2): every edge gets the same learned embedding
  cpsd_full (GS)     edge_attr_cpsd_full, 800-d, mapped by GraphSAGE's encoder to one positive weight
  cpsd_full_orient   hstack(edge_attr_cpsd_full, orientation), 802-d, float32

install() is now a check that the library carries them; orientation() is the library's
edge_orientation().
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
    import GraphSAGE_Modal_GNN as gs
    return gs.edge_orientation(data)


def install() -> None:
    """No patching: assert the library carries every mode this module used to add."""
    import EdgeGATConv_Modal_GNN as eg
    import GraphSAGE_Modal_GNN as gs

    missing = [m for m in (GS_CPSD_FULL, GS_GEOM_ORIENT) if m not in gs.EDGE_FEATURE_MODES]
    missing += [m for m in (EG_GEOM_ORIENT, EG_ZERO, "cpsd_full_orient") if m not in eg.EDGE_FEATURE_MODES]
    if missing:
        raise RuntimeError(f"library edge modes missing: {missing}")
