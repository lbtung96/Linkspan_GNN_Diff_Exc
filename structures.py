"""Single source of truth for which real structures each stage operates on.

The source repository carried **twelve** independent copies of this list, at
three different lengths, and they had drifted:

  * `build_phase3_{node,edge}_features.py`        10  (all, incl. Sykkylven)
  * `build_phase3_features_corrected.py`,
    `E2/stage2_zeroshot.py`, `E2/stage3_finetune.py` 9  (Festoya added 2026-08-31)
  * `inspect_phase3_propagation.py`                9  (but Sykkylven, not Festoya)
  * everything else                                8  (pre-Festoya)

The drift was not harmless.  `E6/task2_propagation_only_control.py` shipped with
a hard-coded 8 while the CSV beside it held 9 - so the script as shipped could
not reproduce its own stored result.  Import from here instead of re-declaring.

Order matters.  The nested LOOCV protocol picks the validation structure as
``LOOCV[(i + 1) % len(LOOCV)]``, so Festoya is appended at the END: that changes
only Stranda's partner rather than re-rotating every fold.
"""

from __future__ import annotations

#: The eight structures every result before 2026-08-31 was computed on.
#: Kept so historical tables can be reproduced exactly.
LOOCV_8: tuple[str, ...] = (
    "Hellesylt", "Hundeidvik", "Leknes", "Linge", "Saebo",
    "Eidsdal", "Liabygda", "Stranda",
)

#: The current LOOCV set, and the set the paper reports.
LOOCV: tuple[str, ...] = LOOCV_8 + ("Festoya",)

#: Everything with a graph and features built, including the excluded structure.
#: Used by the graph and feature builders, which have no reason to exclude.
ALL: tuple[str, ...] = LOOCV + ("Sykkylven",)

#: Why a structure is not in LOOCV.
EXCLUDED: dict[str, str] = {
    "Sykkylven": (
        "Physically asymmetric balcony lengths.  The FE solver gained "
        "`length_left_mm` / `length_right_mm` support, but Sykkylven's model has "
        "not been re-parameterised and re-run with it, so its FE mode shapes are "
        "wrong at a geometric level.  The MAC pattern gives it away: bending poor "
        "(B1 0.603, B2 0.518), torsion coincidentally good (T1 0.963, T2 0.902)."
    ),
}

#: Modal targets, in the one order used everywhere.
MODES: tuple[str, ...] = ("B1", "T1", "B2", "T2")

#: MAC acceptance thresholds that deviate from the default during mode matching.
MAC_THRESHOLD_OVERRIDES: dict[str, float] = {"Eidsdal": 0.70}


def check(names) -> list[str]:
    """Validate a caller-supplied structure list against ALL.

    Raises ValueError naming the unknown entries, rather than failing later with
    a confusing missing-file error.
    """
    names = list(names)
    unknown = [n for n in names if n not in ALL]
    if unknown:
        raise ValueError(
            f"unknown structure(s): {unknown}. Known: {list(ALL)}"
        )
    return names
