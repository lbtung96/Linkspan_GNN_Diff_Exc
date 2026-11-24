"""T13 step 4 - zero-shot on the nine real bridges for every arm, through the paper's s5 path.

    python tools/run_stage.py pipeline/k_real.py

T11's h_real.part_zeroshot, imported and called unchanged, with h_common.T11_RESULTS re-pointed
in memory to results/T13_feasibility so its outputs (real_zeroshot.csv, real_gates.json) land
here.  That path is s5_zeroshot.load_structures() / evaluate() (the corrected Phase P field
inputs via fr_common.REAL_FEATURES, hybrid graphs, SSI labels and FE targets as s5; torch's
default threads), with the orientation input built from each hybrid graph's own edge_attr_geom.
Its gates: edgegat_N2000 reproduces s5_zeroshot.csv exactly; edgegat_N600 reproduces its
N = 600 rows of s5_transfer_vs_n.csv; s5's feature convention; the orientation input; s5's
input-dependence gate per network.
Added here: T11's two arms, re-scored, must equal T11's real_zeroshot.csv rows exactly.
"""

from __future__ import annotations

import json
import sys

import k_common as K

H = K.H
T11_ZS = H.T11_RESULTS / "real_zeroshot.csv"


def main() -> int:
    t11_rows = K.read_csv(T11_ZS)
    H.T11_RESULTS = K.T13_RESULTS
    import h_real

    rc = h_real.part_zeroshot(tuple(K.ALL_NETS), "")
    mine = K.read_csv(K.T13_RESULTS / "real_zeroshot.csv")
    g = json.loads((K.T13_RESULTS / "real_gates.json").read_text(encoding="utf-8"))
    out = {}
    for net in K.T11_ARMS:
        a = [r for r in mine if r["arm"] == net]
        b = [r for r in t11_rows if r["arm"] == net]
        num = [c for c in b[0] if c not in ("arm", "seed", "checkpoint", "structure", "mode", "edge_feature_mode")]
        keys_ok = [(r["structure"], r["mode"]) for r in a] == [(r["structure"], r["mode"]) for r in b]
        md = max(abs(float(x[c]) - float(y[c])) for x, y in zip(a, b) for c in num) if keys_ok else None
        out[net] = {"rows": len(a), "max_abs_diff": md, "pass": bool(keys_ok and len(a) == 36 and md == 0.0)}
    g["zeroshot"]["t11_arms_reproduce_t11"] = {"pass": all(v["pass"] for v in out.values()), "per_net": out}
    g["zeroshot"]["pass"] = all(v.get("pass") is not False for v in g["zeroshot"].values() if isinstance(v, dict))
    K.write_json(K.T13_RESULTS / "real_gates.json", g)
    print(json.dumps(out, indent=1))
    return 0 if (rc == 0 and g["zeroshot"]["pass"]) else 1


if __name__ == "__main__":
    sys.exit(main())
