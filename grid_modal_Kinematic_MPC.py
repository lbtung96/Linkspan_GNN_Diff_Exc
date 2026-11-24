"""Python translation of `grid_modal_demo_3d_offset_upgraded_v22.m`.

Units: N-mm-tonne.
DOF per node: [ux, uy, uz, rx, ry, rz].
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import eigsh


@dataclass
class Section:
    name: str
    A: float
    Iy: float
    Iz: float
    J: float
    ky: float
    kz: float


@dataclass
class Material:
    E: float
    G: float
    nu: float
    rho: float


def kNpm_to_Npmm(k_kN_per_m: float) -> float:
    return k_kN_per_m


def default_params_grid_3d_offset_upgraded_v22() -> dict:
    p: dict = {}
    p["L_mm"] = 14905 #14905(Magerholm)
    p["Nx"] = 16 #16(Magerholm)
    p["Ng"] = 5 #5(Magerholm)
    p["dy_mm"] = 2250 #2250(Magerholm)
    p["crossbeam_include_ends"] = False
    p["x_crossbeam_mm_list"] = None  # Set to a list of x-mm values to override uniform cross beam spacing.
    p["diag_pattern"] = "global_x" # Options: None (no bracing), globalX_attach/global_x, staggered_multi_x/staggered_x, single_partial_x_ng4/single_partial_x.

    p["E"] = 218900.0
    p["nu"] = 0.30
    p["rho"] = 7.039e-9
    p["G"] = p["E"] / (2 * (1 + p["nu"]))

    p["sec"] = {
        "girder": dict(A=2.0e4, Iy=1.6e9, Iz=5.0e8, J=3.0e7, ky=1.2, kz=1.2),
        "cross": dict(A=8.0e3, Iy=1.2e8, Iz=4.0e7, J=5.0e6, ky=1.2, kz=1.2),
        "diag": dict(A=4.5e3, Iy=1.0e7, Iz=1.0e7, J=2.0e6, ky=1.1, kz=1.1),
        "bottom": dict(A=1.35e4, Iy=7.0e8, Iz=2.5e8, J=2.0e7, ky=1.2, kz=1.2),
    }

    b_x, h_z = 890.0, 15.0
    p["sec"]["front"] = dict(
        A=b_x * h_z,
        Iy=(h_z * b_x**3) / 12,
        Iz=(b_x * h_z**3) / 12,
        J=(1 / 3) * b_x * h_z**3,
        ky=5 / 6,
        kz=5 / 6,
    )

    h, b, tw, tf = 380.0, 150.0, 11.7, 17.5
    hw = h - 2 * tf
    p["sec"]["balcony"] = dict(
        A=2 * (b * tf) + (tw * hw),
        Iz=2 * ((b * tf**3) / 12 + (b * tf) * ((h / 2 - tf / 2) ** 2)) + (tw * hw**3) / 12,
        Iy=2 * ((tf * b**3) / 12) + (hw * tw**3) / 12,
        J=(1 / 3) * (2 * b * tf**3 + hw * tw**3),
        ky=5 / 6,
        kz=5 / 6,
    )

    t, h = 15.0, 380.0
    p["sec"]["edge"] = dict(
        A=t * h,
        Iy=(t * h**3) / 12,
        Iz=(h * t**3) / 12,
        J=(1 / 3) * h * t**3,
        ky=5 / 6,
        kz=5 / 6,
    )

    p["offset"] = dict(cross_z=100.0, diag_z=-127.5, bottom_z=-472.0, front_z=115.0, balcony_z=70.0) # cross_z=100.0, diag_z=-127.5, bottom_z=-472.0, front_z=115.0, balcony_z=70.0 (Magerholm).
    p["diag_release"] = True
    p["rigidlink"] = dict(multA=1e3, multI=1e3, multJ=1e3, ky=1.2, kz=1.2)
    p["spring"] = dict(kx_kN_per_m=1862, kz_kN_per_m=18333) # kx_kN_per_m=2000, kz_kN_per_m=20000 (Magerholm)
    p["spring"]["kx_N_per_mm"] = kNpm_to_Npmm(p["spring"]["kx_kN_per_m"])
    p["spring"]["kz_N_per_mm"] = kNpm_to_Npmm(p["spring"]["kz_kN_per_m"])

    p["bottom_beam"] = dict(x_frac=9190 / 14905, overhang_mm=570) # x_frac=9190 / 14905, overhang_mm=570.0 (Magerholm)
    p["stabilize"] = dict(enable=True, k_rz_Nmm_per_rad=1e6, ref_girder_j=0, ref_x=0.0)
    p["balcony"] = dict(
        enable=True, #True (Magerholm)
        side="both", # Options: left, right, both #(Magerholm).
        x_center_frac= 13290 / 14905, # 13290 / 14905 (Magerholm)
        nBeams= 3, # 3 (Magerholm)
        dx_mm=950, # 950.0 (Magerholm)
        length_mm=2500, # 2500.0 (Magerholm)
        edge_beam_enable=True, # True (Magerholm)
        brace_enable=True, # True (Magerholm)
    )
    p["nsm"] = dict(
        enable=True,
        total_mass_tonne=30, # 30 (Magerholm)
        to_girders=True,
        to_cross=True,
        to_balcony=True,
        to_edge=False,
        to_brace=False,
        to_front=False,
        to_bottom=False,
        to_diagonals=False,
    )
    p["nmodes"] = 20
    p["plot"] = dict(nplot=15, deform_scale=0, z_visual_scale=3.0)
    p["geom_key_mm"] = 1e-6
    return p


def unique_tol(xs: np.ndarray, tol: float) -> np.ndarray:
    xs = np.sort(np.ravel(xs).astype(float))
    if xs.size == 0:
        return xs
    keep = np.ones(xs.size, dtype=bool)
    for k in range(1, xs.size):
        if abs(xs[k] - xs[k - 1]) < tol:
            keep[k] = False
    return xs[keep]


def interp_x_on_line(x0: float, y0: float, x1: float, y1: float, yq: float) -> float:
    t = (yq - y0) / (y1 - y0)
    return x0 + t * (x1 - x0)


def make_key(j_girder: int, x: float, key_scale_mm: float) -> int:
    xr = int(round(x / key_scale_mm))
    return int(j_girder) * 10**13 + xr


def mk_section(s: dict, name: str) -> Section:
    return Section(name=name, A=s["A"], Iy=s["Iy"], Iz=s["Iz"], J=s["J"], ky=s["ky"], kz=s["kz"])


def set_release_bits(rx1: bool, ry1: bool, rz1: bool, rx2: bool, ry2: bool, rz2: bool) -> int:
    bits = [rx1, ry1, rz1, rx2, ry2, rz2]
    rel = 0
    for k, bit in enumerate(bits):
        if bit:
            rel |= 1 << k
    return rel


def compute_nsm_distribution(params: dict, nodes: np.ndarray, elems: np.ndarray, groups: dict) -> dict:
    nsm = dict(enable=bool(params.get("nsm", {}).get("enable", False)), total_mass_tonne=0.0,
               m_add_per_length_tonne_per_mm=0.0, groups=np.zeros(20, dtype=bool), total_target_length_mm=0.0)
    if not nsm["enable"]:
        return nsm

    npm = params["nsm"]
    nsm["total_mass_tonne"] = npm["total_mass_tonne"]
    gmask = np.zeros(20, dtype=bool)
    flags = [
        ("to_girders", groups["G_GIRDER"]), ("to_cross", groups["G_CROSS"]), ("to_balcony", groups["G_BALCONY"]),
        ("to_edge", groups["G_EDGE"]), ("to_brace", groups["G_BBRACE"]), ("to_front", groups["G_FRONT"]),
        ("to_bottom", groups["G_BOTTOM"]), ("to_diagonals", groups["G_DIAG"]),
    ]
    for key, g in flags:
        if npm.get(key, False):
            gmask[g - 1] = True

    ltot = 0.0
    for e in elems:
        grp, mat_id = int(e[3]), int(e[5])
        if mat_id != 1:
            continue
        if grp < 1 or grp > gmask.size or not gmask[grp - 1]:
            continue
        n1, n2 = int(e[1]) - 1, int(e[2]) - 1
        ltot += np.linalg.norm(nodes[n2, 1:4] - nodes[n1, 1:4])

    if ltot <= 0:
        return nsm

    nsm["groups"] = gmask
    nsm["total_target_length_mm"] = ltot
    nsm["m_add_per_length_tonne_per_mm"] = nsm["total_mass_tonne"] / ltot
    return nsm


def _pick_near_far_crossbeam_x(x_cb: np.ndarray, x_bottom: float) -> Tuple[float, float]:
    if x_cb.size == 0:
        raise ValueError("No crossbeam x-locations are available for diagonal bracing.")
    d = np.abs(x_cb - x_bottom)
    x_near = float(x_cb[int(np.argmin(d))])
    x_far = float(x_cb[int(np.argmax(d))])
    return x_near, x_far


def _normalize_diag_pattern(diag_pattern: Optional[object]) -> Optional[str]:
    """Normalize diagonal-bracing pattern names. ``None`` means no diagonal bracing."""
    if diag_pattern is None:
        return None
    pattern = str(diag_pattern)
    if pattern.strip().lower() in ("none", ""):
        return None
    alias_map = {
        "globalX_attach": "global_x",
        "global_x": "global_x",
        "staggered_multi_x": "staggered_multi_x",
        "staggered_x": "staggered_multi_x",
        "single_partial_x_ng4": "single_partial_x_ng4",
        "single_partial_x": "single_partial_x_ng4",
    }
    if pattern not in alias_map:
        raise ValueError(
            f"Unknown diag_pattern '{pattern}'. Supported: None, global_x/globalX_attach, "
            "staggered_multi_x/staggered_x, single_partial_x_ng4/single_partial_x."
        )
    return alias_map[pattern]


def _diag_line_specs_from_pattern(params: dict, x_cb: np.ndarray, x_bottom: float, x0: float, x1: float, Ng: int) -> List[Tuple[int, float, int, float]]:
    pattern = _normalize_diag_pattern(params.get("diag_pattern", "globalX_attach"))
    if pattern is None:
        return []
    x_near, x_far = _pick_near_far_crossbeam_x(x_cb, x_bottom)
    x_first = float(np.min(x_cb))
    x_last = float(np.max(x_cb))
    j_top, j_bot = Ng - 1, 0
    j_mid = Ng // 2

    if pattern == "global_x":
        return [(j_bot, x0, j_top, x1), (j_bot, x1, j_top, x0)]

    if pattern == "staggered_multi_x":
        if Ng < 2:
            raise ValueError("staggered_multi_x requires at least 2 girders.")
        lines = [
            (j_top, x_far, j_bot, x_near),
            (j_top, x_near, j_bot, x_far),
        ]
        if Ng % 2 == 1:
            # Odd Ng: first X plus two extra branches meeting at middle/far ("><>").
            lines.append((j_top, x_near, j_mid, x_last))
            lines.append((j_bot, x_near, j_mid, x_last))
        else:
            # Even Ng: add a second full X ("><><"), outer-to-outer.
            lines.append((j_top, x_near, j_bot, x_last))
            lines.append((j_bot, x_near, j_top, x_last))
        return lines

    if pattern == "single_partial_x_ng4":
        if Ng != 4:
            raise ValueError("single_partial_x_ng4 is defined only for Ng == 4.")
        # "3rd girder from left->right in y direction" with left at max-j side.
        j_target = Ng - 3
        return [(j_top, x_last, j_target, x_near)]

    raise ValueError(f"Unsupported normalized diag_pattern '{pattern}'.")


def build_grid_model_3d_upgraded_v22(params: dict) -> dict:
    L, Nx, Ng, dy = params["L_mm"], params["Nx"], params["Ng"], params["dy_mm"]
    dx = L / Nx
    x_base = np.arange(Nx + 1) * dx

    i0, i1 = (0, Nx) if params["crossbeam_include_ends"] else (1, Nx - 1)
    x0, x1 = i0 * dx, i1 * dx
    x_bottom = params["bottom_beam"]["x_frac"] * L

    x_balcony = np.array([])
    b = params.get("balcony", {})
    if b.get("enable", False):
        xc, nb, dxB = b["x_center_frac"] * L, b["nBeams"], b["dx_mm"]
        if nb == 1:
            x_balcony = np.array([xc])
        elif nb > 1:
            x_balcony = xc + (np.arange(nb) - (nb - 1) / 2) * dxB

    x_brace_anchor = np.array([])
    if b.get("enable", False) and b.get("brace_enable", False) and x_balcony.size >= 1:
        x_brace_anchor = np.array([x_balcony.min() - b["dx_mm"]])

    if params.get("x_crossbeam_mm_list") is not None:
        x_cb = unique_tol(np.asarray(params["x_crossbeam_mm_list"], dtype=float), 1e-9)
    else:
        x_cb = x_base if params["crossbeam_include_ends"] else x_base[1:-1]

    x_list: List[np.ndarray] = []
    for _ in range(Ng):
        xs = unique_tol(np.r_[x_base, x_cb, x_bottom, x_balcony, x_brace_anchor], 1e-9)
        x_list.append(xs)

    diag_pattern_name = _normalize_diag_pattern(params.get("diag_pattern", "globalX_attach"))
    single_partial_direct = diag_pattern_name == "single_partial_x_ng4"
    diag_line_specs = _diag_line_specs_from_pattern(params, x_cb, x_bottom, x0, x1, Ng)
    for j0, xs, j1, xe in diag_line_specs:
        if j0 == j1:
            continue
        y0, y1 = j0 * dy, j1 * dy
        if single_partial_direct:
            j_ids = [j0, j1]
        else:
            j_min, j_max = min(j0, j1), max(j0, j1)
            j_ids = list(range(j_min, j_max + 1))
        for j in j_ids:
            yj = j * dy
            xj = interp_x_on_line(xs, y0, xe, y1, yj)
            x_list[j] = unique_tol(np.r_[x_list[j], xj], 1e-9)

    key_scale = max(params["geom_key_mm"], 1e-6)
    node_map: Dict[int, int] = {}
    nodes: List[List[float]] = []

    for j in range(Ng):
        y = j * dy
        for x in x_list[j]:
            nid = len(nodes) + 1
            nodes.append([nid, float(x), float(y), 0.0])
            node_map[make_key(j, float(x), key_scale)] = nid

    def getN(j: int, x: float) -> int:
        return node_map[make_key(j, float(x), key_scale)]

    sections: Dict[int, Section] = {
        1: mk_section(params["sec"]["girder"], "girder"),
        2: mk_section(params["sec"]["cross"], "cross"),
        3: mk_section(params["sec"]["diag"], "diag"),
        5: mk_section(params["sec"]["bottom"], "bottom"),
        6: mk_section(params["sec"]["front"], "front"),
        7: mk_section(params["sec"]["balcony"], "balcony"),
        8: mk_section(params["sec"]["edge"], "edge"),
        9: mk_section(params["sec"]["edge"], "balcony_brace"),
    }
    materials = {1: Material(params["E"], params["G"], params["nu"], params["rho"]), 2: Material(params["E"], params["G"], params["nu"], 0.0)}
    G_GIRDER, G_CROSS, G_DIAG, G_RIGID, G_BOTTOM, G_FRONT, G_BALCONY, G_EDGE, G_BBRACE = (1, 2, 3, 4, 5, 6, 7, 8, 9)
    groups = locals()

    elems: List[List[float]] = []
    caches = {k: {} for k in ["cross", "diag", "bottom", "front", "balcony"]}
    mpcs: List[List[int]] = []

    def add_elem(n1: int, n2: int, grp: int, sec: int, mat: int, rel: int = 0) -> None:
        elems.append([len(elems) + 1, n1, n2, grp, sec, mat, rel])

    nodes_arr = np.array(nodes, dtype=float)

    def get_offset_node_and_link(base_nid: int, zoff: float, cache: dict) -> int:
        if base_nid in cache:
            return cache[base_nid]
        xb, yb, zb = nodes_arr[base_nid - 1, 1:4]
        off_id = nodes_arr.shape[0] + len(nonlocal_nodes) + 1
        nonlocal_nodes.append([off_id, xb, yb, zb + zoff])
        cache[base_nid] = off_id
        mpcs.append([base_nid, off_id])
        return off_id

    nonlocal_nodes: List[List[float]] = []

    for j in range(Ng):
        xs = x_list[j]
        for k in range(xs.size - 1):
            add_elem(getN(j, xs[k]), getN(j, xs[k + 1]), G_GIRDER, 1, 1)

    for xi in x_cb:
        for j in range(Ng - 1):
            nA = get_offset_node_and_link(getN(j, xi), params["offset"]["cross_z"], caches["cross"])
            nB = get_offset_node_and_link(getN(j + 1, xi), params["offset"]["cross_z"], caches["cross"])
            add_elem(nA, nB, G_CROSS, 2, 1)

    rel_diag = set_release_bits(True, True, True, True, True, True) if params["diag_release"] else 0

    def add_diag_line(j0: int, xs: float, j1: int, xe: float) -> None:
        if j0 == j1:
            return
        y0, y1 = j0 * dy, j1 * dy
        pts_xy = []
        if single_partial_direct:
            j_ids = [j0, j1]
        else:
            j_min, j_max = min(j0, j1), max(j0, j1)
            j_ids = list(range(j_min, j_max + 1))
        for jg in j_ids:
            yg = jg * dy
            xg = interp_x_on_line(xs, y0, xe, y1, yg)
            pts_xy.append((xg, yg))

        off_nodes = []
        pts = np.array(pts_xy, dtype=float)
        for xk, yk in pts[np.argsort(pts[:, 1])]:
            jg = int(round(yk / dy))
            base_n = getN(jg, xk)
            if base_n in caches["diag"]:
                off_n = caches["diag"][base_n]
            else:
                off_n = get_offset_node_and_link(base_n, params["offset"]["diag_z"], caches["diag"])
            off_nodes.append(off_n)
        for k in range(len(off_nodes) - 1):
            add_elem(off_nodes[k], off_nodes[k + 1], G_DIAG, 3, 1, rel_diag)

    for j0, xs, j1, xe in diag_line_specs:
        add_diag_line(j0, xs, j1, xe)

    y_outer_min, y_outer_max = 0.0, (Ng - 1) * dy
    yL, yR = y_outer_min - params["bottom_beam"]["overhang_mm"], y_outer_max + params["bottom_beam"]["overhang_mm"]
    y_nodes = np.r_[yL, np.arange(Ng) * dy, yR]
    bottom_node_ids: List[int] = []
    for yk in y_nodes:
        if abs(yk - yL) < 1e-9 or abs(yk - yR) < 1e-9:
            nid_new = nodes_arr.shape[0] + len(nonlocal_nodes) + 1
            nonlocal_nodes.append([nid_new, x_bottom, yk, params["offset"]["bottom_z"]])
            bottom_node_ids.append(nid_new)
        else:
            jg = int(round(yk / dy))
            bottom_node_ids.append(get_offset_node_and_link(getN(jg, x_bottom), params["offset"]["bottom_z"], caches["bottom"]))
    for k in range(len(bottom_node_ids) - 1):
        add_elem(bottom_node_ids[k], bottom_node_ids[k + 1], G_BOTTOM, 5, 1)

    front_nodes = [get_offset_node_and_link(getN(j, L), params["offset"]["front_z"], caches["front"]) for j in range(Ng)]
    for j in range(Ng - 1):
        add_elem(front_nodes[j], front_nodes[j + 1], G_FRONT, 6, 1)

    balcony_left_out, balcony_right_out = [], []
    if b.get("enable", False) and x_balcony.size:
        add_left = b["side"] in ("left", "both")
        add_right = b["side"] in ("right", "both")
        len_left  = float(b.get("length_left_mm",  b.get("length_mm", 0.0)))
        len_right = float(b.get("length_right_mm", b.get("length_mm", 0.0)))
        for xi in x_balcony:
            if add_left:
                in_off = get_offset_node_and_link(getN(0, xi), params["offset"]["balcony_z"], caches["balcony"])
                out_n = nodes_arr.shape[0] + len(nonlocal_nodes) + 1
                nonlocal_nodes.append([out_n, xi, y_outer_min - len_left, params["offset"]["balcony_z"]])
                add_elem(in_off, out_n, G_BALCONY, 7, 1)
                balcony_left_out.append((xi, out_n))
            if add_right:
                in_off = get_offset_node_and_link(getN(Ng - 1, xi), params["offset"]["balcony_z"], caches["balcony"])
                out_n = nodes_arr.shape[0] + len(nonlocal_nodes) + 1
                nonlocal_nodes.append([out_n, xi, y_outer_max + len_right, params["offset"]["balcony_z"]])
                add_elem(in_off, out_n, G_BALCONY, 7, 1)
                balcony_right_out.append((xi, out_n))

        if b.get("edge_beam_enable", False):
            for side in (sorted(balcony_left_out), sorted(balcony_right_out)):
                for k in range(len(side) - 1):
                    add_elem(side[k][1], side[k + 1][1], G_EDGE, 8, 1)

        if b.get("brace_enable", False) and x_balcony.size:
            xmin, x_target = float(x_balcony.min()), float(x_balcony.min() - b["dx_mm"])
            for lst, j in [(balcony_left_out, 0), (balcony_right_out, Ng - 1)]:
                if not lst:
                    continue
                matches = [n for x, n in lst if abs(x - xmin) < 1e-6]
                if matches:
                    n_in = get_offset_node_and_link(getN(j, x_target), params["offset"]["balcony_z"], caches["balcony"])
                    add_elem(n_in, matches[0], G_BBRACE, 9, 1)

    if nonlocal_nodes:
        nodes_arr = np.vstack([nodes_arr, np.array(nonlocal_nodes, dtype=float)])

    Nn = nodes_arr.shape[0]
    dof_map = np.arange(1, 6 * Nn + 1).reshape(Nn, 6)

    bc = []
    for j in range(Ng):
        n0 = getN(j, 0.0)
        bc += [[n0, 2, 0], [n0, 3, 0]]

    springs = [[getN(j, 0.0), 1, params["spring"]["kx_N_per_mm"]] for j in range(Ng)]
    springs += [[bottom_node_ids[0], 3, params["spring"]["kz_N_per_mm"]], [bottom_node_ids[-1], 3, params["spring"]["kz_N_per_mm"]]]
    if params.get("stabilize", {}).get("enable", False):
        nref = getN(params["stabilize"]["ref_girder_j"], params["stabilize"]["ref_x"])
        springs.append([nref, 6, params["stabilize"]["k_rz_Nmm_per_rad"]])

    elems_arr = np.array(elems, dtype=float)
    mpc_pairs = np.array(mpcs, dtype=int) if len(mpcs) else np.zeros((0, 2), dtype=int)
    nsm = compute_nsm_distribution(params, nodes_arr, elems_arr, groups)
    n_slave = int(np.unique(mpc_pairs[:, 1]).size) if mpc_pairs.size else 0
    ndof_full = 6 * Nn
    ndof_red = ndof_full - 6 * n_slave
    no_rigid_group4 = bool(not np.any(elems_arr[:, 3] == 4)) if elems_arr.size else True
    no_rigid_sec4 = bool(not np.any(elems_arr[:, 4] == 4)) if elems_arr.size else True
    diag_elem_count = int(np.count_nonzero(elems_arr[:, 3] == G_DIAG)) if elems_arr.size else 0
    print(
        f"Model check: Nn={Nn}, rigid_mpc_slaves={n_slave}, "
        f"full_ndof={ndof_full}, ndof_red={ndof_red}, ndof_reduced_ok={ndof_red <= ndof_full and ndof_red > 0}"
    )
    print(f"no_rigid_group4={no_rigid_group4}, no_rigid_sec4={no_rigid_sec4}")
    print(f"diag_pattern = {params.get('diag_pattern', 'globalX_attach')}")
    print(f"diagonal braces = {diag_elem_count}")
    return dict(
        nodes=nodes_arr,
        elems=elems_arr,
        Ng=int(Ng),
        Nx=int(Nx),
        L_mm=float(L),
        dy_mm=float(dy),
        sections=sections,
        materials=materials,
        bc=np.array(bc, dtype=float),
        dof=dict(ndof=6 * Nn, map=dof_map),
        springs=np.array(springs, dtype=float),
        nsm=nsm,
        bottom_node_ids=np.array(bottom_node_ids, dtype=int),
        diag_pattern=params.get("diag_pattern", "globalX_attach"),
        diag_elem_count=diag_elem_count,
        diag_line_specs=np.array(diag_line_specs, dtype=float) if len(diag_line_specs) else np.zeros((0, 4), dtype=float),
        mpc_pairs=mpc_pairs,
        vref_primary=np.array([0.0, 0.0, 1.0]),
        vref_fallback=np.array([0.0, 1.0, 0.0]),
    )


def timoshenko_bending_2d(EI: float, L: float, phi: float) -> np.ndarray:
    fac = EI / (L**3 * (1 + phi))
    return fac * np.array([
        [12, 6 * L, -12, 6 * L],
        [6 * L, (4 + phi) * L**2, -6 * L, (2 - phi) * L**2],
        [-12, -6 * L, 12, -6 * L],
        [6 * L, (2 - phi) * L**2, -6 * L, (4 + phi) * L**2],
    ])


def eb_bending_mass_2d(m: float, L: float) -> np.ndarray:
    return (m / 420) * np.array([
        [156, 22 * L, 54, -13 * L],
        [22 * L, 4 * L**2, 13 * L, -3 * L**2],
        [54, 13 * L, 156, -22 * L],
        [-13 * L, -3 * L**2, -22 * L, 4 * L**2],
    ])


def beam3d_timoshenko_KM_local(sec: Section, mat: Material, L: float, m_add_per_len: float) -> Tuple[np.ndarray, np.ndarray]:
    E, G, rho = mat.E, mat.G, mat.rho
    A, Iy, Iz, J, ky, kz = sec.A, sec.Iy, sec.Iz, sec.J, sec.ky, sec.kz
    ke = np.zeros((12, 12))
    me = np.zeros((12, 12))

    ids = np.ix_([0, 6], [0, 6])
    ke[ids] += np.array([[1, -1], [-1, 1]]) * (E * A / L)
    ids = np.ix_([3, 9], [3, 9])
    ke[ids] += np.array([[1, -1], [-1, 1]]) * (G * J / L)

    kb = timoshenko_bending_2d(E * Iz, L, (12 * E * Iz) / (ky * G * A * L**2))
    idx = [1, 5, 7, 11]
    ke[np.ix_(idx, idx)] += kb

    kb = timoshenko_bending_2d(E * Iy, L, (12 * E * Iy) / (kz * G * A * L**2))
    idx = [2, 4, 8, 10]
    ke[np.ix_(idx, idx)] += kb

    if rho <= 0 and m_add_per_len <= 0:
        return ke, me

    m = (rho * A + max(0.0, m_add_per_len)) * L
    me[np.ix_([0, 6], [0, 6])] += (m / 6) * np.array([[2, 1], [1, 2]])
    me[np.ix_([1, 5, 7, 11], [1, 5, 7, 11])] += eb_bending_mass_2d(m, L)
    me[np.ix_([2, 4, 8, 10], [2, 4, 8, 10])] += eb_bending_mass_2d(m, L)
    if rho > 0:
        Jm = rho * (Iy + Iz) * L
        me[np.ix_([3, 9], [3, 9])] += (Jm / 6) * np.array([[2, 1], [1, 2]])
    return ke, me


def apply_end_releases_local(K: np.ndarray, M: np.ndarray, rel_code: int) -> Tuple[np.ndarray, np.ndarray]:
    rel = np.zeros(12, dtype=bool)
    mapping = [3, 4, 5, 9, 10, 11]
    for k in range(6):
        if rel_code & (1 << k):
            rel[mapping[k]] = True
    kept, rels = np.where(~rel)[0], np.where(rel)[0]
    if rels.size == 0:
        return K, M

    Kkk, Mkk = K[np.ix_(kept, kept)], M[np.ix_(kept, kept)]
    Kkr, Krk, Krr = K[np.ix_(kept, rels)], K[np.ix_(rels, kept)], K[np.ix_(rels, rels)]
    Mkr, Mrk, Mrr = M[np.ix_(kept, rels)], M[np.ix_(rels, kept)], M[np.ix_(rels, rels)]

    if np.linalg.cond(Krr) > 1e12:
        Krr = Krr + (1e-9 * max(1.0, np.linalg.norm(Krr, "fro"))) * np.eye(Krr.shape[0])
    Kcond = Kkk - Kkr @ np.linalg.solve(Krr, Krk)

    if Mrr.size == 0 or np.all(np.abs(Mrr) < 1e-30):
        Mcond = Mkk
    else:
        if np.linalg.cond(Mrr) > 1e12:
            Mrr = Mrr + (1e-12 * max(1.0, np.linalg.norm(Mrr, "fro"))) * np.eye(Mrr.shape[0])
        Mcond = Mkk - Mkr @ np.linalg.solve(Mrr, Mrk)

    Kc, Mc = np.zeros((12, 12)), np.zeros((12, 12))
    Kc[np.ix_(kept, kept)] = Kcond
    Mc[np.ix_(kept, kept)] = Mcond
    return Kc, Mc


def pick_vref_for_element(d: np.ndarray, grp: int, sec_id: int, model: dict) -> np.ndarray:
    _ = d, sec_id
    vref = model["vref_primary"]
    if grp in (1, 2, 9):
        vref = np.array([0.0, 0.0, 1.0])
    elif grp in (7, 8):
        vref = np.array([1.0, 0.0, 0.0])
    return vref


def local_rotation_rule1(
    d: np.ndarray,
    vref_primary: np.ndarray,
    vref_fallback: np.ndarray,
    vref_override: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, dict]:
    xhat = d / np.linalg.norm(d)
    debug_axes = False

    vref = (vref_override if vref_override is not None else vref_primary).reshape(-1)
    ytmp = np.cross(vref, xhat)
    if np.linalg.norm(ytmp) < 1e-12:
        vref = vref_fallback.reshape(-1)
        ytmp = np.cross(vref, xhat)
        if np.linalg.norm(ytmp) < 1e-12:
            raise ValueError("Failed local axes")

    yhat = ytmp / np.linalg.norm(ytmp)
    zhat = np.cross(xhat, yhat)
    zhat = zhat / np.linalg.norm(zhat)

    if debug_axes:
        # Optional orientation sanity prints.
        if abs(xhat[0]) > 0.99 and abs(xhat[1]) < 1e-2:
            print(f"[axes-debug X] xhat={xhat}, yhat={yhat}, zhat={zhat}")
        if abs(xhat[1]) > 0.99 and abs(xhat[0]) < 1e-2:
            print(f"[axes-debug Y] xhat={xhat}, yhat={yhat}, zhat={zhat}")

    R = np.column_stack([xhat, yhat, zhat])
    return R, dict(xhat=xhat, yhat=yhat, zhat=zhat, vref=vref)


def T_beam3d(R: np.ndarray) -> np.ndarray:
    Z = np.zeros((3, 3))
    Tn = np.block([[R, Z], [Z, R]])
    return np.block([[Tn, np.zeros((6, 6))], [np.zeros((6, 6)), Tn]])


def assemble_KM_3d_with_nsm(model: dict) -> Tuple[csc_matrix, csc_matrix]:
    nodes, elems, ndof = model["nodes"], model["elems"], model["dof"]["ndof"]
    rows: List[int] = []
    cols: List[int] = []
    valsK: List[float] = []
    valsM: List[float] = []
    skipped_zero_len = 0

    nsm_enable = model.get("nsm", {}).get("enable", False)
    m_add = model.get("nsm", {}).get("m_add_per_length_tonne_per_mm", 0.0)
    gmask = model.get("nsm", {}).get("groups", np.zeros(20, dtype=bool))
    balcony_edge_override_count = 0
    balcony_rep_printed = False

    for e in elems:
        n1, n2 = int(e[1]), int(e[2])
        grp, sec_id, mat_id, rel_code = int(e[3]), int(e[4]), int(e[5]), int(e[6])
        sec, mat = model["sections"][sec_id], model["materials"][mat_id]
        x1, x2 = nodes[n1 - 1, 1:4], nodes[n2 - 1, 1:4]
        d = x2 - x1
        Le = np.linalg.norm(d)
        if Le <= 1e-12:
            skipped_zero_len += 1
            continue
        vref = pick_vref_for_element(d, grp, sec_id, model)
        R, dbg = local_rotation_rule1(d, model["vref_primary"], model["vref_fallback"], vref_override=vref)
        if grp in (7, 8):
            balcony_edge_override_count += 1
        if (not balcony_rep_printed) and grp == 7:
            print(f"[vref-check balcony] xhat={dbg['xhat']}, vref={dbg['vref']}")
            balcony_rep_printed = True

        madd = m_add if (nsm_enable and mat_id == 1 and 1 <= grp <= gmask.size and gmask[grp - 1]) else 0.0
        keL, meL = beam3d_timoshenko_KM_local(sec, mat, Le, madd)
        if rel_code != 0:
            keL, meL = apply_end_releases_local(keL, meL, rel_code)
        T = T_beam3d(R)
        keG, meG = T.T @ keL @ T, T.T @ meL @ T

        dofs = np.r_[model["dof"]["map"][n1 - 1], model["dof"]["map"][n2 - 1]] - 1
        rr, cc = np.meshgrid(dofs, dofs, indexing="ij")
        rows.extend(rr.ravel().tolist())
        cols.extend(cc.ravel().tolist())
        valsK.extend(keG.ravel().tolist())
        valsM.extend(meG.ravel().tolist())

    K = coo_matrix((valsK, (rows, cols)), shape=(ndof, ndof)).tocsc()
    M = coo_matrix((valsM, (rows, cols)), shape=(ndof, ndof)).tocsc()
    K = ((K + K.T) * 0.5).tocsc()
    M = ((M + M.T) * 0.5).tocsc()
    if skipped_zero_len > 0:
        print(f"Warning: skipped {skipped_zero_len} near-zero-length element(s) during assembly.")
    print(f"[vref-check] balcony_edge_overrides={balcony_edge_override_count}")
    return K, M


def add_nodal_springs(K: csc_matrix, model: dict) -> csc_matrix:
    K = K.tolil(copy=True)
    for nid, dof, kval in model.get("springs", np.zeros((0, 3))):
        gdof = int(6 * (int(nid) - 1) + int(dof) - 1)
        K[gdof, gdof] += float(kval)
    return K.tocsc()


def build_rigid_mpc_T(model: dict) -> Tuple[csc_matrix, np.ndarray]:
    ndof_full = int(model["dof"]["ndof"])
    mpc_pairs = np.asarray(model.get("mpc_pairs", np.zeros((0, 2), dtype=int)), dtype=int)
    if mpc_pairs.size == 0:
        eye_idx = np.arange(ndof_full, dtype=int)
        T = coo_matrix((np.ones(ndof_full), (eye_idx, eye_idx)), shape=(ndof_full, ndof_full)).tocsc()
        return T, eye_idx

    nodes = model["nodes"]
    slave_nodes = np.unique(mpc_pairs[:, 1]).astype(int)
    is_slave = np.zeros(ndof_full, dtype=bool)
    for sn in slave_nodes:
        base = 6 * (sn - 1)
        is_slave[base : base + 6] = True

    keep_full_dofs = np.where(~is_slave)[0]
    full_to_red = np.full(ndof_full, -1, dtype=int)
    full_to_red[keep_full_dofs] = np.arange(keep_full_dofs.size, dtype=int)

    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []

    for gdof in keep_full_dofs:
        rows.append(int(gdof))
        cols.append(int(full_to_red[gdof]))
        vals.append(1.0)

    for master_nid, slave_nid in mpc_pairs:
        master_nid = int(master_nid)
        slave_nid = int(slave_nid)
        r = nodes[slave_nid - 1, 1:4] - nodes[master_nid - 1, 1:4]
        dx, dy, dz = float(r[0]), float(r[1]), float(r[2])
        C = np.array([[0.0, dz, -dy], [-dz, 0.0, dx], [dy, -dx, 0.0]])
        A = np.block([[np.eye(3), C], [np.zeros((3, 3)), np.eye(3)]])
        mb = 6 * (master_nid - 1)
        sb = 6 * (slave_nid - 1)
        for lr in range(6):
            row_full = sb + lr
            for lc in range(6):
                coeff = float(A[lr, lc])
                if abs(coeff) < 1e-18:
                    continue
                master_full = mb + lc
                red_col = full_to_red[master_full]
                if red_col < 0:
                    raise ValueError(f"Invalid MPC: master node {master_nid} DOF {lc + 1} is not independent.")
                rows.append(row_full)
                cols.append(int(red_col))
                vals.append(coeff)

    T = coo_matrix((vals, (rows, cols)), shape=(ndof_full, keep_full_dofs.size)).tocsc()
    return T, keep_full_dofs


def solve_modes_3d(
    K: csc_matrix,
    M: csc_matrix,
    bc: np.ndarray,
    nmodes: int,
    full_to_red: np.ndarray,
    T: Optional[csc_matrix] = None,
    M_full: Optional[csc_matrix] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    ndof_red = K.shape[0]
    fixed = np.zeros(ndof_red, dtype=bool)
    for nid, dof, _ in bc:
        gdof_full = int(6 * (int(nid) - 1) + int(dof) - 1)
        if gdof_full < 0 or gdof_full >= full_to_red.size:
            continue
        gdof_red = int(full_to_red[gdof_full])
        if gdof_red < 0:
            print(f"Warning: BC on slave DOF ignored (node={int(nid)}, dof={int(dof)}).")
            continue
        fixed[gdof_red] = True
    free = np.where(~fixed)[0]

    Kff, Mff = K[free][:, free], M[free][:, free]
    sigma = (2 * np.pi * 0.5) ** 2
    if Kff.shape[0] <= 2:
        raise ValueError("Not enough free DOFs after BC/MPC reduction to solve modes.")
    k_use = min(nmodes, Kff.shape[0] - 2)
    vals, vecs = eigsh(Kff, k=k_use, M=Mff, sigma=sigma, which="LM", tol=1e-10, maxiter=6000)

    idx = np.argsort(np.real(vals))
    lam = np.real(vals[idx])
    V = vecs[:, idx]
    lam[lam < 0] = 0
    freq_hz = np.sqrt(lam) / (2 * np.pi)

    Phi_red = np.zeros((ndof_red, k_use))
    Phi_red[free, :] = V
    if T is not None:
        Phi = T @ Phi_red
    else:
        Phi = Phi_red

    Mnorm = M_full if M_full is not None else M
    for k in range(k_use):
        phi = Phi[:, k]
        mn = np.real(phi @ (Mnorm @ phi))
        if mn > 0:
            Phi[:, k] = phi / np.sqrt(mn)
    return freq_hz, Phi, {"free_red": free, "fixed_red": np.where(fixed)[0]}


def set_axes_equal_3d(ax, x: np.ndarray, y: np.ndarray, z: np.ndarray, z_visual_scale: float = 1.0) -> None:
    xmn, xmx = float(np.min(x)), float(np.max(x))
    ymn, ymx = float(np.min(y)), float(np.max(y))
    zmn, zmx = float(np.min(z)), float(np.max(z))
    xr = max(xmx - xmn, 1.0)
    yr = max(ymx - ymn, 1.0)
    zr = max(zmx - zmn, 1.0)
    pad = 0.02 * max(xr, yr, zr)

    ax.set_xlim(xmn - pad, xmx + pad)
    ax.set_ylim(ymn - pad, ymx + pad)
    ax.set_zlim(zmn - pad, zmx + pad)
    z_scale = max(float(z_visual_scale), 1e-9)
    ax.set_box_aspect((xr, yr, z_scale * zr))


def plot_model_3d(model: dict, ax, params: Optional[dict] = None) -> None:
    nodes = model["nodes"]
    elems = model["elems"]
    grey = "0.65"

    for e in elems:
        n1, n2 = int(e[1]) - 1, int(e[2]) - 1
        ax.plot(
            [nodes[n1, 1], nodes[n2, 1]],
            [nodes[n1, 2], nodes[n2, 2]],
            [nodes[n1, 3], nodes[n2, 3]],
            color=grey,
            lw=1.1,
        )
    ax.plot(nodes[:, 1], nodes[:, 2], nodes[:, 3], "k.", markersize=6)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.view_init(elev=20, azim=35)
    z_scale = float(params.get("plot", {}).get("z_visual_scale", 1.0)) if params is not None else 1.0
    set_axes_equal_3d(ax, nodes[:, 1], nodes[:, 2], nodes[:, 3], z_visual_scale=z_scale)


def plot_mode_3d(model: dict, phi: np.ndarray, ttl: str, params: dict, ax) -> None:
    nodes = model["nodes"]
    elems = model["elems"]
    nn = nodes.shape[0]

    ux = np.zeros(nn)
    uy = np.zeros(nn)
    uz = np.zeros(nn)
    for nid in range(nn):
        base = 6 * nid
        ux[nid] = phi[base + 0]
        uy[nid] = phi[base + 1]
        uz[nid] = phi[base + 2]

    mag = np.max(np.sqrt(ux**2 + uy**2 + uz**2))
    if mag < 1e-18:
        scale = 1.0
    else:
        if params["plot"]["deform_scale"] > 0:
            scale = float(params["plot"]["deform_scale"])
        else:
            span = float(np.max(nodes[:, 1]) - np.min(nodes[:, 1]))
            scale = 0.04 * span / mag

    xd = nodes[:, 1] + scale * ux
    yd = nodes[:, 2] + scale * uy
    zd = nodes[:, 3] + scale * uz

    grey = "0.65"
    blue = "#0072BD"

    for e in elems:
        n1, n2 = int(e[1]) - 1, int(e[2]) - 1
        ax.plot(
            [nodes[n1, 1], nodes[n2, 1]],
            [nodes[n1, 2], nodes[n2, 2]],
            [nodes[n1, 3], nodes[n2, 3]],
            color=grey,
            lw=1.0,
        )
    ax.plot(nodes[:, 1], nodes[:, 2], nodes[:, 3], "k.", markersize=6)

    for e in elems:
        n1, n2 = int(e[1]) - 1, int(e[2]) - 1
        ax.plot([xd[n1], xd[n2]], [yd[n1], yd[n2]], [zd[n1], zd[n2]], color=blue, lw=1.7)
    ax.plot(xd, yd, zd, ".", color=blue, markersize=5)

    ax.set_title(ttl)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.view_init(elev=20, azim=35)
    z_scale = float(params.get("plot", {}).get("z_visual_scale", 1.0))
    set_axes_equal_3d(
        ax,
        np.r_[nodes[:, 1], xd],
        np.r_[nodes[:, 2], yd],
        np.r_[nodes[:, 3], zd],
        z_visual_scale=z_scale,
    )


def run_grid_modal_with_params_and_plot(
    params: dict,
    *,
    do_plot: bool = True,
    run_modal: Optional[bool] = True,
) -> dict:
    """Run the standard modal workflow for a provided parameter dictionary."""
    params = dict(params)
    model = build_grid_model_3d_upgraded_v22(params)

    if do_plot:
        import matplotlib.pyplot as plt
        print("Plotting undeformed model for bracing check...")
        fig = plt.figure("Undeformed Model (3D)")
        ax = fig.add_subplot(111, projection="3d")
        plot_model_3d(model, ax, params=params)
        ax.set_title(f"Undeformed upgraded grid model (v2.2) - {model.get('diag_pattern', 'diag')}")
        plt.show()

    if run_modal is None:
        if do_plot:
            ans = input("Run modal analysis now? [y/N]: ").strip().lower()
            run_modal = ans in ("y", "yes", "1", "true")
        else:
            run_modal = True

    if not run_modal:
        print("Modal analysis skipped. Returning model only.")
        return {"params": params, "model": model}

    K_full, M_full = assemble_KM_3d_with_nsm(model)
    K_full = add_nodal_springs(K_full, model)

    ndof_full = int(model["dof"]["ndof"])
    mpc_pairs = np.asarray(model.get("mpc_pairs", np.zeros((0, 2), dtype=int)), dtype=int)
    if mpc_pairs.size:
        T, keep_dofs = build_rigid_mpc_T(model)
        full_to_red = np.full(ndof_full, -1, dtype=int)
        full_to_red[keep_dofs] = np.arange(keep_dofs.size, dtype=int)
        K = (T.T @ K_full @ T).tocsc()
        M = (T.T @ M_full @ T).tocsc()
    else:
        T = None
        full_to_red = np.arange(ndof_full, dtype=int)
        K = K_full
        M = M_full

    freq_hz, Phi, info = solve_modes_3d(K, M, model["bc"], params["nmodes"], full_to_red=full_to_red, T=T, M_full=M_full)

    if do_plot:
        import matplotlib.pyplot as plt
        print("Plotting mode shapes...")
        nplot = min(int(params["plot"]["nplot"]), freq_hz.size)
        plt.rcParams["figure.max_open_warning"] = max(50, nplot + 5)

        for k in range(nplot):
            figk = plt.figure(f"Mode {k + 1} (3D)")
            axk = figk.add_subplot(111, projection="3d")
            plot_mode_3d(model, Phi[:, k], f"Mode {k + 1}: {freq_hz[k]:.3f} Hz", params, axk)

        plt.show()

    print(f"Computed {freq_hz.size} modes (Hz):")
    print(np.array2string(freq_hz, precision=6, separator=", "))
    return {"params": params, "model": model, "K": K, "M": M, "K_full": K_full, "M_full": M_full, "freq_hz": freq_hz, "Phi": Phi, "info": info}


def grid_modal_demo_3d_offset_upgraded_v22(do_plot: bool = False, run_modal: Optional[bool] = None) -> dict:
    params = default_params_grid_3d_offset_upgraded_v22()
    return run_grid_modal_with_params_and_plot(params, do_plot=do_plot, run_modal=run_modal)


def gdof0(nid: int, dof_1based: int) -> int:
    return 6 * (nid - 1) + (dof_1based - 1)


def find_nodes_by_coordinate(model: dict, *, x=None, y=None, z=None, tol=1e-6) -> np.ndarray:
    coords = model["nodes"][:, 1:4]
    ids = model["nodes"][:, 0].astype(int)
    mask = np.ones(ids.size, dtype=bool)
    if x is not None:
        mask &= np.abs(coords[:, 0] - float(x)) <= tol
    if y is not None:
        mask &= np.abs(coords[:, 1] - float(y)) <= tol
    if z is not None:
        mask &= np.abs(coords[:, 2] - float(z)) <= tol
    return ids[mask]


def build_influence_vector(model: dict, node_ids: np.ndarray, dof_1based: int) -> np.ndarray:
    dof_1based = int(dof_1based)
    if dof_1based < 1 or dof_1based > 6:
        raise ValueError(f"dof_1based must be in [1, 6], got {dof_1based}.")
    ndof_full = int(model["dof"]["ndof"]) if "dof" in model else 6 * model["nodes"].shape[0]
    r = np.zeros(ndof_full, dtype=np.float64)
    for nid in np.asarray(node_ids, dtype=int):
        idx = gdof0(int(nid), dof_1based)
        if 0 <= idx < ndof_full:
            r[idx] = 1.0
    return r


def build_influence_matrix(model: dict, node_ids: np.ndarray, dof_1based_list: list[int]) -> tuple[csc_matrix, np.ndarray]:
    ndof_full = int(model["dof"]["ndof"]) if "dof" in model else 6 * model["nodes"].shape[0]
    nodes = np.asarray(node_ids, dtype=int).reshape(-1)
    dofs = [int(d) for d in dof_1based_list]
    for d in dofs:
        if d < 1 or d > 6:
            raise ValueError(f"All dof_1based values must be in [1, 6], got {d}.")

    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    input_map: List[List[int]] = []
    col = 0
    for nid in nodes:
        for dof in dofs:
            idx = gdof0(int(nid), int(dof))
            if 0 <= idx < ndof_full:
                rows.append(int(idx))
                cols.append(int(col))
                vals.append(1.0)
                input_map.append([int(nid), int(dof)])
                col += 1
    R = coo_matrix((vals, (rows, cols)), shape=(ndof_full, col)).tocsc()
    return R, np.asarray(input_map, dtype=int)


def modal_participation(Phi: np.ndarray, M_full: csc_matrix, r: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    Mr = M_full @ r
    Gamma = np.asarray(Phi.T @ Mr, dtype=float).reshape(-1)
    denom = float(r @ Mr)
    if denom <= 0:
        raise ValueError("Invalid influence vector or mass matrix: r^T M r must be positive.")
    M_eff = (Gamma**2) / denom
    total_eff = float(np.sum(M_eff))
    if total_eff <= 0:
        raise ValueError("Effective modal mass is non-positive; check Phi, M_full, and influence vector r.")
    cum_ratio = np.cumsum(M_eff) / total_eff
    idx95 = np.where(cum_ratio >= 0.95)[0]
    n95 = int(idx95[0] + 1) if idx95.size else int(Phi.shape[1])
    n95 = min(n95, int(Phi.shape[1]))
    return Gamma, M_eff, cum_ratio, n95


def get_girder_node_ids(model: dict) -> np.ndarray:
    """Return unique node IDs connected to girder elements (grp == 1)."""
    elems = np.asarray(model.get("elems", np.zeros((0, 7))), dtype=float)
    if elems.size == 0:
        return np.zeros(0, dtype=int)
    grp = elems[:, 3].astype(int)
    mask = grp == 1
    if not np.any(mask):
        return np.zeros(0, dtype=int)
    n1 = elems[mask, 1].astype(int)
    n2 = elems[mask, 2].astype(int)
    return np.unique(np.r_[n1, n2]).astype(int)


def select_girder_sensor_nodes(model: dict, x_fracs: list[float], tol_x: float = 1e-6) -> np.ndarray:
    """Select ordered girder-only sensor nodes for U3 reduction.

    Ordering is deterministic: girder index j=0..Ng-1, then x_fracs in provided order.
    Exact x matches are preferred when |x-x_target|<=tol_x; otherwise nearest girder node is used.
    """
    if len(x_fracs) == 0:
        raise ValueError("x_fracs must not be empty.")

    nodes = np.asarray(model["nodes"], dtype=float)
    girder_node_ids = get_girder_node_ids(model)
    if girder_node_ids.size == 0:
        raise ValueError("No girder nodes found (grp==1).")

    Ng = int(model.get("Ng", 0))
    L_mm = float(model.get("L_mm", 0.0))
    dy_mm = float(model.get("dy_mm", 0.0))
    if Ng <= 0 or L_mm <= 0 or dy_mm <= 0:
        y_all = nodes[girder_node_ids - 1, 2]
        y_unique = unique_tol(y_all, 1e-6)
        Ng = int(y_unique.size)
        dy_mm = float(np.median(np.diff(np.sort(y_unique)))) if y_unique.size > 1 else 0.0
        x_all = nodes[girder_node_ids - 1, 1]
        L_mm = float(np.max(x_all))
    if Ng <= 0 or L_mm <= 0 or dy_mm <= 0:
        raise ValueError(f"Invalid geometry for sensor selection: Ng={Ng}, L_mm={L_mm}, dy_mm={dy_mm}.")

    tol_y = 1e-6
    y_min = 0.0
    y_max = float((Ng - 1) * dy_mm)
    idx = girder_node_ids - 1
    xg = nodes[idx, 1]
    yg = nodes[idx, 2]
    band = (yg >= y_min - tol_y) & (yg <= y_max + tol_y)
    if not np.any(band):
        raise ValueError("No girder nodes in the expected y-band for sensor selection.")
    node_ids = girder_node_ids[band]
    xg = xg[band]
    yg = yg[band]

    selected: List[int] = []
    for j in range(Ng):
        y_target = float(j * dy_mm)
        jmask = np.abs(yg - y_target) <= tol_y
        if not np.any(jmask):
            raise ValueError(f"No girder node candidates found at girder j={j}, y_target={y_target:.6f}.")
        cand_ids = node_ids[jmask]
        cand_x = xg[jmask]
        for frac in x_fracs:
            x_target = float(frac) * L_mm
            dx_abs = np.abs(cand_x - x_target)
            i_best = int(np.argmin(dx_abs))
            # Prefer exact match when available, otherwise nearest candidate.
            if dx_abs[i_best] > float(tol_x):
                # Keep nearest node fallback for non-exact station definitions.
                pass
            selected.append(int(cand_ids[i_best]))

    return np.asarray(selected, dtype=int)


def reduce_modes_u3(Phi_full: np.ndarray, sensor_node_ids: np.ndarray) -> np.ndarray:
    """Reduce full mode shapes to U3-only sensor rows: shape (nSensors, nModes)."""
    Phi = np.asarray(Phi_full, dtype=np.float64)
    n_modes = int(Phi.shape[1])
    sensors = np.asarray(sensor_node_ids, dtype=int).reshape(-1)
    out = np.zeros((sensors.size, n_modes), dtype=np.float64)
    for i, nid in enumerate(sensors):
        out[i, :] = Phi[gdof0(int(nid), 3), :]
    return out


def reduce_modes_channel(Phi_full: np.ndarray, sensor_node_ids: np.ndarray, dof_1based: int) -> np.ndarray:
    """Reduce full mode shapes to a single DOF channel at sensor nodes."""
    d = int(dof_1based)
    if d < 1 or d > 6:
        raise ValueError(f"dof_1based must be in [1,6], got {d}.")
    Phi = np.asarray(Phi_full, dtype=np.float64)
    n_modes = int(Phi.shape[1])
    sensors = np.asarray(sensor_node_ids, dtype=int).reshape(-1)
    out = np.zeros((sensors.size, n_modes), dtype=np.float64)
    for i, nid in enumerate(sensors):
        out[i, :] = Phi[gdof0(int(nid), d), :]
    return out


def reduce_modes_u2(Phi_full: np.ndarray, sensor_node_ids: np.ndarray) -> np.ndarray:
    """Reduce full mode shapes to U2-only sensor rows."""
    return reduce_modes_channel(Phi_full, sensor_node_ids, dof_1based=2)


def reduce_modes_rz(Phi_full: np.ndarray, sensor_node_ids: np.ndarray) -> np.ndarray:
    """Reduce full mode shapes to RZ-only sensor rows."""
    return reduce_modes_channel(Phi_full, sensor_node_ids, dof_1based=6)


def _infer_sensor_geometry(model: dict, params: Optional[dict]) -> tuple[float, float, float]:
    """Infer (L_mm, Ng, dy_mm) when params are missing/incomplete."""
    p = params or {}
    girder_node_ids = get_girder_node_ids(model)
    if girder_node_ids.size == 0:
        raise ValueError("Cannot infer sensor geometry: no girder nodes found (grp==1).")
    nodes = np.asarray(model["nodes"], dtype=float)
    idx = girder_node_ids - 1
    xg = nodes[idx, 1]
    yg = nodes[idx, 2]

    x_min = float(np.min(xg))
    x_max = float(np.max(xg))
    L_mm = float(p.get("L_mm", x_max - x_min if x_max > x_min else x_max))
    if L_mm <= 0:
        raise ValueError("Invalid inferred L_mm. Provide params['L_mm'] or ensure model geometry is valid.")

    y_unique = unique_tol(yg, 1e-6)
    Ng = int(p.get("Ng", y_unique.size))
    if Ng < 2:
        raise ValueError("Invalid Ng inferred for sensor selection; need at least 2 girders.")

    if "dy_mm" in p:
        dy_mm = float(p["dy_mm"])
    else:
        if y_unique.size < 2:
            raise ValueError("Cannot infer dy_mm from girder nodes; need at least two unique y-levels.")
        y_d = np.diff(np.sort(y_unique))
        dy_mm = float(np.median(y_d))
    if dy_mm <= 0:
        raise ValueError("Invalid inferred dy_mm. Provide params['dy_mm'] or verify model y-grid.")

    return L_mm, float(Ng), dy_mm


def select_sensor_nodes_outer_girders(model: dict, params: Optional[dict], x_fracs: list[float], ytol: float = 1e-6) -> np.ndarray:
    """Select sensor nodes on outer girders at requested x-fraction stations."""
    if ytol <= 0:
        raise ValueError(f"ytol must be positive, got {ytol}.")
    if len(x_fracs) == 0:
        raise ValueError("x_fracs must not be empty.")

    L_mm, Ng, dy_mm = _infer_sensor_geometry(model, params)
    y_targets = [0.0, (Ng - 1.0) * dy_mm]

    girder_node_ids = get_girder_node_ids(model)
    if girder_node_ids.size == 0:
        raise ValueError("No girder nodes found for sensor selection (grp==1).")

    nodes = np.asarray(model["nodes"], dtype=float)
    idx = girder_node_ids - 1
    xg = nodes[idx, 1]
    yg = nodes[idx, 2]
    x0 = float(np.min(xg))

    selected: List[int] = []
    for y_target in y_targets:
        ymask = np.abs(yg - float(y_target)) <= float(ytol)
        if not np.any(ymask):
            raise ValueError(
                f"No girder candidates found for y_target={y_target:.6g} with ytol={ytol:.3g}. "
                "Increase ytol or verify Ng/dy_mm."
            )
        cand_ids = girder_node_ids[ymask]
        cand_x = xg[ymask]
        for frac in x_fracs:
            x_target = x0 + float(frac) * float(L_mm)
            k = int(np.argmin(np.abs(cand_x - x_target)))
            selected.append(int(cand_ids[k]))

    sensor_node_ids = np.asarray(selected, dtype=int)
    expected = 2 * len(x_fracs)
    if sensor_node_ids.size != expected:
        raise RuntimeError(f"Sensor selection failed: expected {expected} nodes, got {sensor_node_ids.size}.")
    return sensor_node_ids


def build_sensor_dofs(sensor_node_ids: np.ndarray, dof_1based_list: Optional[list[int]] = None) -> np.ndarray:
    """Build global 0-based DOF indices for selected sensor nodes."""
    if dof_1based_list is None:
        dof_1based_list = [3]
    if len(dof_1based_list) == 0:
        raise ValueError("dof_1based_list must not be empty.")
    dofs: List[int] = []
    for nid in np.asarray(sensor_node_ids, dtype=int).reshape(-1):
        for d in dof_1based_list:
            d = int(d)
            if d < 1 or d > 6:
                raise ValueError(f"dof_1based must be in [1, 6], got {d}.")
            dofs.append(gdof0(int(nid), d))
    return np.asarray(dofs, dtype=int)


def extract_reduced_modes(Phi_full: np.ndarray, sensor_dofs: np.ndarray) -> np.ndarray:
    """Extract modal matrix rows at sensor DOFs."""
    return np.asarray(Phi_full, dtype=float)[np.asarray(sensor_dofs, dtype=int), :]


def mac_matrix(PhiA: np.ndarray, PhiB: np.ndarray) -> np.ndarray:
    """Compute MAC matrix between two modal bases with columns as mode vectors."""
    A = np.asarray(PhiA, dtype=float)
    B = np.asarray(PhiB, dtype=float)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("PhiA and PhiB must be 2D arrays.")
    if A.shape[0] != B.shape[0]:
        raise ValueError(f"MAC requires same DOF count. Got {A.shape[0]} and {B.shape[0]}.")

    cross = np.abs(A.T @ B) ** 2
    a_den = np.sum(A * A, axis=0, dtype=float)
    b_den = np.sum(B * B, axis=0, dtype=float)
    denom = np.outer(a_den, b_den)
    out = np.zeros_like(cross, dtype=float)
    nz = denom > 0.0
    out[nz] = cross[nz] / denom[nz]
    return out


def match_modes_by_mac(Phi_red_new: np.ndarray, Phi_ref_red: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Greedy one-to-one MAC matching from reference modes to new-mode indices."""
    Mmac = mac_matrix(Phi_red_new, Phi_ref_red)
    n_new, n_ref = Mmac.shape
    if n_ref < 1:
        raise ValueError("No reference modes provided for matching.")
    if n_new < n_ref:
        raise ValueError(f"Need at least {n_ref} new modes for one-to-one matching, got {n_new}.")

    best_new_indices: List[int] = []
    best_scores: List[float] = []
    remaining = set(range(n_new))
    for j_ref in range(n_ref):
        cands = sorted(remaining)
        scores = Mmac[cands, j_ref]
        i_loc = int(np.argmax(scores))
        i_new = int(cands[i_loc])
        best_new_indices.append(i_new)
        best_scores.append(float(scores[i_loc]))
        remaining.remove(i_new)
    return np.asarray(best_new_indices, dtype=int), np.asarray(best_scores, dtype=float)


def save_reference_modes_npz(
    path: str,
    ref_freqs_hz: np.ndarray,
    Phi_ref_red: np.ndarray,
    sensor_node_ids: np.ndarray,
    sensor_dofs: np.ndarray,
    ref_mode_indices_0based: np.ndarray,
    meta_dict: Optional[dict] = None,
) -> None:
    """Save reference reduced modes and metadata to an NPZ file."""
    meta_json = json.dumps(meta_dict or {}, sort_keys=True)
    np.savez(
        path,
        ref_freqs_hz=np.asarray(ref_freqs_hz, dtype=float),
        Phi_ref_red=np.asarray(Phi_ref_red, dtype=float),
        sensor_node_ids=np.asarray(sensor_node_ids, dtype=int),
        sensor_dofs=np.asarray(sensor_dofs, dtype=int),
        ref_mode_indices_0based=np.asarray(ref_mode_indices_0based, dtype=int),
        meta_json=np.asarray(meta_json),
    )


def load_reference_modes_npz(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load reference reduced modes and metadata from an NPZ file."""
    data = np.load(path, allow_pickle=False)
    ref_freqs_hz = np.asarray(data["ref_freqs_hz"], dtype=float)
    Phi_ref_red = np.asarray(data["Phi_ref_red"], dtype=float)
    sensor_node_ids = np.asarray(data["sensor_node_ids"], dtype=int)
    sensor_dofs = np.asarray(data["sensor_dofs"], dtype=int)
    ref_mode_indices = np.asarray(data["ref_mode_indices_0based"], dtype=int)
    meta = {}
    if "meta_json" in data:
        raw = data["meta_json"]
        if np.ndim(raw) == 0:
            meta_str = str(raw.item())
        else:
            meta_str = str(raw)
        if meta_str.strip():
            meta = json.loads(meta_str)
    return ref_freqs_hz, Phi_ref_red, sensor_node_ids, sensor_dofs, ref_mode_indices, meta


# ---------------------------------------------------------------------------
# Interactive geometry adjustment helpers (used only by --adjustGeo CLI)
# ---------------------------------------------------------------------------

def classify_model_nodes(model: dict) -> dict:
    """Return sets of node IDs by role: girder, mpc_slave, free."""
    mpc_pairs = np.asarray(model.get("mpc_pairs", np.zeros((0, 2), dtype=int)), dtype=int)
    slave_ids = set(int(s) for s in mpc_pairs[:, 1]) if mpc_pairs.size else set()
    girder_ids = set(int(n) for n in get_girder_node_ids(model))
    all_ids = set(int(n) for n in model["nodes"][:, 0].astype(int))
    free_ids = all_ids - slave_ids - girder_ids
    return {"girder": girder_ids, "mpc_slave": slave_ids, "free": free_ids}


def propagate_master_move(model: dict, master_nid: int, new_xyz: np.ndarray) -> None:
    """Move a master node and slide all its MPC slaves to preserve their z-offset."""
    nodes = model["nodes"]
    idx = master_nid - 1
    old_z = float(nodes[idx, 3])
    nodes[idx, 1:4] = new_xyz
    mpc_pairs = np.asarray(model.get("mpc_pairs", np.zeros((0, 2), dtype=int)), dtype=int)
    for m, s in mpc_pairs:
        if int(m) == master_nid:
            s_idx = int(s) - 1
            z_offset = float(nodes[s_idx, 3]) - old_z
            nodes[s_idx, 1] = new_xyz[0]
            nodes[s_idx, 2] = new_xyz[1]
            nodes[s_idx, 3] = new_xyz[2] + z_offset


def plot_model_3d_labeled(model: dict, ax, params: Optional[dict], node_classes: dict) -> None:
    """Render the undeformed model with node ID labels and group colour-coding."""
    nodes = model["nodes"]
    elems = model["elems"]
    grp_colors = {1: "0.55", 2: "#2196F3", 3: "#4CAF50", 4: "#9E9E9E",
                  5: "#FF9800", 6: "#9C27B0", 7: "#F44336", 8: "#E91E63", 9: "#795548"}
    grp_names = {1: "girder", 2: "cross", 3: "diag", 4: "rigid",
                 5: "bottom", 6: "front", 7: "balcony", 8: "edge", 9: "bbrace"}
    seen: set = set()
    for e in elems:
        n1, n2, grp = int(e[1]) - 1, int(e[2]) - 1, int(e[3])
        clr = grp_colors.get(grp, "0.4")
        lbl = grp_names.get(grp, f"grp{grp}") if grp not in seen else None
        seen.add(grp)
        ax.plot(
            [nodes[n1, 1], nodes[n2, 1]], [nodes[n1, 2], nodes[n2, 2]], [nodes[n1, 3], nodes[n2, 3]],
            color=clr, lw=1.2, label=lbl,
        )
    labeled = node_classes["girder"] | node_classes["free"]
    slaves = node_classes["mpc_slave"]
    for row in nodes:
        nid = int(row[0])
        x, y, z = float(row[1]), float(row[2]), float(row[3])
        if nid in labeled:
            ax.text(x, y, z, f" {nid}", fontsize=5, color="black", ha="left", va="bottom")
        elif nid in slaves:
            ax.text(x, y, z, f" {nid}", fontsize=4, color="0.60", ha="left", va="bottom")
    ax.plot(nodes[:, 1], nodes[:, 2], nodes[:, 3], "k.", markersize=4)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.view_init(elev=20, azim=35)
    handles, labels_leg = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels_leg, loc="upper left", fontsize=6, ncol=2)
    z_scale = float(params.get("plot", {}).get("z_visual_scale", 1.0)) if params is not None else 1.0
    set_axes_equal_3d(ax, nodes[:, 1], nodes[:, 2], nodes[:, 3], z_visual_scale=z_scale)
    ax.set_title("Geometry Check — [m]odal  [a]djust  [q]uit")


def print_node_table(model: dict, node_classes: dict) -> None:
    """Print a formatted coordinate table for all nodes."""
    nodes = model["nodes"]
    girder_ids = node_classes["girder"]
    slave_ids = node_classes["mpc_slave"]
    print(f"\n{'ID':>5}  {'type':<10}  {'x (mm)':>14}  {'y (mm)':>14}  {'z (mm)':>14}")
    print("-" * 65)
    for row in nodes:
        nid = int(row[0])
        x, y, z = float(row[1]), float(row[2]), float(row[3])
        if nid in girder_ids:
            ntype = "girder"
        elif nid in slave_ids:
            ntype = "mpc_slave"
        else:
            ntype = "free"
        print(f"{nid:>5}  {ntype:<10}  {x:>14.3f}  {y:>14.3f}  {z:>14.3f}")
    print()


def interactive_adjust_geometry(model: dict, params: Optional[dict]) -> dict:
    """Interactive node coordinate editing loop.  Returns the (possibly modified) model."""
    import matplotlib.pyplot as plt

    nodes = model["nodes"]
    mpc_pairs = np.asarray(model.get("mpc_pairs", np.zeros((0, 2), dtype=int)), dtype=int)
    elems_arr = np.asarray(model.get("elems", np.zeros((0, 7))), dtype=float)

    master_to_slaves: Dict[int, List[int]] = {}
    slave_to_master: Dict[int, int] = {}
    for m, s in mpc_pairs:
        master_to_slaves.setdefault(int(m), []).append(int(s))
        slave_to_master[int(s)] = int(m)

    # Cross beam topology: node_id -> list of elem IDs (grp==2), elem_id -> (n1, n2)
    cb_node_to_elems: Dict[int, List[int]] = {}
    cb_elem_endpoints: Dict[int, Tuple[int, int]] = {}
    if elems_arr.size:
        for e in elems_arr:
            eid, n1, n2, grp = int(e[0]), int(e[1]), int(e[2]), int(e[3])
            if grp == 2:
                cb_node_to_elems.setdefault(n1, []).append(eid)
                cb_node_to_elems.setdefault(n2, []).append(eid)
                cb_elem_endpoints[eid] = (n1, n2)

    node_classes = classify_model_nodes(model)
    girder_y_lock: Dict[int, float] = {nid: float(nodes[nid - 1, 2]) for nid in node_classes["girder"]}
    original_coords: Dict[int, np.ndarray] = {int(row[0]): row[1:4].copy() for row in nodes}

    print("\nEditing commands:")
    print("  <id> <x> <y> <z>   move node to new coordinates")
    print("  list               re-print node table")
    print("  reset <id>         restore original coordinates for a node")
    print("  done               finish editing\n")

    fig_lbl = "Geometry Adjustment"
    while True:
        plt.figure(fig_lbl, figsize=(13, 7))
        plt.clf()
        ax = plt.gcf().add_subplot(111, projection="3d")
        plot_model_3d_labeled(model, ax, params, node_classes)
        plt.pause(0.05)
        plt.show(block=False)

        cmd = input("adjust> ").strip()
        if not cmd:
            continue

        parts = cmd.split()
        verb = parts[0].lower()

        if verb == "done":
            plt.close(fig_lbl)
            break

        if verb == "list":
            print_node_table(model, node_classes)
            continue

        if verb == "reset":
            if len(parts) != 2:
                print("Usage: reset <node_id>")
                continue
            try:
                nid = int(parts[1])
            except ValueError:
                print(f"Invalid node ID '{parts[1]}'.")
                continue
            if nid < 1 or nid > nodes.shape[0]:
                print(f"Node {nid} out of range.")
                continue
            orig = original_coords[nid]
            if nid in node_classes["mpc_slave"]:
                nodes[nid - 1, 1:4] = orig
                print(f"Node {nid} (mpc_slave) reset to ({orig[0]:.3f}, {orig[1]:.3f}, {orig[2]:.3f}).")
            elif nid in node_classes["girder"]:
                propagate_master_move(model, nid, orig)
                print(f"Node {nid} (girder) reset to ({orig[0]:.3f}, {orig[1]:.3f}, {orig[2]:.3f}); slaves updated.")
            else:
                nodes[nid - 1, 1:4] = orig
                print(f"Node {nid} (free) reset to ({orig[0]:.3f}, {orig[1]:.3f}, {orig[2]:.3f}).")
            continue

        # Default: <id> <x> <y> <z>
        if len(parts) != 4:
            print("Usage: <node_id> <x> <y> <z>  |  list  |  reset <id>  |  done")
            continue
        try:
            nid = int(parts[0])
            nx, ny, nz = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            print("Parse error — expected: <int> <float> <float> <float>.")
            continue

        if nid < 1 or nid > nodes.shape[0]:
            print(f"Node {nid} does not exist (valid range 1–{nodes.shape[0]}).")
            continue

        if nid in node_classes["mpc_slave"]:
            master = slave_to_master.get(nid, "?")
            print(f"Node {nid} is an MPC slave — edit its master (node {master}) instead.")
            continue

        if nid in node_classes["girder"]:
            y_lock = girder_y_lock[nid]
            if abs(ny - y_lock) > 1e-3 or abs(nz) > 1e-3:
                print(f"  Note: girder node y locked to {y_lock:.3f} mm, z locked to 0.0 mm — snapping.")
                ny, nz = y_lock, 0.0
            old_x = float(nodes[nid - 1, 1])
            propagate_master_move(model, nid, np.array([nx, ny, nz]))
            print(f"Moved girder node {nid}: x {old_x:.3f} → {nx:.3f} mm")

            # Warn about cross beam skew
            warned = False
            for sl in master_to_slaves.get(nid, []):
                for eid in cb_node_to_elems.get(sl, []):
                    n1_e, n2_e = cb_elem_endpoints[eid]
                    peer_sl = n2_e if n1_e == sl else n1_e
                    peer_master = slave_to_master.get(peer_sl)
                    if peer_master is not None and peer_master != nid:
                        peer_x = float(nodes[peer_master - 1, 1])
                        if abs(peer_x - nx) > 1e-3:
                            if not warned:
                                print("  ** Cross beam skew warning **")
                                warned = True
                            print(f"     Elem {eid}: peer girder node {peer_master} still at x={peer_x:.3f} mm.")
                            print(f"     To keep perpendicular: move node {peer_master} to x={nx:.3f} mm.")
        else:
            old = nodes[nid - 1, 1:4].copy()
            nodes[nid - 1, 1:4] = [nx, ny, nz]
            print(f"Moved free node {nid}: ({old[0]:.3f}, {old[1]:.3f}, {old[2]:.3f}) → ({nx:.3f}, {ny:.3f}, {nz:.3f})")

    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    # ------------------------------------------------------------------
    # Press Run (VS Code) to solve THIS structure with its real geometry.
    #   RUN_STRUCTURE = "Sykkylven"  -> pull params from
    #                                   build_phase3_graphs.STRUCTURES
    #   RUN_STRUCTURE = None         -> built-in defaults (Magerholm geometry)
    # Override from the CLI with  --structure NAME  or  --structure none.
    #
    # This lives entirely inside __main__. Importers of this module are
    # unaffected and default_params_grid_3d_offset_upgraded_v22() is unchanged,
    # so no pipeline script sees any difference.
    # ------------------------------------------------------------------
    RUN_STRUCTURE: Optional[str] = "Sykkylven"

    parser = argparse.ArgumentParser(description="Grid modal FE solver / geometry check")
    parser.add_argument("--adjustGeo", action="store_true",
                        help="Interactive geometry adjustment mode before modal analysis")
    parser.add_argument("--structure", default=RUN_STRUCTURE,
                        help="Named structure from build_phase3_graphs.STRUCTURES, "
                             "or 'none' for the built-in default geometry")
    args = parser.parse_args()

    def _params_for_structure(name: Optional[str]) -> dict:
        params = default_params_grid_3d_offset_upgraded_v22()
        if not name or str(name).lower() == "none":
            print("[structure] built-in default geometry (Magerholm)")
            return params
        from copy import deepcopy
        from build_phase3_graphs import STRUCTURES, _deep_update
        if name not in STRUCTURES:
            raise SystemExit(f"Unknown structure {name!r}. "
                             f"Available: {sorted(STRUCTURES)}")
        _deep_update(params, deepcopy(STRUCTURES[name]))
        print(f"[structure] {name}: L_mm={params['L_mm']} Nx={params['Nx']} "
              f"Ng={params['Ng']} dy_mm={params['dy_mm']} "
              f"diag_pattern={params['diag_pattern']}")
        bal = params.get("balcony", {})
        if bal.get("enable"):
            left = bal.get("length_left_mm", bal.get("length_mm"))
            right = bal.get("length_right_mm", bal.get("length_mm"))
            print(f"[structure] balcony: left={left} mm  right={right} mm  "
                  f"x_center_frac={bal.get('x_center_frac')}  "
                  f"nBeams={bal.get('nBeams')}  dx_mm={bal.get('dx_mm')}")
        return params

    params = _params_for_structure(args.structure)

    if not args.adjustGeo:
        run_grid_modal_with_params_and_plot(params, do_plot=True, run_modal=None)
    else:
        import matplotlib.pyplot as plt

        model = build_grid_model_3d_upgraded_v22(params)

        while True:
            node_classes = classify_model_nodes(model)
            fig = plt.figure("Geometry Check", figsize=(13, 7))
            fig.clf()
            ax = fig.add_subplot(111, projection="3d")
            plot_model_3d_labeled(model, ax, params, node_classes)
            plt.pause(0.05)
            plt.show(block=False)

            print("\n" + "=" * 60)
            ans = input("Options: [m]odal analysis  [a]djust geometry  [q]uit: ").strip().lower()
            plt.close("all")

            if ans in ("m", "modal"):
                break
            elif ans in ("a", "adjust"):
                print_node_table(model, node_classes)
                model = interactive_adjust_geometry(model, params)
            elif ans in ("q", "quit", ""):
                print("Exiting without modal analysis.")
                raise SystemExit(0)
            else:
                print(f"Unknown option '{ans}'. Enter m, a, or q.")

        # Run modal analysis on the (possibly adjusted) model
        K_full, M_full = assemble_KM_3d_with_nsm(model)
        K_full = add_nodal_springs(K_full, model)

        ndof_full = int(model["dof"]["ndof"])
        mpc_pairs_arr = np.asarray(model.get("mpc_pairs", np.zeros((0, 2), dtype=int)), dtype=int)
        if mpc_pairs_arr.size:
            T, keep_dofs = build_rigid_mpc_T(model)
            full_to_red = np.full(ndof_full, -1, dtype=int)
            full_to_red[keep_dofs] = np.arange(keep_dofs.size, dtype=int)
            K = (T.T @ K_full @ T).tocsc()
            M = (T.T @ M_full @ T).tocsc()
        else:
            T = None
            full_to_red = np.arange(ndof_full, dtype=int)
            K = K_full
            M = M_full

        freq_hz, Phi, info = solve_modes_3d(
            K, M, model["bc"], params["nmodes"],
            full_to_red=full_to_red, T=T, M_full=M_full,
        )
        print(f"\nComputed {freq_hz.size} modes (Hz):")
        print(np.array2string(freq_hz, precision=6, separator=", "))

        nplot = min(int(params["plot"]["nplot"]), freq_hz.size)
        plt.rcParams["figure.max_open_warning"] = max(50, nplot + 5)
        for k in range(nplot):
            figk = plt.figure(f"Mode {k + 1} (3D)")
            axk = figk.add_subplot(111, projection="3d")
            plot_mode_3d(model, Phi[:, k], f"Mode {k + 1}: {freq_hz[k]:.3f} Hz", params, axk)
        plt.show()
