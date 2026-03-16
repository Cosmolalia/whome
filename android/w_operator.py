
# (Content identical to previous attempt; see docstring at top.)

"""
w_operator.py — Unified operator pipeline for Akatalêptos W-approximants

Implements:
- build_graph(k, G1, G2, S, N) -> (vertices, edges, boundary_ids)
- add_glue_edges(vertices, boundary_ids, lam, w_glue, G1, G2) -> (edges_update, psi_by_id)
- build_magnetic_laplacian(vertices, edges, s, psi_by_id) -> sparse Hermitian Laplacian matrix (CSR), id_to_idx
- solve_spectrum(L, M, tol) -> sorted eigenvalues (numpy array)
- select_I_gap(eigs_ref, K_gap=32) -> list[int]
- extract_ratios(eigs, I_gap) -> ratio dict with labels
- serialize_run(meta, eigs, ratios, path) -> write files

Design note:
A literal cartesian product X_{k,n,N} explodes. This module uses a deterministic 1D golden-winding orbit
through the torus grid to approximate T^2 while keeping vertex counts feasible for eigensolvers.
Penrose hull is currently stubbed as a single patch type (penrose_patch=0). Hook points are marked TODO.

Protocol version: W-spectrum-v1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List, Set, Optional, Iterable, Any
import json
import hashlib
import math

import numpy as np

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
except Exception:  # pragma: no cover
    sp = None
    spla = None

PROTOCOL_VERSION = "W-spectrum-v1"
MAX_BOUNDARY_CUBES = 250
TORUS_ORBIT_LEN_MODE = "G1"

def _stable_json(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True)

def stable_id(obj: Any, n: int = 16) -> str:
    s = _stable_json(obj)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:n]

@dataclass(frozen=True)
class VertexKey:
    menger_address: Tuple[int, ...]
    torus_idx: int
    torus_coords: Tuple[int, int]
    penrose_patch: int
    circle_coord: int

def vertex_id(key: VertexKey) -> str:
    return stable_id({
        "m": list(key.menger_address),
        "ti": key.torus_idx,
        "t": [key.torus_coords[0], key.torus_coords[1]],
        "p": key.penrose_patch,
        "s": key.circle_coord
    })

# --- Menger sponge kept offsets (20 of 27) ---
KEPT_OFFSETS: List[Tuple[int,int,int]] = [
    (a,b,c)
    for a in (0,1,2) for b in (0,1,2) for c in (0,1,2)
    if not ((a==1)+(b==1)+(c==1) >= 2)
]
J_TO_OFF = {j: off for j, off in enumerate(KEPT_OFFSETS)}

def menger_address_to_coord(address: Tuple[int, ...]) -> Tuple[int,int,int]:
    x = y = z = 0
    for j in address:
        dx, dy, dz = J_TO_OFF[j]
        x = 3*x + dx
        y = 3*y + dy
        z = 3*z + dz
    return (x,y,z)

def address_grid_digits(address: Tuple[int, ...]) -> List[Tuple[int,int,int]]:
    return [J_TO_OFF[j] for j in address]

def generate_menger_kept_addresses(k: int) -> List[Tuple[int, ...]]:
    addresses: List[Tuple[int,...]] = [tuple()]
    for _ in range(k):
        new: List[Tuple[int,...]] = []
        for a in addresses:
            for j in range(len(KEPT_OFFSETS)):
                new.append(a + (j,))
        addresses = new
    return addresses

def boundary_addresses(k: int, max_cubes: int = MAX_BOUNDARY_CUBES) -> List[Tuple[int,...]]:
    addrs = generate_menger_kept_addresses(k)
    coords = {menger_address_to_coord(a): a for a in addrs}
    kept = set(coords.keys())
    neigh = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    boundary: List[Tuple[int,...]] = []
    for c, a in coords.items():
        x,y,z = c
        is_b = False
        for dx,dy,dz in neigh:
            if (x+dx, y+dy, z+dz) not in kept:
                is_b = True
                break
        if is_b:
            boundary.append(a)
    boundary.sort(key=lambda a: menger_address_to_coord(a))
    if max_cubes is not None and len(boundary) > max_cubes:
        boundary = boundary[:max_cubes]
    return boundary

# --- Psi(lambda) ---
phi = (1 + 5**0.5) / 2
TAU = 2*math.pi

def cantor_coord(digits: Iterable[int]) -> float:
    s = 0.0
    p = 1.0
    for d in digits:
        p *= 3.0
        s += d / p
    return s % 1.0

def psi_lambda_from_digits(grid_digits_xyz: List[Tuple[int,int,int]], lam: float) -> Tuple[float,float]:
    xs = [x for (x,_,_) in grid_digits_xyz]
    ys = [y for (_,y,_) in grid_digits_xyz]
    zs = [z for (_,_,z) in grid_digits_xyz]
    u = cantor_coord(xs)
    v = cantor_coord(ys)
    w = cantor_coord(zs)
    psi1 = u
    psi2 = (u + phi * (lam*v + (1-lam)*w)) % 1.0
    return (psi1, psi2)

# --- Torus 1D orbit ---
def torus_orbit(G1: int, G2: int) -> List[Tuple[int,int]]:
    step2 = int(math.floor((phi - 1.0) * G2))
    if step2 % G2 == 0:
        step2 = 1
    if TORUS_ORBIT_LEN_MODE == "G1":
        L = G1
    else:
        try:
            L = int(TORUS_ORBIT_LEN_MODE)
        except Exception:
            L = G1
    pts: List[Tuple[int,int]] = []
    t1 = 0
    t2 = 0
    for _ in range(L):
        pts.append((t1 % G1, t2 % G2))
        t1 += 1
        t2 += step2
    return pts

def torus_shift_idx(t_idx: int, psi_cont: Tuple[float,float], L: int) -> int:
    delta = int(math.floor(((psi_cont[0] + psi_cont[1]) % 1.0) * L + 0.5)) % L
    return (t_idx + delta) % L

# --- Graph builder ---
def build_graph(k: int, G1: int, G2: int, S: int, N: int) -> Tuple[Dict[str, Dict[str, Any]], Dict[Tuple[str,str], Dict[str, Any]], Set[str]]:
    b_addrs = boundary_addresses(k, max_cubes=MAX_BOUNDARY_CUBES)
    orbit = torus_orbit(G1, G2)
    L = len(orbit)
    penrose_patches = [0]  # TODO: real patch enumeration at radius N

    vertices: Dict[str, Dict[str, Any]] = {}
    boundary_ids: Set[str] = set()
    digits_by_addr = {a: address_grid_digits(a) for a in b_addrs}
    coord_by_addr = {a: menger_address_to_coord(a) for a in b_addrs}

    for a in b_addrs:
        grid_digits = digits_by_addr[a]
        c = coord_by_addr[a]
        for p in penrose_patches:
            for t_idx, (t1,t2) in enumerate(orbit):
                for tau in range(S):
                    key = VertexKey(a, t_idx, (t1,t2), p, tau)
                    vid = vertex_id(key)
                    vertices[vid] = {
                        "id": vid,
                        "menger_address": list(a),
                        "torus_idx": t_idx,
                        "torus_coords": (t1,t2),
                        "penrose_patch": p,
                        "circle_coord": tau,
                        "is_boundary": True,
                        "grid_digits": grid_digits,
                        "menger_coord": c,
                        "k": k,
                        "N": N
                    }
                    boundary_ids.add(vid)

    edges: Dict[Tuple[str,str], Dict[str, Any]] = {}

    def _add_edge(u: str, v: str, w: float, is_glue: bool = False, owner: Optional[str] = None):
        if u == v:
            return
        a,b = (u,v) if u < v else (v,u)
        rec = edges.get((a,b))
        if rec is None:
            edges[(a,b)] = {"w": float(w), "is_glue": bool(is_glue)}
            if owner is not None:
                edges[(a,b)]["owner"] = owner
        else:
            rec["w"] += float(w)
            rec["is_glue"] = rec["is_glue"] or bool(is_glue)
            if owner is not None and "owner" not in rec:
                rec["owner"] = owner

    vid_by_tuple: Dict[Tuple[Tuple[int,...], int, int, int], str] = {}
    for vid, meta in vertices.items():
        a = tuple(meta["menger_address"])
        p = meta["penrose_patch"]
        t_idx = meta["torus_idx"]
        tau = meta["circle_coord"]
        vid_by_tuple[(a,p,t_idx,tau)] = vid

    addr_by_coord = {coord_by_addr[a]: a for a in b_addrs}
    kept_coords = set(addr_by_coord.keys())
    neigh = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]

    # Menger adjacency across fibers
    for c, a in addr_by_coord.items():
        x,y,z = c
        for dx,dy,dz in neigh:
            c2 = (x+dx, y+dy, z+dz)
            if c2 in kept_coords:
                a2 = addr_by_coord[c2]
                for p in penrose_patches:
                    for t_idx in range(L):
                        for tau in range(S):
                            u = vid_by_tuple[(a,p,t_idx,tau)]
                            v = vid_by_tuple[(a2,p,t_idx,tau)]
                            _add_edge(u, v, w=1.0)

    # Torus orbit adjacency
    for a in b_addrs:
        for p in penrose_patches:
            for tau in range(S):
                for t_idx in range(L):
                    t_next = (t_idx + 1) % L
                    u = vid_by_tuple[(a,p,t_idx,tau)]
                    v = vid_by_tuple[(a,p,t_next,tau)]
                    _add_edge(u, v, w=1.0)

    # Circle adjacency
    for a in b_addrs:
        for p in penrose_patches:
            for t_idx in range(L):
                for tau in range(S):
                    tau_next = (tau + 1) % S
                    u = vid_by_tuple[(a,p,t_idx,tau)]
                    v = vid_by_tuple[(a,p,t_idx,tau_next)]
                    _add_edge(u, v, w=1.0)

    return vertices, edges, boundary_ids

# --- Glue edges ---
def add_glue_edges(vertices: Dict[str, Dict[str, Any]],
                   boundary_ids: Set[str],
                   lam: float,
                   w_glue: float,
                   G1: int,
                   G2: int) -> Tuple[Dict[Tuple[str,str], Dict[str, Any]], Dict[str, Tuple[float,float]]]:
    orbit = torus_orbit(G1, G2)
    L = len(orbit)

    psi_by_id: Dict[str, Tuple[float,float]] = {}
    for vid in boundary_ids:
        psi_by_id[vid] = psi_lambda_from_digits(vertices[vid]["grid_digits"], lam)

    edges_update: Dict[Tuple[str,str], Dict[str, Any]] = {}

    def _add_edge_local(u: str, v: str, w: float, owner: str):
        if u == v:
            return
        a,b = (u,v) if u < v else (v,u)
        rec = edges_update.get((a,b))
        if rec is None:
            edges_update[(a,b)] = {"w": float(w), "is_glue": True, "owner": owner}
        else:
            rec["w"] += float(w)
            rec["is_glue"] = True
            rec["owner"] = min(rec.get("owner", owner), owner)

    for b_id in sorted(boundary_ids):
        meta = vertices[b_id]
        psi = psi_by_id[b_id]
        t_idx = int(meta["torus_idx"])
        t_idx_plus = torus_shift_idx(t_idx, psi, L)
        a = tuple(meta["menger_address"])
        p = int(meta["penrose_patch"])
        tau = int(meta["circle_coord"])
        t1,t2 = orbit[t_idx_plus]
        key_plus = VertexKey(a, t_idx_plus, (t1,t2), p, tau)
        b_plus_id = vertex_id(key_plus)

        if b_plus_id not in vertices:
            vertices[b_plus_id] = {
                "id": b_plus_id,
                "menger_address": list(a),
                "torus_idx": t_idx_plus,
                "torus_coords": (t1,t2),
                "penrose_patch": p,
                "circle_coord": tau,
                "is_boundary": True,
                "grid_digits": meta["grid_digits"],
                "menger_coord": meta["menger_coord"],
                "k": meta.get("k"),
                "N": meta.get("N"),
                "_created_by": "glue_shift"
            }
            psi_by_id[b_plus_id] = psi

        _add_edge_local(b_id, b_plus_id, w=w_glue, owner=b_id)

    return edges_update, psi_by_id

def merge_edges(base: Dict[Tuple[str,str], Dict[str, Any]],
                update: Dict[Tuple[str,str], Dict[str, Any]]) -> Dict[Tuple[str,str], Dict[str, Any]]:
    out = dict(base)
    for k, rec in update.items():
        if k not in out:
            out[k] = dict(rec)
        else:
            out[k]["w"] = float(out[k].get("w", 0.0)) + float(rec.get("w", 0.0))
            out[k]["is_glue"] = bool(out[k].get("is_glue", False) or rec.get("is_glue", False))
            if rec.get("is_glue", False):
                owner = rec.get("owner")
                if owner is not None:
                    out[k]["owner"] = min(out[k].get("owner", owner), owner)
    return out

# --- Magnetic Laplacian ---
def build_magnetic_laplacian(vertices: Dict[str, Dict[str, Any]],
                             edges: Dict[Tuple[str,str], Dict[str, Any]],
                             s: Tuple[int,int],
                             psi_by_id: Dict[str, Tuple[float,float]]) -> Tuple["sp.csr_matrix", Dict[str,int]]:
    if sp is None or spla is None:
        raise RuntimeError("scipy is required (scipy.sparse, scipy.sparse.linalg).")

    v_ids = sorted(vertices.keys())
    id_to_idx = {vid: i for i, vid in enumerate(v_ids)}
    n = len(v_ids)

    deg = np.zeros(n, dtype=np.float64)
    rows: List[int] = []
    cols: List[int] = []
    data: List[np.complex128] = []

    def _alpha(owner: str) -> float:
        psi = psi_by_id.get(owner)
        if psi is None:
            return 0.0
        return TAU * (s[0]*psi[0] + s[1]*psi[1])

    for (u,v), rec in edges.items():
        w = float(rec["w"])
        iu = id_to_idx[u]
        iv = id_to_idx[v]
        deg[iu] += w
        deg[iv] += w

        if rec.get("is_glue", False):
            owner = rec.get("owner")
            a = _alpha(owner) if owner is not None else 0.0
            # phase defined in direction owner -> other.
            if owner == u:
                a_u_to_v = a
            elif owner == v:
                a_u_to_v = -a
            else:
                a_u_to_v = a
        else:
            a_u_to_v = 0.0

        val_uv = w * complex(math.cos(a_u_to_v), math.sin(a_u_to_v))
        val_vu = np.conjugate(val_uv)

        rows.append(iu); cols.append(iv); data.append(val_uv)
        rows.append(iv); cols.append(iu); data.append(val_vu)

    A = sp.csr_matrix((np.array(data, dtype=np.complex128), (np.array(rows), np.array(cols))), shape=(n,n))
    D = sp.diags(deg, format="csr")
    L = D - A
    return L, id_to_idx

# --- Spectrum ---
def solve_spectrum(L: "sp.csr_matrix", M: int, tol: float = 1e-10) -> np.ndarray:
    if spla is None:
        raise RuntimeError("scipy.sparse.linalg is required.")
    n = L.shape[0]
    k = min(M+1, n-2) if n > 3 else min(M+1, n)
    if k < 2:
        vals = np.linalg.eigvalsh(L.toarray())
        return np.sort(np.real(vals))
    # Deterministic initial vector — ARPACK's random v0 causes cross-platform divergence
    v0 = np.ones(n, dtype=np.float64) / np.sqrt(n)
    try:
        vals = spla.eigsh(L, k=k, which="SM", tol=tol, v0=v0, return_eigenvectors=False, maxiter=200000)
        return np.sort(np.real(vals))
    except Exception:
        vals = spla.eigsh(L, k=k, sigma=0.0, which="LM", tol=tol, v0=v0, return_eigenvectors=False, maxiter=200000)
        return np.sort(np.real(vals))

# --- Eigenvalue hashing (canonical — all clients must match) ---
def hash_eigenvalues(eigs) -> str:
    """Deterministic hash of eigenvalue array. Round to 10 decimals, sort, SHA-256."""
    rounded = [round(float(e), 10) for e in sorted(eigs)]
    payload = json.dumps(rounded, separators=(',', ':'))
    return hashlib.sha256(payload.encode()).hexdigest()

# --- Ratio protocol ---
def select_I_gap(eigs_ref: np.ndarray, K_gap: int = 32, zero_tol: float = 1e-12) -> List[int]:
    eigs = np.sort(np.real(np.array(eigs_ref, dtype=np.float64)))
    eigs_nz = eigs[eigs > zero_tol]
    if len(eigs_nz) < 4:
        return []
    M = len(eigs_nz)
    gaps: List[Tuple[float,int]] = []
    for i in range(M-1):
        lam_i = eigs_nz[i]
        lam_ip1 = eigs_nz[i+1]
        g = (lam_ip1 - lam_i) / lam_i
        gaps.append((g, i+1))  # 1-based
    gaps.sort(key=lambda t: (-t[0], t[1]))
    I = [i for _, i in gaps[:min(K_gap, len(gaps))]]
    I.sort()
    return I

def extract_ratios(eigs: np.ndarray, I_gap: List[int], R0: int = 32, zero_tol: float = 1e-12) -> Dict[str, float]:
    eigs = np.sort(np.real(np.array(eigs, dtype=np.float64)))
    eigs_nz = eigs[eigs > zero_tol]
    M = len(eigs_nz)
    out: Dict[str, float] = {}

    max_i = min(R0, M-1)
    for i in range(1, max_i+1):
        out[f"A/r_{i}"] = float(eigs_nz[i] / eigs_nz[i-1])

    for i in I_gap:
        if i < 1 or i >= M:
            continue
        lam_i = eigs_nz[i-1]
        lam_ip1 = eigs_nz[i]
        out[f"B/r_{i}"] = float(lam_ip1 / lam_i)
        out[f"B/g_{i}"] = float((lam_ip1 - lam_i) / lam_i)
        if i+1 < M:
            lam_ip2 = eigs_nz[i+1]
            out[f"B/c_{i}"] = float((lam_ip2 * lam_i) / (lam_ip1 * lam_ip1))
    return out

# --- Serialization ---
def serialize_run(meta: Dict[str, Any], eigs: np.ndarray, ratios: Dict[str, float], path: str) -> None:
    import os
    os.makedirs(path, exist_ok=True)
    meta_out = dict(meta)
    meta_out["protocol_version"] = PROTOCOL_VERSION

    with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2, sort_keys=True)

    with open(os.path.join(path, "eigs.txt"), "w", encoding="utf-8") as f:
        for v in np.sort(np.real(eigs)):
            f.write(f"{v:.17g}\n")

    with open(os.path.join(path, "ratios.jsonl"), "w", encoding="utf-8") as f:
        for label in sorted(ratios.keys()):
            f.write(json.dumps({"label": label, "value": float(ratios[label])}, sort_keys=True) + "\n")


if __name__ == "__main__":
    # Smoke test with progress prints (kept small on purpose).
    # For your full protocol, use run_sweep.py (also provided).
    import traceback
    try:
        print("[w_operator] smoke test start")
        k=2; G1=G2=32; S=16; N=2
        print(f"[w_operator] build_graph(k={k}, G1={G1}, G2={G2}, S={S}, N={N}) ...")
        vertices, edges, boundary_ids = build_graph(k, G1, G2, S, N)
        print(f"[w_operator] vertices={len(vertices):,} edges={len(edges):,} boundary_vertices={len(boundary_ids):,}")

        non_glue_w = [rec["w"] for rec in edges.values() if not rec.get("is_glue", False)]
        med_w = float(np.median(non_glue_w)) if non_glue_w else 1.0
        w_glue = 1e6 * med_w
        lam = 0.5
        print(f"[w_operator] add_glue_edges(lambda={lam}, w_glue={w_glue:g}) ...")
        upd, psi_by = add_glue_edges(vertices, boundary_ids, lam, w_glue, G1, G2)
        edges2 = merge_edges(edges, upd)
        num_glue = sum(1 for r in edges2.values() if r.get("is_glue", False))
        print(f"[w_operator] merged edges={len(edges2):,} (glue_edges={num_glue:,})")

        probes = [(1,0),(0,1),(1,1)]
        for s in probes:
            print(f"[w_operator] build_magnetic_laplacian probe s={s} ...")
            L, _ = build_magnetic_laplacian(vertices, edges2, s=s, psi_by_id=psi_by)
            print(f"[w_operator] solve_spectrum(M=32) ...")
            eigs = solve_spectrum(L, M=32, tol=1e-8)
            print(f"[w_operator] eigs[0:6] = {[float(x) for x in eigs[:6]]}")
            I_gap = select_I_gap(eigs, K_gap=8)
            ratios = extract_ratios(eigs, I_gap, R0=8)
            print(f"[w_operator] ratios_count={len(ratios)} I_gap={I_gap[:8]}")

        print("[w_operator] smoke test done")
    except Exception:
        print("[w_operator] smoke test FAILED with exception:")
        traceback.print_exc()
