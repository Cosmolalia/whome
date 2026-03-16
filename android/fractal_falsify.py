"""
fractal_falsify.py — Monte Carlo fractal falsification for Akataleptos

Given a random 3D removal pattern on a b×b×b grid, build the contact graph,
compute eigenvalues, and score how well the spectrum reproduces the 13 physical
constants that the Menger sponge produces.

If ONLY the Menger sponge (remove subcubes with 2+ center coords from 3×3×3)
produces these constants, that's strong evidence the framework is not numerology.

Work unit: seed → removal pattern → contact graph → eigenvalues → match score
"""

import numpy as np
from itertools import product as iterproduct
try:
    from scipy import linalg
except ImportError:
    from numpy import linalg
import hashlib
import json


# ═══════════════════════════════════════════════════════════
# The 13 target constants (from Menger sponge)
# ═══════════════════════════════════════════════════════════

TARGETS = {
    'fine_structure':   137.035999084,
    'muon_mass':        206.768283,
    'proton_mass':      1836.15267343,
    'strong_coupling':  0.1179,
    'cabibbo':          0.2253,
    'mw_mz':            0.88147,
    'higgs':            125.25,
    'top_quark':        173.1,
    'z_boson':          91.1876,
    'w_boson':          80.379,
    'vcb':              0.0408,
    'wolfenstein_a':    0.804,
    'cp_phase':         1.144,
}

# Tolerance for matching (relative)
MATCH_TOL = 0.01  # 1%


# ═══════════════════════════════════════════════════════════
# Fractal graph construction
# ═══════════════════════════════════════════════════════════

def generate_removal_pattern(b, seed):
    """Generate a random removal pattern for a b×b×b grid.
    Returns set of (x,y,z) positions to KEEP.

    Constraints:
    - Must remove at least 1 subcube (otherwise it's just a solid cube)
    - Must keep the graph connected (at least face-adjacent path between all kept)
    - Must keep at least b^2 subcubes (need enough for a spectrum)
    """
    rng = np.random.default_rng(seed)
    all_pos = [(x, y, z) for x, y, z in iterproduct(range(b), repeat=3)]
    n_total = b ** 3

    # Random number to remove: between 1 and n_total // 2
    max_remove = max(1, n_total // 2)
    n_remove = rng.integers(1, max_remove + 1)

    # Random selection of positions to remove
    remove_idx = rng.choice(n_total, n_remove, replace=False)
    remove_set = set(all_pos[i] for i in remove_idx)
    kept = [p for p in all_pos if p not in remove_set]

    # Check connectivity via BFS
    if len(kept) < 2:
        return kept, remove_set

    kept_set = set(kept)
    visited = {kept[0]}
    queue = [kept[0]]
    while queue:
        pos = queue.pop(0)
        for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            nb = (pos[0]+dx, pos[1]+dy, pos[2]+dz)
            if nb in kept_set and nb not in visited:
                visited.add(nb)
                queue.append(nb)

    if len(visited) < len(kept):
        # Not connected — take only the largest component
        kept = list(visited)

    return kept, remove_set


def build_contact_graph(positions):
    """Build adjacency matrix for face-adjacent positions.
    Returns adjacency matrix and number of edges.
    """
    n = len(positions)
    if n == 0:
        return np.zeros((0, 0)), 0

    pos_to_idx = {p: i for i, p in enumerate(positions)}
    adj = np.zeros((n, n), dtype=np.float64)

    for i, p in enumerate(positions):
        for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            nb = (p[0]+dx, p[1]+dy, p[2]+dz)
            if nb in pos_to_idx:
                adj[i, pos_to_idx[nb]] = 1.0

    n_edges = int(np.sum(adj)) // 2
    return adj, n_edges


def build_level2(positions_l1, b):
    """Build level-2 fractal by replacing each kept L1 position with a
    scaled copy of the kept pattern. Returns L2 positions."""
    l2_positions = []
    for p1 in positions_l1:
        for p2 in positions_l1:
            # L2 position = p1 * b + p2
            l2_positions.append((p1[0]*b + p2[0], p1[1]*b + p2[1], p1[2]*b + p2[2]))
    return l2_positions


# ═══════════════════════════════════════════════════════════
# Spectral analysis
# ═══════════════════════════════════════════════════════════

def compute_spectrum(adj):
    """Compute eigenvalues of graph Laplacian and adjacency matrix."""
    n = adj.shape[0]
    if n < 2:
        return np.array([]), np.array([])

    # Degree matrix
    deg = np.diag(adj.sum(axis=1))
    laplacian = deg - adj

    # Eigenvalues
    lap_eigs = np.sort(np.real(linalg.eigvalsh(laplacian)))
    adj_eigs = np.sort(np.real(linalg.eigvalsh(adj)))[::-1]  # descending

    return lap_eigs, adj_eigs


def extract_parameters(positions, adj, lap_eigs, adj_eigs):
    """Try to extract Menger-like parameters from the spectrum.

    For the true Menger sponge:
    - b=3 (base), d=3 (dimension)
    - S=5 (trace = largest eigenvalue of adjacency)
    - P=2 (product = determinant relationship)
    - Delta=17 (discriminant = S^2 - 4P)
    - r=7 (removed count)
    - k=20 (kept count = b^d - r)

    We extract analogues for arbitrary fractals.
    """
    n_kept = len(positions)
    # Can't determine b and d directly for arbitrary patterns
    # but we can get S, P from the characteristic polynomial of adj

    if len(adj_eigs) < 2:
        return None

    # S = largest eigenvalue (adjacency)
    S_val = adj_eigs[0]

    # For the Menger sponge, the adj eigenvalues satisfy x^2 - Sx + P = 0
    # So P = product of the two largest eigenvalues
    # More generally: look at the spectral gap structure

    # Try to identify the dominant polynomial
    # If eigenvalues cluster, the "effective" polynomial might be
    # x^2 - S*x + P where S = sum of roots, P = product

    # Use the two most extreme eigenvalues
    lam1 = adj_eigs[0]  # largest
    lam2 = adj_eigs[-1]  # smallest (most negative)

    S_eff = lam1 + lam2  # trace-like
    P_eff = lam1 * lam2  # det-like

    # Also compute from Laplacian
    # Fiedler value (algebraic connectivity) = 2nd smallest Laplacian eigenvalue
    fiedler = lap_eigs[1] if len(lap_eigs) > 1 else 0

    # Spectral radius
    spectral_radius = max(abs(adj_eigs[0]), abs(adj_eigs[-1]))

    return {
        'n_kept': n_kept,
        'n_edges': int(np.sum(adj)) // 2,
        'S': float(S_val),
        'P_eff': float(P_eff),
        'S_eff': float(S_eff),
        'Delta_eff': float(S_eff**2 - 4*P_eff),
        'fiedler': float(fiedler),
        'spectral_radius': float(spectral_radius),
        'spectral_gap': float(adj_eigs[0] - adj_eigs[1]) if len(adj_eigs) > 1 else 0,
    }


def eval_formulas(b, d, S, P, r, k):
    """Apply the 13 algebraic formulas from the Akataleptos paper.
    Returns dict of {constant_name: predicted_value}.

    Parameters are structural: b=base, d=dim, S=trace, P=product,
    r=removed, k=kept. Delta = S²-4P.
    """
    Delta = S**2 - 4*P
    preds = {}

    try:
        if k > 0 and P > 0 and S > 0 and Delta != 0:
            preds['fine_structure'] = S*b**3 + P + (P*b)**2 / (k/P)**3
            preds['muon_mass'] = P * (S*k + d) + P**5 * b / S**3
            preds['proton_mass'] = b**2 * Delta * (P**2 * b + (P/k)**3)
            preds['strong_coupling'] = P / Delta
            preds['cabibbo'] = b**2 / (Delta + k + d)
            preds['mw_mz'] = 1 - P/Delta - (P/k)**3
            preds['higgs'] = S**3 + S/k
            preds['top_quark'] = Delta * k / P + d + P/k
            preds['z_boson'] = k*S - r - P + d / P**4
            preds['w_boson'] = k*(S-1) + P*(k-1) / (S*k)
            if r > 0:
                preds['vcb'] = P / r**2
            preds['wolfenstein_a'] = k / S**2 + P**2 * (P/k)**3
            if abs(P/S) <= 1:
                preds['cp_phase'] = np.arccos(P/S)
    except (ZeroDivisionError, ValueError, OverflowError):
        pass

    return preds


def score_constants(b, d, n_kept, n_removed, params, lap_eigs, adj_eigs):
    """Score how many of the 13 physical constants can be reproduced.

    The paper's 7 parameters are STRUCTURAL, not spectral:
      b=3, d=3 are fixed (grid base, dimension)
      r, k are determined by the removal pattern
      S, P are the key unknowns

    For the Menger sponge, S=5, P=2 are forced by the removal rule
    (P=2 = number of center coords in threshold).

    For arbitrary fractals, we TRY ALL integer (S,P) pairs and report
    the best-scoring combination. This is the maximally generous test —
    even with the best possible S,P, random fractals should score low.
    """
    r = n_removed
    k = n_kept

    best_score = 0
    best_matches = []
    best_S = 0
    best_P = 0

    # Try all integer (S, P) pairs — generous search
    for S in range(1, 12):
        for P in range(1, S):  # P < S always
            preds = eval_formulas(b, d, S, P, r, k)

            matches = []
            for name, pred in preds.items():
                target = TARGETS.get(name)
                if target and target != 0:
                    err = abs(pred - target) / target
                    if err < MATCH_TOL:
                        matches.append({
                            'constant': name,
                            'predicted': float(pred),
                            'target': target,
                            'error_pct': err * 100,
                            'formula': f'S={S}, P={P}, b={b}, d={d}, r={r}, k={k}',
                        })

            if len(matches) > best_score:
                best_score = len(matches)
                best_matches = matches
                best_S = S
                best_P = P

    return best_score, best_matches


# ═══════════════════════════════════════════════════════════
# Work unit
# ═══════════════════════════════════════════════════════════

def run_work_unit(seed, b=3, level=1):
    """Execute one falsification work unit.

    Args:
        seed: random seed (deterministic)
        b: grid base (3, 4, or 5)
        level: fractal iteration depth (1 or 2)

    Returns dict with results.
    """
    import time
    t0 = time.perf_counter()

    # Generate fractal
    positions, removed = generate_removal_pattern(b, seed)
    n_kept = len(positions)
    n_removed = len(removed)

    if n_kept < 4:
        return {
            'seed': seed, 'b': b, 'level': level,
            'status': 'too_small', 'n_kept': n_kept,
            'compute_seconds': time.perf_counter() - t0,
        }

    # Build higher level if requested
    if level >= 2 and n_kept <= 30:  # cap L2 at 30 L1 positions (900 L2 vertices)
        positions = build_level2(positions, b)

    # Build contact graph
    adj, n_edges = build_contact_graph(positions)

    # Compute spectrum
    lap_eigs, adj_eigs = compute_spectrum(adj)

    # Extract parameters
    params = extract_parameters(positions, adj, lap_eigs, adj_eigs)

    # Score against physical constants using algebraic formulas
    n_matched, match_details = score_constants(
        b, 3, len(positions), n_removed, params, lap_eigs, adj_eigs
    )

    dt = time.perf_counter() - t0

    # Check if this is the actual Menger pattern
    is_menger = False
    if b == 3 and level == 1:
        menger_removed = set()
        for x, y, z in iterproduct(range(3), repeat=3):
            if sum(1 for c in (x, y, z) if c == 1) >= 2:
                menger_removed.add((x, y, z))
        is_menger = (removed == menger_removed)

    # Hash for integrity
    eig_hash = hashlib.sha256(
        json.dumps([round(float(e), 10) for e in sorted(lap_eigs)],
                   separators=(',', ':')).encode()
    ).hexdigest()

    return {
        'seed': seed,
        'b': b,
        'level': level,
        'n_kept': len(positions),
        'n_removed': n_removed,
        'n_edges': n_edges,
        'n_eigenvalues': len(lap_eigs),
        'eigenvalues': lap_eigs.tolist(),
        'params': params,
        'n_matched': n_matched,
        'matches': match_details,
        'is_menger': is_menger,
        'eigenvalues_hash': eig_hash,
        'compute_seconds': dt,
    }


# ═══════════════════════════════════════════════════════════
# Batch runner (for local testing)
# ═══════════════════════════════════════════════════════════

def exhaustive_threshold_scan(max_base=7):
    """Exhaustively scan ALL threshold-based removal rules for bases 2-max_base.

    For each base b and threshold P (remove subcubes with ≥P center coords):
    - Compute k, r, S=b+P, Delta=S²-4P
    - Evaluate all 13 formulas
    - Report match count

    This is the REAL falsification: does any (b, P) other than (3, 2)
    produce 12+ physical constants?
    """
    print(f"Exhaustive threshold scan: bases 2-{max_base}")
    print(f"{'='*70}")
    print(f"{'b':>3} {'P':>3} {'S':>3} {'Δ':>4} {'k':>6} {'r':>5} {'Match':>5}  Constants")
    print(f"{'-'*3} {'-'*3} {'-'*3} {'-'*4} {'-'*6} {'-'*5} {'-'*5}  {'-'*30}")

    for b in range(2, max_base + 1):
        d = 3
        n_total = b ** d
        for P in range(1, d + 1):
            # Count subcubes with ≥P center coords
            r = 0
            for coords in iterproduct(range(b), repeat=d):
                center_count = sum(1 for c in coords if c == (b-1)//2)
                # For even b, there's no single center — skip
                if b % 2 == 0:
                    break
                if center_count >= P:
                    r += 1
            if b % 2 == 0:
                continue  # even bases don't have clean center definition

            k = n_total - r
            S = b + P
            Delta = S**2 - 4*P

            preds = eval_formulas(b, d, S, P, r, k)
            n_match = 0
            matched_names = []
            for name, pred in preds.items():
                target = TARGETS.get(name)
                if target and abs(pred - target) / target < MATCH_TOL:
                    n_match += 1
                    matched_names.append(name)

            marker = ' ★★★ MENGER' if (b == 3 and P == 2) else ''
            print(f"{b:3d} {P:3d} {S:3d} {Delta:4d} {k:6d} {r:5d} {n_match:5d}  "
                  f"{', '.join(matched_names[:5])}{marker}")

    print(f"\n{'='*70}")
    print("If only (b=3, P=2) matches 12+, the Menger sponge is uniquely selected.")


def run_batch(n_trials=1000, b=3, level=1):
    """Run a batch of random falsification trials locally."""
    import time

    print(f"Running {n_trials} falsification trials (b={b}, level={level})")
    print(f"{'='*70}")

    results = []
    best_match = 0
    menger_matches = 0
    t0 = time.perf_counter()

    for seed in range(n_trials):
        r = run_work_unit(seed, b=b, level=level)
        results.append(r)

        if r.get('n_matched', 0) > best_match:
            best_match = r['n_matched']
            print(f"\n  NEW BEST: seed={seed}, {r['n_matched']} constants matched!")
            for m in r.get('matches', []):
                val = m.get('predicted', m.get('ratio', 0))
                print(f"    {m['constant']}: {val:.6f} (target {m['target']}, "
                      f"error {m['error_pct']:.3f}%)")

        if r.get('is_menger'):
            menger_matches = r.get('n_matched', 0)
            print(f"\n  [MENGER] seed={seed}: {menger_matches} constants matched")

        if (seed + 1) % 100 == 0:
            dt = time.perf_counter() - t0
            match_dist = {}
            for res in results:
                nm = res.get('n_matched', 0)
                match_dist[nm] = match_dist.get(nm, 0) + 1
            print(f"  {seed+1}/{n_trials} | {(seed+1)/dt:.0f}/s | "
                  f"match distribution: {dict(sorted(match_dist.items()))}")

    dt = time.perf_counter() - t0

    # Final statistics
    match_counts = [r.get('n_matched', 0) for r in results if r.get('status') != 'too_small']
    valid = [r for r in results if r.get('status') != 'too_small']

    print(f"\n{'='*70}")
    print(f"RESULTS: {len(valid)} valid trials in {dt:.1f}s ({len(valid)/dt:.0f}/s)")
    print(f"  Mean matches: {np.mean(match_counts):.2f}")
    print(f"  Max matches: {max(match_counts) if match_counts else 0}")
    print(f"  Menger matches: {menger_matches}")

    # Distribution
    from collections import Counter
    dist = Counter(match_counts)
    print(f"  Distribution:")
    for kk in sorted(dist.keys()):
        bar = '#' * min(dist[kk], 50)
        print(f"    {kk:2d} constants: {dist[kk]:5d} ({100*dist[kk]/len(valid):.1f}%) {bar}")

    # Key question: does anyone beat or tie Menger?
    beats_menger = sum(1 for c in match_counts if c >= menger_matches and menger_matches > 0)
    print(f"\n  Fractals matching >= Menger ({menger_matches}): {beats_menger}")

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=1000)
    parser.add_argument('--base', type=int, default=3)
    parser.add_argument('--level', type=int, default=1)
    parser.add_argument('--exhaustive', action='store_true',
                        help='Exhaustive threshold scan across all odd bases')
    parser.add_argument('--max-base', type=int, default=11)
    args = parser.parse_args()

    if args.exhaustive:
        exhaustive_threshold_scan(max_base=args.max_base)
    else:
        run_batch(n_trials=args.trials, b=args.base, level=args.level)
