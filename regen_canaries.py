#!/usr/bin/env python3
"""Regenerate canary hashes at 10-decimal precision.
Step 1: Run locally to compute canaries.json (slow, CPU-bound)
Step 2: scp canaries.json to VPS, run import_canaries.py (fast, DB-only)
"""
import json, hashlib, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import w_operator as wop
import numpy as np

def hash_eigs_10(eigs):
    rounded = [round(float(e), 10) for e in sorted(eigs)]
    payload = json.dumps(rounded, separators=(',',':'))
    return hashlib.sha256(payload.encode()).hexdigest()

canary_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "canaries.json")

if os.path.exists(canary_path):
    with open(canary_path) as f:
        canaries = json.load(f)
    print(f"Loaded {len(canaries)} existing canaries, regenerating hashes...")
else:
    # Generate fresh canaries
    print("Generating 50 new canaries...")
    canaries = []
    rng = np.random.RandomState(42)
    LAMBDA_START, LAMBDA_END = 0.4, 0.6
    for i in range(50):
        lam = round(LAMBDA_START + rng.random() * (LAMBDA_END - LAMBDA_START), 6)
        canaries.append({"lambda": lam, "hash": ""})

updated = 0
for i, c in enumerate(canaries):
    lam = c["lambda"]
    try:
        vertices, edges, b_ids = wop.build_graph(2, 16, 16, 8, 2)
        upd, psi_by = wop.add_glue_edges(vertices, b_ids, lam, 1000.0, 16, 16)
        edges_m = wop.merge_edges(edges, upd)
        L, _ = wop.build_magnetic_laplacian(vertices, edges_m, s=(0,0), psi_by_id=psi_by)
        eigs = wop.solve_spectrum(L, M=40)
        new_hash = hash_eigs_10(eigs.tolist())
        c["hash"] = new_hash
        updated += 1
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{len(canaries)}")
    except Exception as e:
        print(f"  Canary {i} failed: {e}")

with open(canary_path, "w") as f:
    json.dump(canaries, f, indent=2)

print(f"Done. {updated}/{len(canaries)} canary hashes written to canaries.json")
print("Next: scp canaries.json to VPS, then run import_canaries.py")
