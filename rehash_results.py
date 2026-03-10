#!/usr/bin/env python3
"""Rehash all existing results from 12-decimal to 10-decimal precision.
Run on VPS: python3 rehash_results.py
This will update eigenvalues_hash in the results table so quorum validation works."""

import sqlite3
import json
import hashlib

DB_PATH = "hive.db"

def hash_eigenvalues_10(eigs):
    rounded = [round(float(e), 10) for e in sorted(eigs)]
    payload = json.dumps(rounded, separators=(',', ':'))
    return hashlib.sha256(payload.encode()).hexdigest()

def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    results = conn.execute("SELECT id, eigenvalues_json, eigenvalues_hash FROM results").fetchall()
    print(f"Found {len(results)} results to rehash")

    updated = 0
    for r in results:
        eigs = json.loads(r['eigenvalues_json'])
        new_hash = hash_eigenvalues_10(eigs)
        if new_hash != r['eigenvalues_hash']:
            conn.execute("UPDATE results SET eigenvalues_hash = ? WHERE id = ?", (new_hash, r['id']))
            updated += 1

    conn.commit()
    print(f"Updated {updated} hashes (out of {len(results)})")

    # Now re-run quorum validation on all jobs with 2+ results
    jobs = conn.execute("""
        SELECT DISTINCT j.id, j.verified
        FROM jobs j
        JOIN results r ON r.job_id = j.id
        WHERE j.verified = 0
        GROUP BY j.id
        HAVING COUNT(r.id) >= 2
    """).fetchall()

    print(f"\nRe-checking quorum for {len(jobs)} jobs with 2+ results...")

    verified_count = 0
    for job in jobs:
        # Check desktop tier
        desktop_results = conn.execute(
            "SELECT eigenvalues_hash FROM results WHERE job_id = ? AND param_tier = 'desktop'",
            (job['id'],)
        ).fetchall()

        if len(desktop_results) >= 2:
            hash_counts = {}
            for dr in desktop_results:
                h = dr['eigenvalues_hash']
                hash_counts[h] = hash_counts.get(h, 0) + 1

            for h, count in hash_counts.items():
                if count >= 2:
                    conn.execute(
                        "UPDATE jobs SET verified = 1, status = 'verified' WHERE id = ?",
                        (job['id'],)
                    )
                    verified_count += 1
                    print(f"  Job {job['id']} VERIFIED ({count} matching desktop results)")
                    break

    conn.commit()
    conn.close()
    print(f"\nDone. {verified_count} jobs newly verified.")

if __name__ == "__main__":
    main()
