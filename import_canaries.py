#!/usr/bin/env python3
"""Import pre-computed canary hashes into hive.db. Fast — no eigenvalue computation."""
import json, sqlite3

with open("canaries.json") as f:
    canaries = json.load(f)

conn = sqlite3.connect("hive.db")
updated = 0
for c in canaries:
    cur = conn.execute("UPDATE jobs SET canary_hash=? WHERE lambda_val=? AND is_canary=1",
                       (c["hash"], c["lambda"]))
    if cur.rowcount > 0:
        updated += 1
conn.commit()
conn.close()
print(f"Imported {updated}/{len(canaries)} canary hashes into hive.db")
