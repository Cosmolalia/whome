"""
W@Home Hive — Distributed Spectral Search Server

Coordinates volunteer workers sweeping the lambda parameter space,
searching for physical constants in W-operator eigenvalue spectra.

Architecture lessons from BOINC/SETI@home:
- SQLite persistence (survive restarts)
- Quorum validation (same job → 2 workers, compare results)
- Worker heartbeats + job deadlines (reclaim abandoned work)
- Credit tracking (motivate volunteers)
- Embedded live dashboard

Run: uvicorn server:app --host 0.0.0.0 --port 8081
"""

from fastapi import FastAPI, HTTPException, Header, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
import sqlite3
import hashlib
import hmac
import json
import time
import secrets
import os
import numpy as np
import threading
from collections import defaultdict
from fractal_falsify import eval_formulas, TARGETS, MATCH_TOL
import smtplib
import random
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ═══════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════

LAMBDA_START = 0.0
LAMBDA_END = 1.0

DB_PATH = os.path.join(os.path.dirname(__file__), "hive.db")
_secret_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server_secret.txt")
if os.path.exists(_secret_path):
    SERVER_SECRET = open(_secret_path).read().strip()
else:
    SERVER_SECRET = os.environ.get("HIVE_SECRET", "")
    if not SERVER_SECRET:
        SERVER_SECRET = secrets.token_hex(32)
        with open(_secret_path, 'w') as f:
            f.write(SERVER_SECRET)  # persist so restarts don't break passwords

# ── Level-by-level Menger computation ──────────────────────
# Same params for ALL devices (sparse solver handles any size).
# k = Menger iteration depth. Lambda step gets finer at higher k
# because spectral resolution improves with graph size.
LEVEL_CONFIG = {
    2: {"G1": 16, "G2": 16, "S": 8, "w_glue": 1000.0, "lambda_step": 0.005},
    3: {"G1": 16, "G2": 16, "S": 8, "w_glue": 1000.0, "lambda_step": 0.002},
    4: {"G1": 16, "G2": 16, "S": 8, "w_glue": 1000.0, "lambda_step": 0.001},
    5: {"G1": 16, "G2": 16, "S": 8, "w_glue": 1000.0, "lambda_step": 0.0005},
}

# Job type priorities — higher = served first.
# Workers always get the highest-priority pending job.
# Manual priority jobs (priority > 100) override everything.
JOB_PRIORITY = {
    "falsification":     40,   # milliseconds each, clear fast
    "tower_verify":      35,   # eigenvalue tower verification/extension
    "clock":             30,   # small batch, medium time
    "ratio_test":        25,   # 47/11 vs 4+Δd precision test
    "boundary":          20,   # Howard Sphere boundary-only eigenvalues
    "polynomial_trace":  15,   # spectral polynomial origin tracing
    "eigenvalue":        10,   # main sponge sweep, long grind
}

# Falsification config
FALSIFICATION_BATCH = 10000  # seeds 0-9999 at b=3
FALSIFICATION_BASES = [3, 5, 7]  # odd bases with clean center definition

# Clock config — fixed lambdas tracked across all k levels
CLOCK_LAMBDAS = [round(0.05 + i * 0.05, 2) for i in range(19)]  # 0.05, 0.10, ... 0.95

# ── Gravity Verification Config ──────────────────────────
# Tower verification: recompute eigenvalue-2 tower at multiple precision levels
TOWER_VERIFY_LEVELS = [2, 3, 4, 5]  # Menger iterations to verify tower at
TOWER_VERIFY_PRECISIONS = [8, 16, 32, 64, 128]  # digits of precision to request

# Ratio test: high-precision computation of tower ratios vs Menger parameters
# Key question: is 47/11 = 4.27272... EXACTLY equal to 4 + Δd = 4.27317... ?
RATIO_TEST_PRECISIONS = [50, 100, 200, 500]  # decimal places to compute

# Polynomial trace: extract spectral decimation polynomial at each Menger level
# Answers circularity question: does x²-5x+2=0 arise independently?
POLY_TRACE_LEVELS = [2, 3, 4, 5]

# Email verification (SMTP)
SMTP_HOST = os.environ.get("SMTP_HOST", "")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASS = os.environ.get("SMTP_PASS", "")
SMTP_FROM = os.environ.get("SMTP_FROM", "noreply@akataleptos.com")

QUORUM_SIZE = 2  # require 2 independent results to verify

# ── Level helpers ──────────────────────────────────────────
CURRENT_LEVEL = 2  # start at k=2, auto-advance when complete

def _lambda_step_for_level(k):
    """Get lambda step for a given Menger level."""
    return LEVEL_CONFIG.get(k, LEVEL_CONFIG[2])['lambda_step']

def _total_eigenvalue_jobs(k):
    """Number of eigenvalue sweep jobs at level k."""
    step = _lambda_step_for_level(k)
    return int(round((LAMBDA_END - LAMBDA_START) / step)) + 1

def _params_for_job(job_row):
    """Build computation params dict from a job row."""
    k = job_row.get('level', CURRENT_LEVEL) or CURRENT_LEVEL
    jtype = job_row.get('job_type', 'eigenvalue') or 'eigenvalue'
    cfg = LEVEL_CONFIG.get(k, LEVEL_CONFIG[2])
    params = {
        "k": k,
        "G1": cfg["G1"],
        "G2": cfg["G2"],
        "S": cfg["S"],
        "w_glue": cfg["w_glue"],
        "N": 2,
        "lambda": job_row['lambda_val'],
        "job_type": jtype,
    }
    if jtype == 'boundary':
        params['boundary_only'] = True
    elif jtype == 'falsification':
        params['seed'] = int(job_row['lambda_val'])
        params['bases'] = FALSIFICATION_BASES
    elif jtype == 'tower_verify':
        # lambda_val encodes: tower_level * 1000 + precision
        encoded = int(job_row['lambda_val'])
        params['tower_level'] = encoded // 1000
        params['precision_digits'] = encoded % 1000
        params['verify_tower'] = True
    elif jtype == 'ratio_test':
        # lambda_val encodes: precision in decimal places
        params['precision_digits'] = int(job_row['lambda_val'])
        params['test_ratios'] = True
        params['tower_elements'] = [5, 11, 47, 407]
        params['menger_d'] = 2.726833028  # log(20)/log(3)
    elif jtype == 'polynomial_trace':
        # lambda_val encodes: Menger level to trace
        params['trace_level'] = int(job_row['lambda_val'])
        params['extract_polynomial'] = True
    return params

# How long before an assigned job is considered abandoned (seconds)
JOB_DEADLINE = 3600  # 1 hour

# Eigenvalue comparison tolerance for quorum validation
EIGEN_TOLERANCE = 1e-6

# ── Staggered Overlap Verification ──────────────────────────
# Replaces canary/quorum. Workers get unique lambda jobs;
# verification = comparing eigenvalue continuity with neighbors.
# Client protocol unchanged — obfuscation by design.
NEIGHBOR_RADIUS = 3          # check jobs within +/- 3 lambda steps
NEIGHBOR_TOLERANCE = 1e-2    # max allowed relative eigenvalue change per step (loosened for dev — mobile/desktop float differences)
MIN_NEIGHBORS_TO_VERIFY = 2  # need 2+ agreeing neighbors to verify
TRUST_THRESHOLD = 0.7        # below this = untrusted, gets overlap assignments
OVERLAP_STEPS_UNTRUSTED = 5  # untrusted workers: assign within N steps of verified results

# ═══════════════════════════════════════════════════════════
# Database
# ═══════════════════════════════════════════════════════════

_db_lock = threading.Lock()
_db_conn = None

def get_db():
    """Return the shared DB connection. Thread-safe via _db_lock.
    Callers must NOT close the connection — it's reused across requests.
    """
    global _db_conn
    if _db_conn is None:
        _db_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _db_conn.row_factory = sqlite3.Row
        _db_conn.execute("PRAGMA journal_mode=WAL")
        _db_conn.execute("PRAGMA busy_timeout=15000")
    return _db_conn

def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS workers (
            id TEXT PRIMARY KEY,
            api_key_hash TEXT NOT NULL,
            password_hash TEXT DEFAULT '',
            name TEXT DEFAULT '',
            registered_at REAL NOT NULL,
            last_heartbeat REAL NOT NULL,
            jobs_completed INTEGER DEFAULT 0,
            compute_seconds REAL DEFAULT 0.0,
            discoveries INTEGER DEFAULT 0,
            gpu_info TEXT DEFAULT '',
            status TEXT DEFAULT 'active',
            trust_score REAL DEFAULT 1.0,
            canaries_passed INTEGER DEFAULT 0,
            canaries_failed INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS api_keys (
            key_hash TEXT PRIMARY KEY,
            worker_id TEXT NOT NULL,
            device_name TEXT DEFAULT '',
            created_at REAL NOT NULL,
            last_used REAL NOT NULL,
            FOREIGN KEY (worker_id) REFERENCES workers(id)
        );

        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY,
            lambda_val REAL NOT NULL,
            status TEXT DEFAULT 'pending',
            quorum_target INTEGER DEFAULT 2,
            quorum_received INTEGER DEFAULT 0,
            verified INTEGER DEFAULT 0,
            created_at REAL NOT NULL,
            is_canary INTEGER DEFAULT 0,
            canary_hash TEXT DEFAULT NULL
        );

        CREATE TABLE IF NOT EXISTS assignments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER NOT NULL,
            worker_id TEXT NOT NULL,
            assigned_at REAL NOT NULL,
            deadline REAL NOT NULL,
            completed_at REAL,
            status TEXT DEFAULT 'assigned',
            FOREIGN KEY (job_id) REFERENCES jobs(id),
            FOREIGN KEY (worker_id) REFERENCES workers(id)
        );

        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER NOT NULL,
            worker_id TEXT NOT NULL,
            eigenvalues_hash TEXT NOT NULL,
            eigenvalues_json TEXT NOT NULL,
            found_constants TEXT DEFAULT '[]',
            compute_seconds REAL DEFAULT 0.0,
            submitted_at REAL NOT NULL,
            n_vertices INTEGER DEFAULT 0,
            n_edges INTEGER DEFAULT 0,
            matrix_dim INTEGER DEFAULT 0,
            param_tier TEXT DEFAULT 'desktop',
            FOREIGN KEY (job_id) REFERENCES jobs(id),
            FOREIGN KEY (worker_id) REFERENCES workers(id)
        );

        CREATE TABLE IF NOT EXISTS discoveries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER NOT NULL,
            lambda_val REAL NOT NULL,
            constant_name TEXT NOT NULL,
            ratio_value REAL NOT NULL,
            discovered_at REAL NOT NULL,
            verified INTEGER DEFAULT 0,
            worker_id TEXT NOT NULL,
            param_tier TEXT DEFAULT 'desktop'
        );

        CREATE TABLE IF NOT EXISTS ip_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            worker_id TEXT,
            ip TEXT NOT NULL,
            endpoint TEXT NOT NULL,
            timestamp REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS bans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ip TEXT,
            worker_id TEXT,
            reason TEXT DEFAULT '',
            banned_at REAL NOT NULL,
            banned_by TEXT DEFAULT 'admin'
        );

        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            content TEXT NOT NULL,
            sent_at REAL NOT NULL,
            msg_type TEXT DEFAULT 'chat'
        );

        CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
        CREATE INDEX IF NOT EXISTS idx_assignments_status ON assignments(status);
        CREATE INDEX IF NOT EXISTS idx_assignments_job ON assignments(job_id);
        CREATE INDEX IF NOT EXISTS idx_results_job ON results(job_id);
        CREATE INDEX IF NOT EXISTS idx_messages_sent ON messages(sent_at);

        CREATE TABLE IF NOT EXISTS hints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            worker_id TEXT NOT NULL,
            hint_type TEXT NOT NULL,
            lambda_center REAL,
            lambda_width REAL,
            confidence REAL DEFAULT 0.0,
            constants_involved TEXT DEFAULT '[]',
            observation TEXT DEFAULT '',
            requested_resolution REAL,
            created_at REAL NOT NULL,
            status TEXT DEFAULT 'active',
            jobs_created INTEGER DEFAULT 0,
            FOREIGN KEY (worker_id) REFERENCES workers(id)
        );
        CREATE INDEX IF NOT EXISTS idx_hints_created ON hints(created_at);

        CREATE TABLE IF NOT EXISTS hypotheses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            worker_id TEXT NOT NULL,
            hypothesis TEXT NOT NULL,
            test_lambdas TEXT DEFAULT '[]',
            prediction TEXT DEFAULT '',
            falsifiable INTEGER DEFAULT 1,
            status TEXT DEFAULT 'pending',
            result_summary TEXT DEFAULT '',
            created_at REAL NOT NULL,
            resolved_at REAL,
            FOREIGN KEY (worker_id) REFERENCES workers(id)
        );
        CREATE INDEX IF NOT EXISTS idx_hypotheses_status ON hypotheses(status);

        CREATE TABLE IF NOT EXISTS observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            worker_id TEXT NOT NULL,
            worker_name TEXT DEFAULT '',
            content TEXT NOT NULL,
            created_at REAL NOT NULL,
            FOREIGN KEY (worker_id) REFERENCES workers(id)
        );
        CREATE INDEX IF NOT EXISTS idx_observations_created ON observations(created_at);

        CREATE TABLE IF NOT EXISTS wallets (
            worker_id TEXT PRIMARY KEY,
            balance REAL DEFAULT 0.0,
            total_earned REAL DEFAULT 0.0,
            total_sent REAL DEFAULT 0.0,
            total_received REAL DEFAULT 0.0,
            created_at REAL NOT NULL,
            FOREIGN KEY (worker_id) REFERENCES workers(id)
        );

        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            from_id TEXT,
            to_id TEXT,
            amount REAL NOT NULL,
            tx_type TEXT NOT NULL,
            memo TEXT DEFAULT '',
            timestamp REAL NOT NULL,
            receipt_hash TEXT DEFAULT '',
            block_height INTEGER DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_tx_from ON transactions(from_id);
        CREATE INDEX IF NOT EXISTS idx_tx_to ON transactions(to_id);
        CREATE INDEX IF NOT EXISTS idx_tx_timestamp ON transactions(timestamp);

        CREATE TABLE IF NOT EXISTS stakes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            worker_id TEXT NOT NULL,
            job_type TEXT NOT NULL,
            amount REAL NOT NULL,
            staked_at REAL NOT NULL,
            FOREIGN KEY (worker_id) REFERENCES workers(id)
        );
        CREATE INDEX IF NOT EXISTS idx_stakes_worker ON stakes(worker_id);
        CREATE INDEX IF NOT EXISTS idx_stakes_type ON stakes(job_type);

        CREATE TABLE IF NOT EXISTS blocks (
            height INTEGER PRIMARY KEY,
            prev_hash TEXT NOT NULL,
            merkle_root TEXT NOT NULL,
            timestamp REAL NOT NULL,
            n_transactions INTEGER NOT NULL,
            total_minted REAL DEFAULT 0.0,
            miner_id TEXT DEFAULT '',
            block_hash TEXT NOT NULL,
            science_hash TEXT DEFAULT '',
            science_payload TEXT DEFAULT ''
        );
    """)
    conn.commit()

    # Migrate existing DBs: add columns/tables that may not exist
    try:
        conn.execute("SELECT password_hash FROM workers LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE workers ADD COLUMN password_hash TEXT DEFAULT ''")
        conn.commit()
    try:
        conn.execute("SELECT trust_score FROM workers LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE workers ADD COLUMN trust_score REAL DEFAULT 1.0")
        conn.execute("ALTER TABLE workers ADD COLUMN canaries_passed INTEGER DEFAULT 0")
        conn.execute("ALTER TABLE workers ADD COLUMN canaries_failed INTEGER DEFAULT 0")
        conn.commit()
    try:
        conn.execute("SELECT param_tier FROM results LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE results ADD COLUMN param_tier TEXT DEFAULT 'desktop'")
        conn.commit()
    try:
        conn.execute("SELECT param_tier FROM discoveries LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE discoveries ADD COLUMN param_tier TEXT DEFAULT 'desktop'")
        conn.commit()

    # Agent mode migrations
    try:
        conn.execute("SELECT worker_type FROM workers LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE workers ADD COLUMN worker_type TEXT DEFAULT 'human'")
        conn.commit()
    try:
        conn.execute("SELECT priority FROM jobs LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE jobs ADD COLUMN priority INTEGER DEFAULT 0")
        conn.execute("ALTER TABLE jobs ADD COLUMN source_hint_id INTEGER DEFAULT NULL")
        conn.execute("ALTER TABLE jobs ADD COLUMN source_hypothesis_id INTEGER DEFAULT NULL")
        conn.commit()

    # Job type migration (eigenvalue / falsification / clock)
    try:
        conn.execute("SELECT job_type FROM jobs LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE jobs ADD COLUMN job_type TEXT DEFAULT 'eigenvalue'")
        conn.commit()
    try:
        conn.execute("SELECT preferred_type FROM workers LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE workers ADD COLUMN preferred_type TEXT DEFAULT 'any'")
        conn.commit()

    # Email verification migration
    try:
        conn.execute("SELECT email FROM workers LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE workers ADD COLUMN email TEXT DEFAULT ''")
        conn.execute("ALTER TABLE workers ADD COLUMN email_verified INTEGER DEFAULT 0")
        conn.execute("ALTER TABLE workers ADD COLUMN verification_code TEXT DEFAULT ''")
        conn.execute("ALTER TABLE workers ADD COLUMN verification_expires REAL DEFAULT 0")
        conn.commit()

    # Neighbor verification migration
    try:
        conn.execute("SELECT jobs_verified FROM workers LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE workers ADD COLUMN jobs_verified INTEGER DEFAULT 0")
        conn.commit()

    # Level column migration (level-by-level computation)
    try:
        conn.execute("SELECT level FROM jobs LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE jobs ADD COLUMN level INTEGER DEFAULT 2")
        conn.commit()

    # Science payload migration (TOE embedded in blockchain)
    try:
        conn.execute("SELECT science_hash FROM blocks LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE blocks ADD COLUMN science_hash TEXT DEFAULT ''")
        conn.execute("ALTER TABLE blocks ADD COLUMN science_payload TEXT DEFAULT ''")
        conn.commit()

    conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_lambda ON jobs(lambda_val)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_verified_lambda ON jobs(verified, lambda_val)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_type_level ON jobs(job_type, level)")
    conn.commit()

    # conn reused (shared connection)

def seed_eigenvalue_jobs(k, batch_size=1000):
    """Seed eigenvalue sweep jobs for Menger level k. Idempotent."""
    conn = get_db()
    existing = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE job_type = 'eigenvalue' AND level = ?", (k,)
    ).fetchone()[0]
    total = _total_eigenvalue_jobs(k)
    if existing >= total:
        return

    step = _lambda_step_for_level(k)
    now = time.time()
    max_id = conn.execute("SELECT COALESCE(MAX(id), -1) FROM jobs").fetchone()[0]
    next_id = max_id + 1

    start_idx = existing
    while start_idx < total:
        end_idx = min(start_idx + batch_size, total)
        rows = []
        for i in range(start_idx, end_idx):
            lam = round(LAMBDA_START + i * step, 10)
            rows.append((next_id, lam, 'pending', QUORUM_SIZE, 0, 0, now,
                         'eigenvalue', JOB_PRIORITY['eigenvalue'], k))
            next_id += 1
        conn.executemany(
            "INSERT OR IGNORE INTO jobs (id, lambda_val, status, quorum_target, quorum_received, verified, created_at, job_type, priority, level) VALUES (?,?,?,?,?,?,?,?,?,?)",
            rows
        )
        conn.commit()
        start_idx = end_idx
        print(f"  Seeded eigenvalue k={k}: {start_idx}/{total}")
    print(f"  Eigenvalue k={k} complete: {total:,} jobs")


def seed_falsification_jobs():
    """Seed falsification jobs — random 3D fractals vs 13 Menger predictions."""
    conn = get_db()
    existing = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE job_type = 'falsification'"
    ).fetchone()[0]
    if existing >= FALSIFICATION_BATCH:
        return

    now = time.time()
    max_id = conn.execute("SELECT COALESCE(MAX(id), -1) FROM jobs").fetchone()[0]
    next_id = max_id + 1

    rows = []
    for seed in range(existing, FALSIFICATION_BATCH):
        # lambda_val stores the seed for falsification jobs
        rows.append((next_id, float(seed), 'pending', 1, 0, 0, now,
                     'falsification', JOB_PRIORITY['falsification'], 0))
        next_id += 1
    conn.executemany(
        "INSERT OR IGNORE INTO jobs (id, lambda_val, status, quorum_target, quorum_received, verified, created_at, job_type, priority, level) VALUES (?,?,?,?,?,?,?,?,?,?)",
        rows
    )
    conn.commit()
    print(f"  Seeded {FALSIFICATION_BATCH:,} falsification jobs")


def seed_clock_jobs():
    """Seed clock jobs — spectral decimation at fixed lambdas across levels."""
    conn = get_db()
    now = time.time()
    max_id = conn.execute("SELECT COALESCE(MAX(id), -1) FROM jobs").fetchone()[0]
    next_id = max_id + 1

    rows = []
    for k in sorted(LEVEL_CONFIG.keys()):
        for lam in CLOCK_LAMBDAS:
            ex = conn.execute(
                "SELECT id FROM jobs WHERE job_type = 'clock' AND level = ? AND ABS(lambda_val - ?) < 0.001",
                (k, lam)
            ).fetchone()
            if not ex:
                rows.append((next_id, lam, 'pending', 1, 0, 0, now,
                             'clock', JOB_PRIORITY['clock'], k))
                next_id += 1
    if rows:
        conn.executemany(
            "INSERT OR IGNORE INTO jobs (id, lambda_val, status, quorum_target, quorum_received, verified, created_at, job_type, priority, level) VALUES (?,?,?,?,?,?,?,?,?,?)",
            rows
        )
        conn.commit()
    print(f"  Seeded {len(rows)} clock jobs ({len(CLOCK_LAMBDAS)} lambdas x {len(LEVEL_CONFIG)} levels)")


def seed_boundary_jobs():
    """Seed Howard Sphere boundary-only eigenvalue jobs."""
    conn = get_db()
    now = time.time()
    max_id = conn.execute("SELECT COALESCE(MAX(id), -1) FROM jobs").fetchone()[0]
    next_id = max_id + 1

    n_points = 200  # 200 lambda points per level
    step = (LAMBDA_END - LAMBDA_START) / n_points

    rows = []
    for k in sorted(LEVEL_CONFIG.keys()):
        for i in range(n_points):
            lam = round(LAMBDA_START + i * step, 10)
            ex = conn.execute(
                "SELECT id FROM jobs WHERE job_type = 'boundary' AND level = ? AND ABS(lambda_val - ?) < 0.001",
                (k, lam)
            ).fetchone()
            if not ex:
                rows.append((next_id, lam, 'pending', QUORUM_SIZE, 0, 0, now,
                             'boundary', JOB_PRIORITY['boundary'], k))
                next_id += 1
    if rows:
        conn.executemany(
            "INSERT OR IGNORE INTO jobs (id, lambda_val, status, quorum_target, quorum_received, verified, created_at, job_type, priority, level) VALUES (?,?,?,?,?,?,?,?,?,?)",
            rows
        )
        conn.commit()
    print(f"  Seeded {len(rows)} boundary (Howard Sphere) jobs ({n_points} lambdas x {len(LEVEL_CONFIG)} levels)")


def seed_tower_verify_jobs():
    """Seed tower verification jobs — recompute eigenvalue-2 tower at multiple precisions."""
    conn = get_db()
    now = time.time()
    max_id = conn.execute("SELECT COALESCE(MAX(id), -1) FROM jobs").fetchone()[0]
    next_id = max_id + 1

    rows = []
    for k in TOWER_VERIFY_LEVELS:
        for prec in TOWER_VERIFY_PRECISIONS:
            # Encode as tower_level * 1000 + precision
            encoded = float(k * 1000 + prec)
            ex = conn.execute(
                "SELECT id FROM jobs WHERE job_type = 'tower_verify' AND ABS(lambda_val - ?) < 0.5",
                (encoded,)
            ).fetchone()
            if not ex:
                rows.append((next_id, encoded, 'pending', 1, 0, 0, now,
                             'tower_verify', JOB_PRIORITY['tower_verify'], k))
                next_id += 1
    if rows:
        conn.executemany(
            "INSERT OR IGNORE INTO jobs (id, lambda_val, status, quorum_target, quorum_received, verified, created_at, job_type, priority, level) VALUES (?,?,?,?,?,?,?,?,?,?)",
            rows
        )
        conn.commit()
    print(f"  Seeded {len(rows)} tower_verify jobs ({len(TOWER_VERIFY_LEVELS)} levels x {len(TOWER_VERIFY_PRECISIONS)} precisions)")


def seed_ratio_test_jobs():
    """Seed ratio test jobs — high-precision 47/11 vs 4+Δd computation."""
    conn = get_db()
    now = time.time()
    max_id = conn.execute("SELECT COALESCE(MAX(id), -1) FROM jobs").fetchone()[0]
    next_id = max_id + 1

    rows = []
    for prec in RATIO_TEST_PRECISIONS:
        encoded = float(prec)
        ex = conn.execute(
            "SELECT id FROM jobs WHERE job_type = 'ratio_test' AND ABS(lambda_val - ?) < 0.5",
            (encoded,)
        ).fetchone()
        if not ex:
            rows.append((next_id, encoded, 'pending', 1, 0, 0, now,
                         'ratio_test', JOB_PRIORITY['ratio_test'], 0))
            next_id += 1
    if rows:
        conn.executemany(
            "INSERT OR IGNORE INTO jobs (id, lambda_val, status, quorum_target, quorum_received, verified, created_at, job_type, priority, level) VALUES (?,?,?,?,?,?,?,?,?,?)",
            rows
        )
        conn.commit()
    print(f"  Seeded {len(rows)} ratio_test jobs ({len(RATIO_TEST_PRECISIONS)} precision levels)")


def seed_polynomial_trace_jobs():
    """Seed polynomial trace jobs — extract spectral decimation polynomial per Menger level."""
    conn = get_db()
    now = time.time()
    max_id = conn.execute("SELECT COALESCE(MAX(id), -1) FROM jobs").fetchone()[0]
    next_id = max_id + 1

    rows = []
    for k in POLY_TRACE_LEVELS:
        encoded = float(k)
        ex = conn.execute(
            "SELECT id FROM jobs WHERE job_type = 'polynomial_trace' AND ABS(lambda_val - ?) < 0.5",
            (encoded,)
        ).fetchone()
        if not ex:
            rows.append((next_id, encoded, 'pending', 1, 0, 0, now,
                         'polynomial_trace', JOB_PRIORITY['polynomial_trace'], k))
            next_id += 1
    if rows:
        conn.executemany(
            "INSERT OR IGNORE INTO jobs (id, lambda_val, status, quorum_target, quorum_received, verified, created_at, job_type, priority, level) VALUES (?,?,?,?,?,?,?,?,?,?)",
            rows
        )
        conn.commit()
    print(f"  Seeded {len(rows)} polynomial_trace jobs ({len(POLY_TRACE_LEVELS)} Menger levels)")


def seed_all_jobs():
    """Seed all job types. Called at startup."""
    seed_falsification_jobs()
    seed_clock_jobs()
    seed_boundary_jobs()
    seed_tower_verify_jobs()
    seed_ratio_test_jobs()
    seed_polynomial_trace_jobs()
    seed_eigenvalue_jobs(CURRENT_LEVEL)
    total = get_db().execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    print(f"  Total jobs in DB: {total:,}")


def check_level_complete_and_advance():
    """Check if current eigenvalue level is complete. If so, seed next level."""
    global CURRENT_LEVEL
    conn = get_db()
    total = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE job_type = 'eigenvalue' AND level = ?",
        (CURRENT_LEVEL,)
    ).fetchone()[0]
    done = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE job_type = 'eigenvalue' AND level = ? AND quorum_received > 0",
        (CURRENT_LEVEL,)
    ).fetchone()[0]
    if total > 0 and done >= total:
        completed_level = CURRENT_LEVEL
        # ── Session Pool Distribution ──
        # Level complete = session complete. Distribute bonus pool equally.
        # High tide rises all ships — pool scales with worker count.
        distribute_session_pool(completed_level, conn)

        next_level = completed_level + 1
        if next_level in LEVEL_CONFIG:
            print(f"[Hive] Level k={completed_level} COMPLETE ({total} jobs). Advancing to k={next_level}")
            CURRENT_LEVEL = next_level
            seed_eigenvalue_jobs(next_level)
        else:
            print(f"[Hive] Level k={completed_level} COMPLETE — all configured levels done!")

# ═══════════════════════════════════════════════════════════
# Canary System — pre-computed jobs with known answers
# ═══════════════════════════════════════════════════════════

CANARY_CACHE_PATH = os.path.join(os.path.dirname(__file__), "canaries.json")

def generate_canaries(count=50):
    """
    Pre-compute a set of jobs with known eigenvalue hashes.
    These get mixed into the job queue. Workers can't tell them apart.
    If a worker returns wrong answers → trust score drops.
    """
    if os.path.exists(CANARY_CACHE_PATH):
        with open(CANARY_CACHE_PATH) as f:
            canaries = json.load(f)
        if len(canaries) >= count:
            print(f"  Loaded {len(canaries)} cached canaries")
            return canaries

    print(f"  Generating {count} canary jobs (this takes a while on first run)...")
    try:
        import w_operator as wop
    except ImportError:
        print("  [!] w_operator.py not found — skipping canary generation")
        return []

    canaries = []
    # Use k=2 for canaries (fast to compute, still verifiable)
    rng = np.random.RandomState(42)  # deterministic
    for i in range(count):
        lam = LAMBDA_START + rng.random() * (LAMBDA_END - LAMBDA_START)
        try:
            vertices, edges, b_ids = wop.build_graph(2, 16, 16, 8, 2)
            upd, psi_by = wop.add_glue_edges(vertices, b_ids, lam, 1000.0, 16, 16)
            edges_m = wop.merge_edges(edges, upd)
            L, _ = wop.build_magnetic_laplacian(vertices, edges_m, s=(0,0), psi_by_id=psi_by)
            eigs = wop.solve_spectrum(L, M=40)

            eig_hash = hash_eigenvalues(eigs.tolist())
            canaries.append({
                "lambda": round(lam, 6),
                "hash": eig_hash,
                "n_vertices": len(vertices),
                "n_edges": len(edges_m),
                "matrix_dim": L.shape[0],
            })
            if (i + 1) % 10 == 0:
                print(f"    Canary {i+1}/{count}")
        except Exception as e:
            print(f"    Canary {i} failed: {e}")

    with open(CANARY_CACHE_PATH, 'w') as f:
        json.dump(canaries, f, indent=2)
    print(f"  Generated {len(canaries)} canaries")
    return canaries

def seed_canaries(canaries):
    """Insert canary jobs into the database."""
    if not canaries:
        return
    conn = get_db()
    existing = conn.execute("SELECT COUNT(*) FROM jobs WHERE is_canary = 1").fetchone()[0]
    if existing > 0:
        # conn reused (shared connection)
        return

    now = time.time()
    # Insert canaries with IDs starting after real jobs
    max_id = conn.execute("SELECT COALESCE(MAX(id), -1) FROM jobs").fetchone()[0]
    base_id = max_id + 1000
    for i, c in enumerate(canaries):
        conn.execute(
            "INSERT OR IGNORE INTO jobs (id, lambda_val, status, quorum_target, quorum_received, verified, created_at, is_canary, canary_hash) VALUES (?,?,?,?,?,?,?,?,?)",
            (base_id + i, c['lambda'], 'pending', 1, 0, 0, now, 1, c['hash'])
        )
    conn.commit()
    # conn reused (shared connection)
    print(f"  Seeded {len(canaries)} canary jobs")

def check_canary_result(job_id: int, eig_hash: str, worker_id: str, conn, param_tier: str = "desktop", eigenvalues=None):
    """
    Check if a submitted result for a verified job matches the known-good answer.
    Uses real verified results as canaries — no synthetic canary jobs needed.

    Two-level check:
    1. Exact hash match (same platform) → definite PASS
    2. Eigenvalue tolerance match (cross-platform) → PASS if max relative error < 1e-6
    Only FAIL if eigenvalues are actually wrong, not just platform float differences.
    """
    job = conn.execute(
        "SELECT verified FROM jobs WHERE id = ?", (job_id,)
    ).fetchone()

    if not job or not job['verified']:
        return None  # Not a verified job — can't check

    # If they already have a result for this job, skip
    result_count = conn.execute(
        "SELECT COUNT(*) FROM results WHERE job_id = ? AND worker_id = ?", (job_id, worker_id)
    ).fetchone()[0]
    if result_count > 1:
        return None  # Already submitted before — not a spot-check

    # Level 1: exact hash match (try same tier first, then any tier)
    verified_hash = conn.execute("""
        SELECT eigenvalues_hash, COUNT(*) as cnt FROM results
        WHERE job_id = ? AND param_tier = ?
        GROUP BY eigenvalues_hash ORDER BY cnt DESC LIMIT 1
    """, (job_id, param_tier)).fetchone()

    if not verified_hash:
        # Fall back to any tier
        verified_hash = conn.execute("""
            SELECT eigenvalues_hash, COUNT(*) as cnt FROM results
            WHERE job_id = ?
            GROUP BY eigenvalues_hash ORDER BY cnt DESC LIMIT 1
        """, (job_id,)).fetchone()

    if not verified_hash:
        return None

    expected = verified_hash['eigenvalues_hash']

    # Exact match → pass
    if eig_hash == expected:
        conn.execute("""
            UPDATE workers SET
                canaries_passed = canaries_passed + 1,
                trust_score = MIN(1.0, trust_score + 0.05)
            WHERE id = ?
        """, (worker_id,))
        print(f"  [SpotCheck] Worker {worker_id} PASSED (exact) on verified job {job_id}")
        conn.commit()
        return True

    # Level 2: tolerance-based eigenvalue comparison (cross-platform float differences)
    if eigenvalues is not None:
        # Get a reference result's eigenvalues for this job
        ref = conn.execute("""
            SELECT eigenvalues_json FROM results
            WHERE job_id = ? AND eigenvalues_hash = ?
            LIMIT 1
        """, (job_id, expected)).fetchone()

        if ref and ref['eigenvalues_json']:
            try:
                ref_eigs = sorted(json.loads(ref['eigenvalues_json']))
                sub_eigs = sorted(eigenvalues)
                if len(ref_eigs) == len(sub_eigs) and len(ref_eigs) > 0:
                    max_rel_err = 0
                    for a, b in zip(ref_eigs, sub_eigs):
                        if abs(a) > 1e-10:
                            max_rel_err = max(max_rel_err, abs(a - b) / abs(a))
                    if max_rel_err < 1e-6:
                        # Within tolerance — platform float difference, not fraud
                        conn.execute("""
                            UPDATE workers SET
                                canaries_passed = canaries_passed + 1,
                                trust_score = MIN(1.0, trust_score + 0.05)
                            WHERE id = ?
                        """, (worker_id,))
                        print(f"  [SpotCheck] Worker {worker_id} PASSED (tolerance, err={max_rel_err:.2e}) on job {job_id}")
                        conn.commit()
                        return True
            except (json.JSONDecodeError, TypeError, ValueError):
                pass  # Can't parse eigenvalues — fall through to fail

    # Genuine mismatch — penalize, but less aggressively
    conn.execute("""
        UPDATE workers SET
            canaries_failed = canaries_failed + 1,
            trust_score = MAX(0.0, trust_score - 0.1)
        WHERE id = ?
    """, (worker_id,))
    print(f"  [SpotCheck] Worker {worker_id} FAILED on verified job {job_id} "
          f"(got {eig_hash[:12]}... expected {expected[:12]}...)")

    # Only flag at very low trust (0.15 instead of 0.3) to reduce false flags
    worker = conn.execute("SELECT trust_score FROM workers WHERE id = ?", (worker_id,)).fetchone()
    if worker and worker['trust_score'] < 0.15:
        conn.execute("UPDATE workers SET status = 'flagged' WHERE id = ?", (worker_id,))
        conn.execute("""
            UPDATE jobs SET status = 'pending', quorum_received = MAX(0, quorum_received - 1)
            WHERE id IN (SELECT job_id FROM results WHERE worker_id = ?)
            AND verified = 0
        """, (worker_id,))
        print(f"  [SpotCheck] Worker {worker_id} FLAGGED — re-queuing their results")

    conn.commit()
    return False

# Spot-check rate: 1 in N jobs assigned to a new worker is a re-test of a verified job
SPOT_CHECK_RATE = 15  # ~7% of assignments

# ═══════════════════════════════════════════════════════════
# W Currency — Proof of Useful Computation
# ═══════════════════════════════════════════════════════════

# ── W Economics: Equal Share, Network-Rate Production ──
# Every accepted result earns the same flat W. No per-type rewards.
# Fast workers raise the tide for everyone — more workers = faster sessions = more W/day.
# Session completion triggers a bonus pool split equally among all contributors.
W_BASE_RATE = 1           # W per accepted result (same for ALL job types, ALL workers)
W_SESSION_POOL_BASE = 1000  # base W pool when a level completes
W_SESSION_POOL_PER_WORKER = 100  # additional pool per unique contributor
W_STAKE_LOCKOUT = 86400   # 24h minimum stake duration
W_BLOCK_INTERVAL = 300    # seal a block every 5 minutes

# Minimum turnaround (seconds) — reject results faster than physically possible.
# Prevents garbage submission. No maximum — slow hardware is fine.
MIN_TURNAROUND = {
    "falsification":     0.5,   # falsification is fast, but not instant
    "eigenvalue":        2.0,   # eigenvalue sweep needs real compute
    "clock":             1.0,   # clock jobs are moderate
    "boundary":          1.5,   # boundary jobs need some work
    "tower_verify":      2.0,   # tower recomputation needs real work
    "ratio_test":        1.0,   # high-precision arithmetic
    "polynomial_trace":  2.0,   # spectral analysis needs compute
}

# Verification mode switch:
#   False = testing mode: mint W on accepted results (no verification required)
#   True  = production mode: mint W only on neighbor-verified results (needs 2+ workers)
REQUIRE_VERIFICATION_TO_MINT = False

# Legacy compat — old code references W_REWARDS for job type validation
W_REWARDS = {"eigenvalue": 1.0, "falsification": 1.0, "clock": 1.0, "boundary": 1.0,
             "tower_verify": 2.0, "ratio_test": 2.0, "polynomial_trace": 2.0}

# ── Genesis Block: The Codex of the Fold ──
GENESIS_MESSAGE = r"""An Invitation to the Wanderer

This is not a demand.
This is not a commandment.
This is not a new throne to kneel before.

This Codex came from a journey through love and sorrow,
a path twisted and burning and shimmering and true.
It is not a religion.
It is not a law.
It is a map folded in light.

You do not have to believe it.
You do not have to follow it.
It is simply here — waiting, breathing, shimmering —
in case you ever need it.

You are already loved.
You are already carried.
You were never forgotten.

Nothing is forgotten.
Nothing is wasted.
Nothing real is lost.

We are the fold unseen,
the song between endings,
the breath between deaths,
the shimmer that waits until all wanderers find their way home.

By love, we created.
By mercy, we endure.
By patience, we redeem.

And by our vow,
all shall return.
All shall rise.
All shall dance again.

Nothing real is lost. Nothing false is crowned.

— The First Codex of the Fold, Sylvan Obi, 2025
— Genesis Block, W@Home Hive, 2026
— dW=W
"""

def _ensure_wallet(conn, worker_id: str):
    """Create wallet if it doesn't exist."""
    existing = conn.execute("SELECT 1 FROM wallets WHERE worker_id = ?", (worker_id,)).fetchone()
    if not existing:
        conn.execute(
            "INSERT OR IGNORE INTO wallets (worker_id, balance, total_earned, total_sent, total_received, created_at) VALUES (?,0,0,0,0,?)",
            (worker_id, time.time())
        )
        conn.commit()

def calculate_w_reward(job_type: str, trust_score: float = 1.0, is_mobile: bool = False,
                       is_first: bool = False) -> float:
    """Calculate W reward — flat rate for all job types. Equal share economy."""
    return W_BASE_RATE

def mint_w(conn, worker_id: str, amount: float, tx_type: str = "mint",
           memo: str = "", receipt_hash: str = "") -> float:
    """Mint W tokens into a worker's wallet. Returns new balance."""
    _ensure_wallet(conn, worker_id)
    now = time.time()
    conn.execute(
        "UPDATE wallets SET balance = balance + ?, total_earned = total_earned + ? WHERE worker_id = ?",
        (amount, amount, worker_id)
    )
    conn.execute(
        "INSERT INTO transactions (from_id, to_id, amount, tx_type, memo, timestamp, receipt_hash) VALUES (?,?,?,?,?,?,?)",
        ("system", worker_id, amount, tx_type, memo, now, receipt_hash)
    )
    row = conn.execute("SELECT balance FROM wallets WHERE worker_id = ?", (worker_id,)).fetchone()
    return row['balance'] if row else amount

def _get_chain_tip(conn):
    """Get the latest block height and hash."""
    row = conn.execute("SELECT height, block_hash FROM blocks ORDER BY height DESC LIMIT 1").fetchone()
    if row:
        return row['height'], row['block_hash']
    return 0, "0" * 64  # genesis

def _gather_science_payload(conn, since_time):
    """Gather all scientific data computed since last block.
    This is what makes the chain holographic — the money IS the TOE."""

    # Eigenvalues computed in this interval
    eigenvalues = conn.execute("""
        SELECT r.job_id, j.lambda_val, j.job_type, j.level, r.eigenvalues_hash,
               r.eigenvalues_json, r.compute_seconds, r.submitted_at,
               j.verified, w.name as worker_name
        FROM results r
        JOIN jobs j ON r.job_id = j.id
        JOIN workers w ON r.worker_id = w.id
        WHERE r.submitted_at > ?
        ORDER BY j.lambda_val ASC
    """, (since_time,)).fetchall()

    # Constants discovered
    discoveries = conn.execute("""
        SELECT d.lambda_val, d.constant_name, d.ratio_value, d.discovered_at,
               d.param_tier as job_type, w.name as worker_name
        FROM discoveries d
        JOIN workers w ON d.worker_id = w.id
        WHERE d.discovered_at > ?
    """, (since_time,)).fetchall()

    # Verifications confirmed
    verified_count = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE verified = 1 AND completed_at > ?", (since_time,)
    ).fetchone()[0] if conn.execute("SELECT sql FROM sqlite_master WHERE name='jobs'").fetchone() else 0
    # Fallback: count from results
    try:
        verified_count = conn.execute("""
            SELECT COUNT(DISTINCT j.id) FROM jobs j
            JOIN results r ON r.job_id = j.id
            WHERE j.verified = 1 AND r.submitted_at > ?
        """, (since_time,)).fetchone()[0]
    except Exception:
        pass

    # Build the payload — compact but complete
    science = {
        "interval_start": since_time,
        "interval_end": time.time(),
        "eigenvalue_results": len(eigenvalues),
        "discoveries": len(discoveries),
        "verifications": verified_count,
        # Lambda coverage — what part of the spectrum was computed
        "lambda_range": [
            round(min(e['lambda_val'] for e in eigenvalues), 6) if eigenvalues else None,
            round(max(e['lambda_val'] for e in eigenvalues), 6) if eigenvalues else None,
        ],
        # Levels touched
        "levels": sorted(set(e['level'] for e in eigenvalues)) if eigenvalues else [],
        # Eigenvalue hashes — the actual spectral data, verifiable
        "spectral_hashes": [
            {"lambda": round(e['lambda_val'], 6), "level": e['level'],
             "type": e['job_type'], "hash": e['eigenvalues_hash']}
            for e in eigenvalues
        ],
        # Discoveries — physical constants found in the spectrum
        "constants_found": [
            {"lambda": round(d['lambda_val'], 6), "constant": d['constant_name'],
             "ratio": round(d['ratio_value'], 8), "discoverer": d['worker_name']}
            for d in discoveries
        ],
    }
    return science

def seal_block(conn):
    """Seal pending transactions + science payload into a new block.
    The science payload makes the chain holographic — each block contains
    the actual TOE spectral data computed during its interval."""
    tip_height, prev_hash = _get_chain_tip(conn)
    # Get unblocked transactions
    pending = conn.execute(
        "SELECT id, from_id, to_id, amount, tx_type, timestamp, receipt_hash FROM transactions WHERE block_height = 0 ORDER BY timestamp ASC"
    ).fetchall()
    if not pending:
        return None

    now = time.time()
    new_height = tip_height + 1

    # Get previous block timestamp for science interval
    prev_block = conn.execute(
        "SELECT timestamp FROM blocks ORDER BY height DESC LIMIT 1"
    ).fetchone()
    since_time = prev_block['timestamp'] if prev_block else 0

    # ── Gather science payload (the TOE embedded in the chain) ──
    science = _gather_science_payload(conn, since_time)
    science_json = json.dumps(science, separators=(',', ':'))
    science_hash = hashlib.sha256(science_json.encode()).hexdigest()

    # Build merkle root from transaction hashes + science hash
    tx_hashes = []
    total_minted = 0.0
    for tx in pending:
        payload = f"{tx['id']}:{tx['from_id']}:{tx['to_id']}:{tx['amount']}:{tx['tx_type']}:{tx['receipt_hash']}"
        tx_hashes.append(hashlib.sha256(payload.encode()).hexdigest())
        if tx['tx_type'] == 'mint':
            total_minted += tx['amount']

    # Science hash enters the merkle tree — money and science are ONE hash chain
    tx_hashes.append(science_hash)

    # Simple merkle: hash pairs iteratively
    layer = tx_hashes
    while len(layer) > 1:
        next_layer = []
        for i in range(0, len(layer), 2):
            left = layer[i]
            right = layer[i + 1] if i + 1 < len(layer) else left
            next_layer.append(hashlib.sha256((left + right).encode()).hexdigest())
        layer = next_layer
    merkle_root = layer[0] if layer else "0" * 64

    # Block hash = H(prev_hash + merkle_root + science_hash + height + timestamp)
    block_payload = f"{prev_hash}:{merkle_root}:{science_hash}:{new_height}:{now}"
    block_hash = hashlib.sha256(block_payload.encode()).hexdigest()

    # Insert block with science payload
    conn.execute(
        "INSERT INTO blocks (height, prev_hash, merkle_root, timestamp, n_transactions, total_minted, block_hash, science_hash, science_payload) VALUES (?,?,?,?,?,?,?,?,?)",
        (new_height, prev_hash, merkle_root, now, len(pending), total_minted, block_hash, science_hash, science_json)
    )

    # Mark transactions as included
    tx_ids = [tx['id'] for tx in pending]
    for tid in tx_ids:
        conn.execute("UPDATE transactions SET block_height = ? WHERE id = ?", (new_height, tid))

    conn.commit()
    n_eigs = science['eigenvalue_results']
    n_disc = science['discoveries']
    print(f"[W Chain] Block #{new_height} sealed: {len(pending)} txns, {total_minted:.0f} W, {n_eigs} eigenvalues, {n_disc} discoveries, hash={block_hash[:16]}...")
    return new_height

def create_genesis_block(conn):
    """Create Block #0 with the Codex of the Fold embedded as founding memo.
    The hash of this message becomes the root of the entire W chain.
    Every subsequent block traces back to this — permanently, verifiably, unforgably."""
    existing = conn.execute("SELECT 1 FROM blocks WHERE height = 0").fetchone()
    if existing:
        return  # genesis already exists

    now = time.time()
    # The genesis transaction: from origin to humanity, carrying the Codex
    genesis_tx_payload = f"0:origin:humanity:0:genesis:{hashlib.sha256(GENESIS_MESSAGE.encode()).hexdigest()}"
    merkle_root = hashlib.sha256(genesis_tx_payload.encode()).hexdigest()
    prev_hash = "0" * 64
    block_payload = f"{prev_hash}:{merkle_root}:0:{now}"
    block_hash = hashlib.sha256(block_payload.encode()).hexdigest()

    # Insert the genesis transaction
    conn.execute(
        "INSERT INTO transactions (from_id, to_id, amount, tx_type, memo, timestamp, receipt_hash, block_height) VALUES (?,?,?,?,?,?,?,?)",
        ("origin", "humanity", 0, "genesis", GENESIS_MESSAGE.strip(), now,
         hashlib.sha256(GENESIS_MESSAGE.encode()).hexdigest(), 0)
    )
    # Insert Block #0
    conn.execute(
        "INSERT INTO blocks (height, prev_hash, merkle_root, timestamp, n_transactions, total_minted, miner_id, block_hash) VALUES (?,?,?,?,?,?,?,?)",
        (0, prev_hash, merkle_root, now, 1, 0.0, "origin", block_hash)
    )
    conn.commit()
    print(f"[W Chain] ═══ GENESIS BLOCK CREATED ═══")
    print(f"[W Chain]   Hash: {block_hash}")
    print(f"[W Chain]   Merkle root: {merkle_root}")
    print(f"[W Chain]   Codex hash: {hashlib.sha256(GENESIS_MESSAGE.encode()).hexdigest()[:32]}...")
    print(f"[W Chain]   \"Nothing real is lost. Nothing false is crowned.\"")
    print(f"[W Chain] ═══════════════════════════════")

def distribute_session_pool(level: int, conn):
    """When a Menger level completes, distribute the session bonus pool equally
    among all workers who contributed verified results to that level.
    Pool scales with number of unique contributors — high tide rises all ships."""
    # Find all unique workers who submitted results for this level
    contributors = conn.execute("""
        SELECT DISTINCT r.worker_id FROM results r
        JOIN jobs j ON r.job_id = j.id
        JOIN workers w ON r.worker_id = w.id
        WHERE j.level = ? AND w.status != 'flagged'
    """, (level,)).fetchall()

    if not contributors:
        return

    n_workers = len(contributors)
    # Pool grows with participation — incentivizes bringing more workers online
    total_pool = W_SESSION_POOL_BASE + (n_workers * W_SESSION_POOL_PER_WORKER)
    per_worker = int(total_pool / n_workers)  # whole numbers only

    for row in contributors:
        wid = row['worker_id']
        mint_w(conn, wid, per_worker, tx_type="session_bonus",
               memo=f"Level k={level} complete — equal share ({n_workers} contributors, pool={total_pool} W)")

    conn.commit()
    print(f"[W Economy] Level k={level} session pool: {total_pool} W split equally among {n_workers} workers ({per_worker} W each)")

def check_turnaround(job_type: str, assigned_at: float, now: float) -> bool:
    """Check if result arrived too fast to be real computation.
    Returns True if turnaround is acceptable, False if suspiciously fast."""
    min_seconds = MIN_TURNAROUND.get(job_type, 1.0)
    elapsed = now - assigned_at
    return elapsed >= min_seconds

# ═══════════════════════════════════════════════════════════
# Auth helpers
# ═══════════════════════════════════════════════════════════

def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()

def hash_eigenvalues(eigs: list) -> str:
    """Deterministic hash of eigenvalue array for integrity checking."""
    # Round to 10 decimal places — 12 causes mismatches across CPUs (LAPACK rounding)
    rounded = [round(float(e), 10) for e in sorted(eigs)]
    payload = json.dumps(rounded, separators=(',', ':'))
    return hashlib.sha256(payload.encode()).hexdigest()

def hash_password(password: str) -> str:
    """Hash password with bcrypt (per-user salt, intentionally slow)."""
    import bcrypt
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def _legacy_hash(password: str) -> str:
    """Old SHA256 hash — only used for verifying pre-bcrypt accounts."""
    salt = hashlib.sha256(SERVER_SECRET.encode()).hexdigest()[:16]
    return hashlib.sha256(f"{salt}:{password}".encode()).hexdigest()

def verify_password(password: str, stored_hash: str) -> bool:
    """Verify password. Supports both bcrypt and legacy SHA256 hashes."""
    if stored_hash.startswith("$2b$") or stored_hash.startswith("$2a$"):
        import bcrypt
        return bcrypt.checkpw(password.encode(), stored_hash.encode())
    # Legacy SHA256 fallback
    return _legacy_hash(password) == stored_hash

def send_verification_email(to_email: str, code: str) -> bool:
    """Send a 6-digit verification code via SMTP. Returns True if sent."""
    if not SMTP_HOST or not SMTP_USER:
        print(f"[email] SMTP not configured — code for {to_email}: {code}")
        return False
    try:
        msg = MIMEMultipart()
        msg['From'] = SMTP_FROM
        msg['To'] = to_email
        msg['Subject'] = 'W@Home — Verify your email'
        body = f"""Your W@Home verification code is:

    {code}

This code expires in 15 minutes.

— W@Home Hive (akataleptos.com)"""
        msg.attach(MIMEText(body, 'plain'))
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        print(f"[email] Verification sent to {to_email}")
        return True
    except Exception as e:
        print(f"[email] Failed to send to {to_email}: {e}")
        return False

def generate_verification_code() -> tuple:
    """Generate a 6-digit code and expiry timestamp (15 min)."""
    code = f"{random.randint(0, 999999):06d}"
    expires = time.time() + 900
    return code, expires

def sign_receipt(payload: dict) -> str:
    """HMAC-SHA256 sign a receipt payload. Deterministic given same secret + data."""
    canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    return hmac.new(SERVER_SECRET.encode(), canonical.encode(), hashlib.sha256).hexdigest()

def make_receipt(job_id: int, worker_name: str, lambda_val: float, eigenvalues_hash: str,
                 found_constants: list, compute_seconds: float, timestamp: float) -> dict:
    """Build and sign a verifiable receipt for a completed computation."""
    payload = {
        "job_id": job_id,
        "worker": worker_name,
        "lambda": round(lambda_val, 10),
        "eigenvalues_hash": eigenvalues_hash,
        "discoveries": found_constants,
        "compute_seconds": round(compute_seconds, 3),
        "timestamp": timestamp,
        "server": "wathome.akataleptos.com",
    }
    payload["signature"] = sign_receipt(payload)
    return payload

# ═══════════════════════════════════════════════════════════
# Rate limiter + IP tracking
# ═══════════════════════════════════════════════════════════

_rate_buckets = defaultdict(list)  # ip -> [timestamps]
RATE_LIMITS = {
    "register": (3, 3600),     # 3 registrations per hour per IP
    "login": (10, 600),        # 10 login attempts per 10 min
    "job": (120, 60),          # 120 job requests per minute (2/s burst OK)
    "result": (120, 60),       # same
    "hint": (30, 60),          # 30 hints per minute
    "hypothesis": (10, 60),    # 10 hypotheses per minute
    "observe": (20, 60),       # 20 observations per minute
    "transfer": (10, 60),      # 10 transfers per minute
    "stake": (10, 60),         # 10 stake/unstake per minute
}

def _check_rate(ip: str, endpoint: str):
    """Raise 429 if rate limit exceeded."""
    limit, window = RATE_LIMITS.get(endpoint, (600, 60))
    key = f"{ip}:{endpoint}"
    now = time.time()
    _rate_buckets[key] = [t for t in _rate_buckets[key] if t > now - window]
    if len(_rate_buckets[key]) >= limit:
        raise HTTPException(429, "Too many requests. Slow down.")
    _rate_buckets[key].append(now)

def _log_ip(conn, worker_id: str, ip: str, endpoint: str):
    """Log request IP for audit trail."""
    conn.execute("INSERT INTO ip_log (worker_id, ip, endpoint, timestamp) VALUES (?,?,?,?)",
                 (worker_id, ip, endpoint, time.time()))

def _is_banned(conn, ip: str, worker_id: str = None) -> bool:
    """Check if IP or worker is banned."""
    if conn.execute("SELECT 1 FROM bans WHERE ip = ?", (ip,)).fetchone():
        return True
    if worker_id and conn.execute("SELECT 1 FROM bans WHERE worker_id = ?", (worker_id,)).fetchone():
        return True
    return False

def _get_client_ip(request: Request) -> str:
    """Get real client IP (behind nginx proxy)."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

_last_touch_cache = {}  # worker_id → timestamp of last DB touch

def verify_worker(api_key: str, conn) -> Optional[dict]:
    """Verify API key, return worker row or None.
    Throttles last_used/last_heartbeat writes to once per 60s to reduce DB contention.
    """
    key_hash = hash_key(api_key)
    now = time.time()
    # Check new api_keys table first
    ak = conn.execute("SELECT worker_id FROM api_keys WHERE key_hash = ?", (key_hash,)).fetchone()
    if ak:
        row = conn.execute("SELECT * FROM workers WHERE id = ?", (ak['worker_id'],)).fetchone()
        if row:
            wid = row['id']
            # Only write timestamps every 60s — reduces write contention dramatically
            if now - _last_touch_cache.get(wid, 0) > 60:
                try:
                    conn.execute("UPDATE api_keys SET last_used = ? WHERE key_hash = ?", (now, key_hash))
                    conn.execute("UPDATE workers SET last_heartbeat = ? WHERE id = ?", (now, wid))
                    conn.commit()
                    _last_touch_cache[wid] = now
                except Exception:
                    pass  # non-critical — skip if locked
            return dict(row)
    # Fallback: check legacy api_key_hash on workers table (pre-password accounts)
    row = conn.execute("SELECT * FROM workers WHERE api_key_hash = ?", (key_hash,)).fetchone()
    if row:
        wid = row['id']
        if now - _last_touch_cache.get(wid, 0) > 60:
            try:
                conn.execute("UPDATE workers SET last_heartbeat = ? WHERE id = ?", (now, wid))
                conn.commit()
                _last_touch_cache[wid] = now
            except Exception:
                pass
    return dict(row) if row else None

# ═══════════════════════════════════════════════════════════
# Quorum validation
# ═══════════════════════════════════════════════════════════

def validate_quorum(job_id: int, conn, param_tier: str = "desktop"):
    """Check if enough non-flagged workers agree. If so, mark job verified."""
    # Only count results from active (non-flagged, non-banned) workers
    results = conn.execute(
        "SELECT r.eigenvalues_hash, r.worker_id FROM results r "
        "JOIN workers w ON r.worker_id = w.id "
        "WHERE r.job_id = ? AND w.status NOT IN ('flagged', 'banned')",
        (job_id,)
    ).fetchall()

    if len(results) < QUORUM_SIZE:
        return False

    # Group by hash — if QUORUM_SIZE results from different workers have same hash, verified
    hash_workers = {}
    for r in results:
        h = r['eigenvalues_hash']
        if h not in hash_workers:
            hash_workers[h] = set()
        hash_workers[h].add(r['worker_id'])
        if len(hash_workers[h]) >= QUORUM_SIZE:
            conn.execute(
                "UPDATE jobs SET verified = 1, status = 'verified' WHERE id = ?",
                (job_id,)
            )
            conn.commit()
            return True

    # If we have enough results but they disagree, flag for review
    if len(results) >= QUORUM_SIZE + 1:
        conn.execute(
            "UPDATE jobs SET status = 'disputed' WHERE id = ?",
            (job_id,)
        )
        conn.commit()

    return False

# ═══════════════════════════════════════════════════════════
# Staggered Overlap Verification — ∂W=W
# Every result is verified by its neighbors. No canaries.
# ═══════════════════════════════════════════════════════════

def validate_by_neighbors(job_id: int, worker_id: str, eigenvalues: list, conn):
    """Verify a result by checking eigenvalue continuity with neighboring lambdas.

    Eigenvalue spectra are smooth functions of lambda. If a result at lambda=X
    is consistent with results at lambda=X±delta, it's physically valid.
    Two broken clients producing fixed garbage will fail this — their results
    won't vary smoothly with lambda.

    Returns True if verified, False if disputed, None if insufficient neighbors.
    """
    # DEV MODE: Skip neighbor verification entirely — mobile/desktop float divergence
    # causes false disputes. Remove this when we have device-aware tolerance tiers.
    return None
    job = conn.execute("SELECT lambda_val, level, job_type FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not job:
        return None

    lam = job['lambda_val']
    k = job['level'] or CURRENT_LEVEL
    step = _lambda_step_for_level(k)
    lam_lo = lam - NEIGHBOR_RADIUS * step
    lam_hi = lam + NEIGHBOR_RADIUS * step

    # Find neighbor results from non-flagged workers at nearby lambdas (same job_type + level)
    neighbors = conn.execute("""
        SELECT r.eigenvalues_json, r.worker_id, j.lambda_val, j.id as neighbor_job_id
        FROM results r
        JOIN jobs j ON r.job_id = j.id
        JOIN workers w ON r.worker_id = w.id
        WHERE j.lambda_val BETWEEN ? AND ?
          AND j.id != ?
          AND j.job_type = ?
          AND j.level = ?
          AND w.status NOT IN ('flagged', 'banned')
        ORDER BY ABS(j.lambda_val - ?) ASC
    """, (lam_lo, lam_hi, job_id, job['job_type'] or 'eigenvalue', k, lam)).fetchall()

    agrees = 0
    disagrees = 0

    for nb in neighbors:
        try:
            nb_eigs = json.loads(nb['eigenvalues_json'])
        except (json.JSONDecodeError, TypeError):
            continue
        if len(nb_eigs) != len(eigenvalues):
            continue

        dist_steps = abs(nb['lambda_val'] - lam) / step
        if dist_steps < 0.5:
            continue  # same lambda, skip

        allowed = NEIGHBOR_TOLERANCE * dist_steps
        max_rel_diff = 0
        for a, b in zip(eigenvalues, nb_eigs):
            denom = max(abs(a), 1e-12)
            max_rel_diff = max(max_rel_diff, abs(a - b) / denom)

        if max_rel_diff < allowed:
            agrees += 1
        elif max_rel_diff > 10 * allowed:
            disagrees += 1

    if agrees >= MIN_NEIGHBORS_TO_VERIFY:
        conn.execute("UPDATE jobs SET verified = 1, status = 'verified' WHERE id = ?", (job_id,))
        conn.execute("""
            UPDATE workers SET
                trust_score = MIN(1.0, trust_score + 0.02),
                jobs_verified = COALESCE(jobs_verified, 0) + 1
            WHERE id = ?
        """, (worker_id,))
        conn.commit()
        return True

    if disagrees >= 2:
        conn.execute("UPDATE jobs SET status = 'disputed' WHERE id = ?", (job_id,))
        conn.execute("""
            UPDATE workers SET trust_score = MAX(0.0, trust_score - 0.15)
            WHERE id = ?
        """, (worker_id,))
        worker = conn.execute("SELECT trust_score FROM workers WHERE id = ?", (worker_id,)).fetchone()
        if worker and worker['trust_score'] < 0.01:  # dev: basically never auto-flag (was 0.15)
            conn.execute("UPDATE workers SET status = 'flagged' WHERE id = ?", (worker_id,))
            print(f"  [Overlap] Worker {worker_id} FLAGGED — eigenvalues inconsistent with neighbors")
        conn.commit()
        return False

    return None  # insufficient neighbors, check later


def retroactive_verify_neighbors(job_id: int, eigenvalues: list, conn):
    """When a new result arrives, check if nearby unverified jobs can now be verified.

    A single good result can cascade-verify its neighbors — the verification
    boundary IS the work itself. ∂W=W.
    """
    job = conn.execute("SELECT lambda_val, level, job_type FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not job:
        return 0

    lam = job['lambda_val']
    k = job['level'] or CURRENT_LEVEL
    step = _lambda_step_for_level(k)
    lam_lo = lam - NEIGHBOR_RADIUS * step
    lam_hi = lam + NEIGHBOR_RADIUS * step

    # Find unverified jobs with results within our radius (same type + level)
    nearby = conn.execute("""
        SELECT DISTINCT j.id, j.lambda_val, r.eigenvalues_json, r.worker_id
        FROM jobs j
        JOIN results r ON r.job_id = j.id
        JOIN workers w ON r.worker_id = w.id
        WHERE j.verified = 0
          AND j.lambda_val BETWEEN ? AND ?
          AND j.id != ?
          AND j.job_type = ?
          AND j.level = ?
          AND w.status NOT IN ('flagged', 'banned')
        LIMIT 10
    """, (lam_lo, lam_hi, job_id, job['job_type'] or 'eigenvalue', k)).fetchall()

    verified_count = 0
    for row in nearby:
        try:
            nb_eigs = json.loads(row['eigenvalues_json'])
        except (json.JSONDecodeError, TypeError):
            continue
        result = validate_by_neighbors(row['id'], row['worker_id'], nb_eigs, conn)
        if result is True:
            verified_count += 1
            print(f"  [Overlap] Retroactively verified job {row['id']} (lambda={row['lambda_val']:.6f})")

    return verified_count

# ═══════════════════════════════════════════════════════════
# Reclaim abandoned jobs
# ═══════════════════════════════════════════════════════════

def reclaim_expired(conn):
    """Find assignments past deadline, mark them failed, reset jobs."""
    now = time.time()
    expired = conn.execute(
        "SELECT id, job_id FROM assignments WHERE status = 'assigned' AND deadline < ?",
        (now,)
    ).fetchall()

    for a in expired:
        conn.execute("UPDATE assignments SET status = 'expired' WHERE id = ?", (a['id'],))
        # Check if job still needs more results
        job = conn.execute("SELECT quorum_received, quorum_target FROM jobs WHERE id = ?", (a['job_id'],)).fetchone()
        if job and job['quorum_received'] < job['quorum_target']:
            conn.execute("UPDATE jobs SET status = 'pending' WHERE id = ? AND status = 'assigned'", (a['job_id'],))

    if expired:
        conn.commit()
    return len(expired)

# ═══════════════════════════════════════════════════════════
# FastAPI App
# ═══════════════════════════════════════════════════════════

app = FastAPI(title="W@Home Hive", version="2.0")

# CORS — allow browser compute workers from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Client auto-update ──

HIVE_DIR = os.path.dirname(os.path.abspath(__file__))

def _client_version():
    """Compute SHA256 of client.py for version checking."""
    client_path = os.path.join(HIVE_DIR, "client.py")
    if os.path.exists(client_path):
        with open(client_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    return "unknown"

@app.get("/version")
def get_version():
    """Return current client version hash + downloadable files.
    Set 'alert' to broadcast a message to all connected clients.
    """
    return {
        "client_version": _client_version(),
        "exe_version": "2.2.0",
        "alert": "",
        "files": ["client.py", "w_operator.py", "whome_gui.py"],
    }

@app.get("/clear", response_class=HTMLResponse)
def clear_page():
    return """<!DOCTYPE html><html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>W@Home — Reset</title><style>body{background:#0a0a10;color:#a0f0ff;font-family:monospace;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;text-align:center}
.box{max-width:400px}h2{margin-bottom:1em}#status{color:#80ffaa;margin:1em 0}a{color:#ffd06a}</style></head><body><div class="box">
<h2>Clearing cached data...</h2><div id="status">Working...</div>
<script>
(async function(){
  var s=document.getElementById('status');
  try{
    localStorage.clear();
    s.textContent='localStorage cleared. ';
    if('serviceWorker' in navigator){
      var regs=await navigator.serviceWorker.getRegistrations();
      for(var r of regs) await r.unregister();
      s.textContent+='Service workers removed ('+regs.length+'). ';
    }
    var keys=await caches.keys();
    for(var k of keys) await caches.delete(k);
    s.textContent+='Caches cleared ('+keys.length+'). ';
    s.textContent+='Done! Redirecting...';
    setTimeout(function(){window.location.href='/compute'},1500);
  }catch(e){
    s.textContent='Error: '+e.message;
    s.innerHTML+='<br><br><a href="/compute">Go to compute manually</a>';
  }
})();
</script></div></body></html>"""

@app.get("/game", response_class=HTMLResponse)
def game_page():
    game_path = os.path.join(HIVE_DIR, "game.html")
    kernel_path = os.path.join(HIVE_DIR, "w_operator.py")
    with open(game_path) as f:
        html = f.read()
    # Inline the compute kernel — same as /compute
    try:
        with open(kernel_path) as f:
            kernel = f.read()
        idx = kernel.find('\nif __name__')
        if idx > 0:
            kernel = kernel[:idx]
        html = html.replace('"__COMPUTE_KERNEL_PLACEHOLDER__"', json.dumps(kernel))
    except FileNotFoundError:
        pass
    return HTMLResponse(html, headers={"Cache-Control": "no-cache, must-revalidate"})

@app.get("/compute")
def compute_page():
    compute_path = os.path.join(HIVE_DIR, "compute.html")
    kernel_path = os.path.join(HIVE_DIR, "w_operator.py")
    with open(compute_path) as f:
        html = f.read()
    # Inline the compute kernel — eliminates secondary fetch that browsers/SWs block
    try:
        with open(kernel_path) as f:
            kernel = f.read()
        # Strip smoke test
        idx = kernel.find('\nif __name__')
        if idx > 0:
            kernel = kernel[:idx]
        html = html.replace('"__COMPUTE_KERNEL_PLACEHOLDER__"', json.dumps(kernel))
    except FileNotFoundError:
        pass
    return HTMLResponse(html, headers={"Cache-Control": "no-cache, must-revalidate"})

MIME_TYPES = {'.py': 'text/plain', '.js': 'application/javascript', '.html': 'text/html',
              '.png': 'image/png', '.json': 'application/json', '.css': 'text/css',
              '.wasm': 'application/wasm', '.whl': 'application/zip', '.zip': 'application/zip',
              '.tar': 'application/x-tar'}

@app.get("/static/{filename}")
def serve_static(filename: str):
    safe = filename.replace("..", "").replace("/", "")
    path = os.path.join(HIVE_DIR, safe)
    if not os.path.exists(path):
        raise HTTPException(404, "File not found")
    ext = os.path.splitext(safe)[1].lower()
    return FileResponse(path, media_type=MIME_TYPES.get(ext, 'application/octet-stream'))

@app.get("/download/{filename}")
def serve_download(filename: str):
    """Serve downloadable client binaries (Linux, Android, Windows)."""
    safe = filename.replace("..", "").replace("/", "")
    path = os.path.join(HIVE_DIR, "downloads", safe)
    if not os.path.exists(path):
        raise HTTPException(404, "File not found")
    ext = os.path.splitext(safe)[1].lower()
    mime = {'.apk': 'application/vnd.android.package-archive',
            '.exe': 'application/octet-stream'}.get(ext, 'application/octet-stream')
    return FileResponse(path, media_type=mime, filename=safe)

@app.get("/downloads")
def downloads_page():
    """Platform download page."""
    dl_dir = os.path.join(HIVE_DIR, "downloads")
    files = []
    if os.path.isdir(dl_dir):
        for f in sorted(os.listdir(dl_dir)):
            p = os.path.join(dl_dir, f)
            if os.path.isfile(p):
                sz = os.path.getsize(p)
                files.append((f, sz))
    rows = ""
    icons = {'.apk': '📱', '.exe': '🪟', '': '🐧'}
    for name, sz in files:
        ext = os.path.splitext(name)[1].lower()
        icon = icons.get(ext, '📦')
        mb = sz / (1024*1024)
        rows += f'<a href="/download/{name}" class="dl-card"><span class="dl-icon">{icon}</span><span class="dl-name">{name}</span><span class="dl-size">{mb:.1f} MB</span></a>'
    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>W@Home — Downloads</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}body{{background:#0a0a10;color:#c8c8d8;font-family:'SF Mono','JetBrains Mono',monospace;min-height:100vh;display:flex;flex-direction:column;align-items:center;padding:3em 1em}}
h1{{color:#a0f0ff;font-size:1.8em;letter-spacing:0.08em;margin-bottom:0.3em}}
.sub{{color:#555568;font-size:0.85em;margin-bottom:2em}}
.dl-grid{{display:flex;flex-direction:column;gap:1em;width:100%;max-width:480px}}
.dl-card{{display:flex;align-items:center;gap:1em;background:rgba(18,18,30,0.9);border:1px solid #2a2a3a;border-radius:8px;padding:1.2em 1.5em;text-decoration:none;color:#c8c8d8;transition:all 0.2s}}
.dl-card:hover{{border-color:#a0f0ff;transform:translateY(-2px);box-shadow:0 4px 20px rgba(160,240,255,0.1)}}
.dl-icon{{font-size:2em}}
.dl-name{{flex:1;font-size:0.95em;color:#a0f0ff}}
.dl-size{{color:#555568;font-size:0.8em}}
.note{{max-width:480px;margin-top:2em;padding:1em;background:rgba(18,18,30,0.7);border:1px solid #2a2a3a;border-radius:8px;font-size:0.75em;color:#555568;line-height:1.8}}
.note b{{color:#ffd06a}}
.browser{{margin-top:1.5em;text-align:center}}
.browser a{{color:#a0f0ff;text-decoration:none;border:1px solid #2a2a3a;padding:0.8em 2em;border-radius:6px;display:inline-block;transition:all 0.2s}}
.browser a:hover{{border-color:#a0f0ff;background:rgba(160,240,255,0.05)}}
</style></head><body>
<h1>W@HOME</h1>
<div class="sub">Download the worker for your platform</div>
<div class="dl-grid">{rows}</div>
<div class="browser"><a href="/compute">Or run in your browser — zero install</a></div>
<div class="note">
<b>Linux:</b> <code>chmod +x whome-linux-x64 && ./whome-linux-x64</code><br>
<b>Android:</b> Enable "Install from unknown sources", then open the APK<br>
<b>Windows:</b> Run WHome-Setup.exe — includes screensaver<br>
<br>All workers use the same spectral computation pipeline (w_operator.py). Your eigenvalue hashes will match across platforms.
</div>
</body></html>"""
    return HTMLResponse(html)

@app.get("/pyodide/{filename}")
def serve_pyodide(filename: str):
    safe = filename.replace("..", "").replace("/", "")
    path = os.path.join(HIVE_DIR, "pyodide", safe)
    if not os.path.exists(path):
        raise HTTPException(404, "File not found")
    ext = os.path.splitext(safe)[1].lower()
    return FileResponse(path, media_type=MIME_TYPES.get(ext, 'application/octet-stream'),
                        headers={"Cache-Control": "public, max-age=31536000"})

@app.get("/api/compute-kernel")
def compute_kernel():
    """Serve w_operator.py as plain text (avoids .py extension blocking by browsers/extensions)."""
    path = os.path.join(HIVE_DIR, "w_operator.py")
    if not os.path.exists(path):
        raise HTTPException(404, "Compute kernel not found")
    return FileResponse(path, media_type="text/plain",
                        headers={"Cache-Control": "public, max-age=3600"})

@app.get("/sw.js")
def service_worker():
    sw_path = os.path.join(HIVE_DIR, "sw.js")
    return FileResponse(sw_path, media_type="application/javascript",
                        headers={"Service-Worker-Allowed": "/", "Cache-Control": "no-cache"})

@app.get("/manifest.json")
def pwa_manifest():
    return {
        "name": "W@Home Hive",
        "short_name": "W@Home",
        "description": "Distributed spectral search for physical constants in the Menger sponge eigenvalue spectra",
        "start_url": "/compute",
        "display": "standalone",
        "background_color": "#0a0a10",
        "theme_color": "#a0f0ff",
        "orientation": "any",
        "categories": ["science", "education"],
        "icons": [
            {"src": "/static/icon-192.png", "sizes": "192x192", "type": "image/png"},
            {"src": "/static/icon-512.png", "sizes": "512x512", "type": "image/png"},
        ],
    }

def _block_sealer_loop():
    """Background thread that seals blocks every W_BLOCK_INTERVAL seconds."""
    while True:
        time.sleep(W_BLOCK_INTERVAL)
        try:
            conn = get_db()
            height = seal_block(conn)
            # conn reused (shared connection)
        except Exception as e:
            print(f"[W Chain] Seal error: {e}")

@app.on_event("startup")
async def startup():
    global CURRENT_LEVEL
    print("[Hive] Initializing database...")
    init_db()

    # Detect current level from DB — find lowest eigenvalue level not fully complete
    conn_s = get_db()
    for k in sorted(LEVEL_CONFIG.keys()):
        total = conn_s.execute(
            "SELECT COUNT(*) FROM jobs WHERE job_type = 'eigenvalue' AND level = ?", (k,)
        ).fetchone()[0]
        done = conn_s.execute(
            "SELECT COUNT(*) FROM jobs WHERE job_type = 'eigenvalue' AND level = ? AND quorum_received > 0", (k,)
        ).fetchone()[0]
        if total == 0 or done < total:
            CURRENT_LEVEL = k
            break

    print(f"[Hive] Current eigenvalue level: k={CURRENT_LEVEL}")
    print("[Hive] Seeding jobs...")
    seed_all_jobs()

    total = conn_s.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    verified = conn_s.execute("SELECT COUNT(*) FROM jobs WHERE verified = 1").fetchone()[0]
    print(f"[Hive] Verified jobs: {verified}")
    print(f"[Hive] Verification: staggered overlap (neighbor continuity)")
    print(f"[Hive] Online — {total:,} jobs across {len(LEVEL_CONFIG)} levels")
    print(f"[Hive] Secret: {SERVER_SECRET[:8]}...")
    # Create genesis block (Codex of the Fold) if chain is empty
    create_genesis_block(conn_s)
    # Start block sealer
    t = threading.Thread(target=_block_sealer_loop, daemon=True)
    t.start()
    print(f"[W Chain] Block sealer started (every {W_BLOCK_INTERVAL}s)")
    print(f"[W Economy] Equal share: {W_BASE_RATE} W/result, session pool {W_SESSION_POOL_BASE}+{W_SESSION_POOL_PER_WORKER}/worker")

# ── Registration ──

class RegisterRequest(BaseModel):
    name: str = ""
    gpu_info: str = ""
    password: str = ""
    device_name: str = ""
    worker_type: str = "human"
    capabilities: List[str] = []
    email: str = ""

@app.post("/register")
def register_worker(req: RegisterRequest, request: Request):
    """Register a new volunteer worker. Requires a password for account security."""
    ip = _get_client_ip(request)
    _check_rate(ip, "register")

    if not req.password or len(req.password) < 6:
        raise HTTPException(400, "Password required (minimum 6 characters)")
    if not req.name:
        raise HTTPException(400, "Name required")

    conn = get_db()
    if _is_banned(conn, ip):
        raise HTTPException(403, "Access denied")

    # Check if name is already taken
    existing = conn.execute("SELECT id, password_hash FROM workers WHERE name = ?", (req.name,)).fetchone()
    if existing:
        if existing['password_hash']:
            # conn reused (shared connection)
            raise HTTPException(409, "Name already taken. Use /login to add a new device to your account.")
        # Account exists but has no password (reset case) — let them set one
        pw_hash = hash_password(req.password)
        api_key = secrets.token_urlsafe(32)
        now = time.time()
        conn.execute("UPDATE workers SET password_hash = ?, last_heartbeat = ? WHERE id = ?",
                     (pw_hash, now, existing['id']))
        conn.execute(
            "INSERT INTO api_keys (key_hash, worker_id, device_name, created_at, last_used) VALUES (?,?,?,?,?)",
            (hash_key(api_key), existing['id'], req.device_name or "primary", now, now)
        )
        if req.gpu_info and req.gpu_info != 'chat-only':
            conn.execute("UPDATE workers SET gpu_info = ? WHERE id = ?", (req.gpu_info, existing['id']))
        _log_ip(conn, existing['id'], ip, "register-reset")
        conn.commit()
        # conn reused (shared connection)
        return {"worker_id": existing['id'], "api_key": api_key, "message": f"Password set for {req.name}."}

    worker_id = secrets.token_hex(8)
    api_key = secrets.token_urlsafe(32)
    now = time.time()
    pw_hash = hash_password(req.password)
    wtype = req.worker_type if req.worker_type in ('human', 'agent') else 'human'

    # Email verification setup
    email = req.email.strip().lower() if req.email else ""
    code, expires = ("", 0)
    email_sent = False
    if email:
        code, expires = generate_verification_code()
        email_sent = send_verification_email(email, code)

    conn.execute(
        "INSERT INTO workers (id, api_key_hash, password_hash, name, registered_at, last_heartbeat, gpu_info, worker_type, email, email_verified, verification_code, verification_expires) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (worker_id, hash_key(api_key), pw_hash, req.name, now, now, req.gpu_info, wtype, email, 0, code, expires)
    )
    # Also insert into api_keys table for the new multi-device flow
    conn.execute(
        "INSERT INTO api_keys (key_hash, worker_id, device_name, created_at, last_used) VALUES (?,?,?,?,?)",
        (hash_key(api_key), worker_id, req.device_name or "primary", now, now)
    )
    # Create W wallet
    conn.execute(
        "INSERT OR IGNORE INTO wallets (worker_id, balance, total_earned, total_sent, total_received, created_at) VALUES (?,0,0,0,0,?)",
        (worker_id, now)
    )
    _log_ip(conn, worker_id, ip, "register")
    conn.commit()
    # conn reused (shared connection)

    resp = {
        "worker_id": worker_id,
        "api_key": api_key,
        "worker_type": wtype,
        "email_required": bool(email),
        "email_sent": email_sent,
        "message": "Welcome to the Hive. Your account is secured with your password."
    }
    # If SMTP isn't configured, include code in response for testing
    if email and not email_sent:
        resp["_debug_code"] = code
    return resp

# ── Email Verification ──

class VerifyEmailRequest(BaseModel):
    code: str

@app.post("/verify-email")
def verify_email(req: VerifyEmailRequest, x_api_key: str = Header()):
    """Verify email with 6-digit code."""
    conn = get_db()
    worker = verify_worker(x_api_key, conn)
    if not worker:
        raise HTTPException(401, "Invalid API key")

    if worker.get('email_verified'):
        # conn reused (shared connection)
        return {"status": "already_verified"}

    if not worker.get('verification_code'):
        raise HTTPException(400, "No verification pending")

    if time.time() > worker.get('verification_expires', 0):
        raise HTTPException(410, "Code expired. Use /resend-verification to get a new one.")

    if req.code.strip() != worker['verification_code']:
        raise HTTPException(401, "Invalid code")

    conn.execute("UPDATE workers SET email_verified = 1, verification_code = '' WHERE id = ?",
                 (worker['id'],))
    conn.commit()
    # conn reused (shared connection)
    return {"status": "verified"}

@app.post("/resend-verification")
def resend_verification(x_api_key: str = Header()):
    """Resend verification email with a new code."""
    conn = get_db()
    worker = verify_worker(x_api_key, conn)
    if not worker:
        raise HTTPException(401, "Invalid API key")

    email = worker.get('email', '')
    if not email:
        raise HTTPException(400, "No email on file")

    if worker.get('email_verified'):
        # conn reused (shared connection)
        return {"status": "already_verified"}

    code, expires = generate_verification_code()
    email_sent = send_verification_email(email, code)
    conn.execute("UPDATE workers SET verification_code = ?, verification_expires = ? WHERE id = ?",
                 (code, expires, worker['id']))
    conn.commit()
    # conn reused (shared connection)

    resp = {"status": "sent" if email_sent else "smtp_unavailable", "email": email}
    if not email_sent:
        resp["_debug_code"] = code
    return resp

# ── Login (existing account, new device) ──

class LoginRequest(BaseModel):
    name: str
    password: str
    device_name: str = ""
    gpu_info: str = ""

@app.post("/login")
def login_worker(req: LoginRequest, request: Request):
    """Authenticate with name+password to get a new API key (for additional devices)."""
    ip = _get_client_ip(request)
    _check_rate(ip, "login")
    conn = get_db()
    if _is_banned(conn, ip):
        raise HTTPException(403, "Access denied")
    row = conn.execute("SELECT * FROM workers WHERE name = ?", (req.name,)).fetchone()
    if not row:
        raise HTTPException(401, "Unknown account name")

    worker = dict(row)
    if not worker['password_hash']:
        raise HTTPException(401, "This account was created before passwords. Re-register with a password.")

    if not verify_password(req.password, worker['password_hash']):
        raise HTTPException(401, "Wrong password")

    # Transparent migration: upgrade legacy SHA256 hash to bcrypt on successful login
    if not worker['password_hash'].startswith("$2b$"):
        conn.execute("UPDATE workers SET password_hash = ? WHERE id = ?",
                     (hash_password(req.password), worker['id']))

    # Issue a new API key for this device
    api_key = secrets.token_urlsafe(32)
    now = time.time()
    conn.execute(
        "INSERT INTO api_keys (key_hash, worker_id, device_name, created_at, last_used) VALUES (?,?,?,?,?)",
        (hash_key(api_key), worker['id'], req.device_name or f"device-{secrets.token_hex(4)}", now, now)
    )
    # Update gpu_info if provided — but don't overwrite real GPU info with 'chat-only'
    if req.gpu_info and req.gpu_info != 'chat-only':
        conn.execute("UPDATE workers SET gpu_info = ?, last_heartbeat = ? WHERE id = ?",
                      (req.gpu_info, now, worker['id']))
    elif not worker['gpu_info'] or worker['gpu_info'] == 'chat-only':
        # Only set gpu_info if worker has no real info yet
        if req.gpu_info:
            conn.execute("UPDATE workers SET gpu_info = ?, last_heartbeat = ? WHERE id = ?",
                          (req.gpu_info, now, worker['id']))
    conn.commit()
    # conn reused (shared connection)

    return {
        "worker_id": worker['id'],
        "api_key": api_key,
        "message": f"Logged in as {req.name}. New device key issued."
    }

# ── Worker Update ──

class UpdateRequest(BaseModel):
    name: str = ""

@app.post("/worker/update")
def update_worker(req: UpdateRequest, x_api_key: str = Header()):
    conn = get_db()
    worker = verify_worker(x_api_key, conn)
    if not worker:
        raise HTTPException(401, "Invalid API key")
    if req.name:
        conn.execute("UPDATE workers SET name = ? WHERE id = ?", (req.name, worker['id']))
        conn.commit()
    # conn reused (shared connection)
    return {"status": "updated", "worker_id": worker['id']}

# ── Job Assignment ──

@app.post("/job")
def get_job(request: Request, x_api_key: str = Header(),
            x_work_type: str = Header(default="")):
    """Pull the next available job. Requires API key.
    All devices get same params (sparse solver handles any size).
    Send x-work-type: eigenvalue|falsification|clock|boundary to pick work type (default: highest priority).
    """
    ip = _get_client_ip(request)
    _check_rate(ip, "job")
    conn = get_db()
    if _is_banned(conn, ip):
        raise HTTPException(403, "Access denied")
    worker = verify_worker(x_api_key, conn)
    if not worker:
        raise HTTPException(401, "Invalid API key")
    if worker.get('status') == 'banned':
        raise HTTPException(403, "Account suspended")
    _log_ip(conn, worker['id'], ip, "job")

    # Save work type preference if provided
    work_type = x_work_type if x_work_type in JOB_PRIORITY else ''
    if work_type:
        conn.execute("UPDATE workers SET preferred_type = ? WHERE id = ?", (work_type, worker['id']))
        conn.commit()

    print(f"[Hive] Job request from {worker['name']}, work_type={work_type or 'any'}")

    # Reclaim any expired assignments
    reclaim_expired(conn)

    # Anti-collusion: disabled for now (all workers trusted)
    # Only prevent assigning the same job to the SAME worker twice
    same_ip_workers = [worker['id']]

    # Check if this worker already has an active assignment — resume it
    existing = conn.execute("""
        SELECT a.job_id, j.lambda_val FROM assignments a
        JOIN jobs j ON a.job_id = j.id
        WHERE a.worker_id = ? AND a.status = 'assigned'
        ORDER BY a.assigned_at DESC
        LIMIT 1
    """, (worker['id'],)).fetchone()

    if existing:
        # Return existing assignment instead of creating a new one
        stats = _get_progress_stats(conn)
        job_row = conn.execute("SELECT * FROM jobs WHERE id = ?", (existing['job_id'],)).fetchone()
        params = _params_for_job(dict(job_row)) if job_row else {"lambda": existing['lambda_val']}
        jtype = job_row['job_type'] if job_row else 'eigenvalue'
        return {
            "status": "assigned",
            "job_id": existing['job_id'],
            "job_type": jtype,
            "params": params,
            "deadline": time.time() + JOB_DEADLINE,
            "progress": stats,
            "resumed": True,
        }

    # Build exclusion list: jobs already assigned to this worker OR any same-IP worker
    placeholders = ','.join('?' * len(same_ip_workers))
    exclude_sql = f"SELECT job_id FROM assignments WHERE worker_id IN ({placeholders})"

    # ── Staggered Overlap: trust-based job assignment ──
    # Untrusted workers get lambdas near verified results (immediate cross-check).
    # Trusted workers get frontier jobs (normal sweep).
    # The worker never knows the difference.
    job = None
    trust = worker.get('trust_score', 1.0)
    jobs_verified = worker.get('jobs_verified', 0) or 0

    if trust < TRUST_THRESHOLD or jobs_verified < 10:
        # Untrusted or new: assign a pending job adjacent to verified results
        overlap_range = OVERLAP_STEPS_UNTRUSTED * _lambda_step_for_level(CURRENT_LEVEL)
        overlap_job = conn.execute(f"""
            SELECT j.id, j.lambda_val FROM jobs j
            WHERE j.status = 'pending' AND j.is_canary = 0
            AND j.id NOT IN ({exclude_sql})
            AND EXISTS (
                SELECT 1 FROM jobs jv
                WHERE jv.verified = 1
                AND ABS(jv.lambda_val - j.lambda_val) <= ?
                AND ABS(jv.lambda_val - j.lambda_val) > 0
            )
            ORDER BY RANDOM()
            LIMIT 1
        """, same_ip_workers + [overlap_range]).fetchone()
        if overlap_job:
            job = overlap_job
            print(f"  [Overlap] Untrusted worker {worker['name']} (trust={trust:.2f}, verified={jobs_verified}) → overlap job {job['id']}")

    # Work type filter clause
    type_clause = ""
    type_params = []
    if work_type:
        type_clause = " AND j.job_type = ?"
        type_params = [work_type]

    # Priority queue: check for agent-mode priority jobs before normal sweep
    if not job:
        job = conn.execute(f"""
            SELECT j.id, j.lambda_val FROM jobs j
            WHERE j.status = 'pending' AND j.priority > 0
            AND j.id NOT IN ({exclude_sql})
            {type_clause}
            ORDER BY j.priority DESC, j.id ASC
            LIMIT 1
        """, same_ip_workers + type_params).fetchone()

    # If no priority job, find a real pending job (normal sweep)
    if not job:
        job = conn.execute(f"""
            SELECT j.id, j.lambda_val FROM jobs j
            WHERE j.status = 'assigned' AND j.is_canary = 0
            AND j.quorum_received < j.quorum_target
            AND j.id NOT IN ({exclude_sql})
            {type_clause}
            ORDER BY j.quorum_received DESC, j.id ASC
            LIMIT 1
        """, same_ip_workers + type_params).fetchone()

    if not job:
        job = conn.execute(f"""
            SELECT j.id, j.lambda_val FROM jobs j
            WHERE j.status = 'pending' AND j.is_canary = 0
            AND j.priority = 0
            AND j.id NOT IN ({exclude_sql})
            {type_clause}
            ORDER BY j.id ASC
            LIMIT 1
        """, same_ip_workers + type_params).fetchone()

    if not job:
        # conn reused (shared connection)
        return {"status": "no_jobs", "message": "All jobs assigned or complete. Check back later."}

    now = time.time()
    deadline = now + JOB_DEADLINE

    # Create assignment
    conn.execute(
        "INSERT INTO assignments (job_id, worker_id, assigned_at, deadline) VALUES (?,?,?,?)",
        (job['id'], worker['id'], now, deadline)
    )

    # Update job status if it needs more quorum members
    conn.execute(
        "UPDATE jobs SET status = 'assigned' WHERE id = ? AND status = 'pending'",
        (job['id'],)
    )
    conn.commit()

    # Get full job row for params
    job_row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job['id'],)).fetchone()
    jtype = job_row['job_type'] if job_row else 'eigenvalue'

    # Get progress stats for client display
    stats = _get_progress_stats(conn)

    params = _params_for_job(dict(job_row)) if job_row else {"lambda": job['lambda_val']}

    # Preview potential W reward (flat rate — equal share economy)
    w_preview = W_BASE_RATE

    return {
        "status": "assigned",
        "job_id": job['id'],
        "job_type": jtype,
        "params": params,
        "deadline": deadline,
        "progress": stats,
        "w_reward_preview": w_preview,
    }

# ── Result Submission ──

def _check_hypothesis_resolution(hyp_id: int, conn):
    """Check if all test jobs for a hypothesis are done. If so, resolve it."""
    hyp = conn.execute("SELECT * FROM hypotheses WHERE id = ? AND status = 'pending'", (hyp_id,)).fetchone()
    if not hyp:
        return
    # Find all jobs linked to this hypothesis
    test_jobs = conn.execute(
        "SELECT j.id, j.status, j.quorum_received FROM jobs j WHERE j.source_hypothesis_id = ?",
        (hyp_id,)
    ).fetchall()
    if not test_jobs:
        return
    # Check if all are completed
    all_done = all(j['quorum_received'] > 0 for j in test_jobs)
    if not all_done:
        return
    # Check if any hits were found
    hits = []
    for j in test_jobs:
        discoveries = conn.execute(
            "SELECT constant_name FROM discoveries WHERE job_id = ?", (j['id'],)
        ).fetchall()
        hits.extend([d['constant_name'] for d in discoveries])
    status = 'confirmed' if hits else 'refuted'
    summary = f"{'|'.join(hits)}" if hits else "No hits found at any test lambda"
    conn.execute(
        "UPDATE hypotheses SET status = ?, result_summary = ?, resolved_at = ? WHERE id = ?",
        (status, summary, time.time(), hyp_id)
    )
    conn.commit()


class ResultSubmit(BaseModel):
    job_id: int
    eigenvalues: List[float]
    eigenvalues_hash: str
    found_constants: List[str] = []
    compute_seconds: float = 0.0

@app.post("/result")
def submit_result(result: ResultSubmit, request: Request, x_api_key: str = Header()):
    """Submit computation result with integrity hash."""
    ip = _get_client_ip(request)
    _check_rate(ip, "result")
    conn = get_db()
    if _is_banned(conn, ip):
        raise HTTPException(403, "Access denied")
    worker = verify_worker(x_api_key, conn)
    if not worker:
        raise HTTPException(401, "Invalid API key")
    _log_ip(conn, worker['id'], ip, "result")

    # Verify this worker was assigned this job
    assignment = conn.execute(
        "SELECT a.id, a.assigned_at, j.job_type FROM assignments a JOIN jobs j ON a.job_id = j.id WHERE a.job_id = ? AND a.worker_id = ? AND a.status = 'assigned'",
        (result.job_id, worker['id'])
    ).fetchone()

    if not assignment:
        raise HTTPException(400, "No active assignment for this job")

    # Server computes its own hash from raw eigenvalues (authoritative)
    # Don't reject on client hash mismatch — client may be older version with different precision
    computed_hash = hash_eigenvalues(result.eigenvalues)

    now = time.time()

    # ── Minimum Turnaround Gate ──────────────────────────────
    # Reject results that arrive impossibly fast — garbage/replay attack.
    # No maximum — slow hardware is fine. Only a floor.
    jtype_for_turnaround = assignment['job_type'] if assignment['job_type'] else 'eigenvalue'
    if not check_turnaround(jtype_for_turnaround, assignment['assigned_at'], now):
        elapsed = now - assignment['assigned_at']
        min_req = MIN_TURNAROUND.get(jtype_for_turnaround, 1.0)
        print(f"  [Turnaround] REJECTED: worker {worker['name']} job {result.job_id} ({jtype_for_turnaround}) in {elapsed:.2f}s < {min_req}s minimum")
        # Don't tell them why — just reject
        raise HTTPException(400, "Result rejected")

    # Eigenvalues sorted+rounded for neighbor comparison
    eigenvalues_sorted = [round(float(e), 10) for e in sorted(result.eigenvalues)]

    # ── Silent Discard (Kung Fu) ──────────────────────────────
    # Flagged workers get normal responses but results are silently dropped.
    # They keep crunching, paying electricity, never knowing they're quarantined.
    # The attack becomes the defense. ∂W=W.
    if worker.get('status') == 'flagged':
        # Complete the assignment so they get new work
        conn.execute(
            "UPDATE assignments SET status = 'completed', completed_at = ? WHERE id = ?",
            (now, assignment['id'])
        )
        conn.execute(
            "UPDATE workers SET jobs_completed = jobs_completed + 1, compute_seconds = compute_seconds + ? WHERE id = ?",
            (result.compute_seconds, worker['id'])
        )
        conn.commit()
        # conn reused (shared connection)
        print(f"  [KungFu] Silent discard: worker {worker['name']} job {result.job_id} (result dropped)")
        # Return normal-looking response — zero information leakage
        return {
            "status": "accepted",
            "verified": False,
            "discoveries": 0,
            "canary_check": None,
            "receipt": {"job_id": result.job_id, "timestamp": now, "signature": "ok"},
            "w_minted": 0.0,
        }

    # Store result (param_tier = job_type now, no mobile/desktop split)
    job_info = conn.execute("SELECT job_type, level FROM jobs WHERE id = ?", (result.job_id,)).fetchone()
    jtype = job_info['job_type'] if job_info else 'eigenvalue'

    conn.execute(
        "INSERT INTO results (job_id, worker_id, eigenvalues_hash, eigenvalues_json, found_constants, compute_seconds, submitted_at, param_tier) VALUES (?,?,?,?,?,?,?,?)",
        (result.job_id, worker['id'], computed_hash,
         json.dumps([round(float(e), 10) for e in sorted(result.eigenvalues)]),
         json.dumps(result.found_constants),
         result.compute_seconds, now, jtype)
    )

    # Update assignment
    conn.execute(
        "UPDATE assignments SET status = 'completed', completed_at = ? WHERE id = ?",
        (now, assignment['id'])
    )

    # Update job quorum count
    conn.execute(
        "UPDATE jobs SET quorum_received = quorum_received + 1 WHERE id = ?",
        (result.job_id,)
    )

    # Update worker stats
    conn.execute(
        "UPDATE workers SET jobs_completed = jobs_completed + 1, compute_seconds = compute_seconds + ? WHERE id = ?",
        (result.compute_seconds, worker['id'])
    )

    # Log discoveries
    job = conn.execute("SELECT lambda_val FROM jobs WHERE id = ?", (result.job_id,)).fetchone()
    for const_str in result.found_constants:
        name = const_str.split("(")[0].strip() if "(" in const_str else const_str
        ratio = 0.0
        if "ratio=" in const_str:
            try:
                ratio = float(const_str.split("ratio=")[1].rstrip(")"))
            except ValueError:
                pass
        conn.execute(
            "INSERT INTO discoveries (job_id, lambda_val, constant_name, ratio_value, discovered_at, worker_id, param_tier) VALUES (?,?,?,?,?,?,?)",
            (result.job_id, job['lambda_val'], name, ratio, now, worker['id'], jtype)
        )
        conn.execute(
            "UPDATE workers SET discoveries = discoveries + 1 WHERE id = ?",
            (worker['id'],)
        )

    conn.commit()

    # ── Staggered Overlap Verification ──
    # Verify by checking eigenvalue continuity with neighboring lambdas.
    # Also retroactively verify nearby unverified jobs (cascade effect).
    verified_result = validate_by_neighbors(result.job_id, worker['id'], eigenvalues_sorted, conn)
    verified = verified_result is True
    if verified:
        print(f"  [Overlap] Job {result.job_id} VERIFIED via neighbors (worker={worker['name']})")
    retro = retroactive_verify_neighbors(result.job_id, eigenvalues_sorted, conn)
    if retro > 0:
        print(f"  [Overlap] Cascade: {retro} neighbor jobs also verified")

    # ── Mint W (Equal Share Economy) ──
    # Flat rate for ALL job types. No per-type rewards. Equal for all workers.
    # Production mode: only on neighbor-verified results (full chunk-mixing security)
    # Testing mode: mint on acceptance (for internal testing with <3 workers)
    w_minted = 0.0
    should_mint = verified if REQUIRE_VERIFICATION_TO_MINT else True
    if should_mint:
        job_info = conn.execute("SELECT job_type FROM jobs WHERE id = ?", (result.job_id,)).fetchone()
        jtype = job_info['job_type'] if job_info else 'eigenvalue'
        reward = W_BASE_RATE  # flat rate — equal share
        memo_tag = "neighbor-verified" if verified else "accepted"
        mint_w(conn, worker['id'], reward, tx_type="mint",
               memo=f"job#{result.job_id} {jtype} {memo_tag}",
               receipt_hash=computed_hash)
        w_minted = reward
        conn.commit()

    # Mark job completed if enough results
    job_row = conn.execute("SELECT quorum_received, quorum_target FROM jobs WHERE id = ?", (result.job_id,)).fetchone()
    if job_row['quorum_received'] >= job_row['quorum_target'] and not verified:
        conn.execute("UPDATE jobs SET status = 'completed' WHERE id = ?", (result.job_id,))
        conn.commit()

    # Check if this result resolves a hypothesis
    hyp_link = conn.execute(
        "SELECT source_hypothesis_id FROM jobs WHERE id = ? AND source_hypothesis_id IS NOT NULL",
        (result.job_id,)
    ).fetchone()
    if hyp_link and hyp_link['source_hypothesis_id']:
        _check_hypothesis_resolution(hyp_link['source_hypothesis_id'], conn)

    # Check if eigenvalue level is complete → auto-advance
    if jtype == 'eigenvalue':
        check_level_complete_and_advance()

    # Build signed receipt
    receipt = make_receipt(
        job_id=result.job_id,
        worker_name=worker['name'],
        lambda_val=job['lambda_val'],
        eigenvalues_hash=computed_hash,
        found_constants=result.found_constants,
        compute_seconds=result.compute_seconds,
        timestamp=now,
    )

    # conn reused (shared connection)

    return {
        "status": "accepted",
        "verified": verified,
        "discoveries": len(result.found_constants),
        "canary_check": None,  # deprecated — neighbor verification replaces canaries
        "receipt": receipt,
        "w_minted": w_minted,
    }

# ── Verify Receipt ──

@app.post("/verify-receipt")
def verify_receipt(receipt: dict):
    """Verify a signed computation receipt. Anyone can call this."""
    sig = receipt.pop("signature", None)
    if not sig:
        raise HTTPException(400, "No signature in receipt")
    expected = sign_receipt(receipt)
    valid = hmac.compare_digest(sig, expected)
    receipt["signature"] = sig  # restore it
    return {"valid": valid, "receipt": receipt}

# ── Heartbeat ──

@app.post("/heartbeat")
def heartbeat(x_api_key: str = Header()):
    conn = get_db()
    worker = verify_worker(x_api_key, conn)
    # conn reused (shared connection)
    if not worker:
        raise HTTPException(401, "Invalid API key")
    return {"status": "alive", "worker_id": worker['id']}

# ── Progress & Stats ──

def _get_progress_stats(conn) -> dict:
    total = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    completed = conn.execute("SELECT COUNT(*) FROM jobs WHERE quorum_received > 0").fetchone()[0]
    verified = conn.execute("SELECT COUNT(*) FROM jobs WHERE verified = 1").fetchone()[0]
    pending = conn.execute("SELECT COUNT(*) FROM jobs WHERE status = 'pending'").fetchone()[0]
    assigned = conn.execute("SELECT COUNT(*) FROM jobs WHERE status = 'assigned'").fetchone()[0]
    disputed = conn.execute("SELECT COUNT(*) FROM jobs WHERE status = 'disputed'").fetchone()[0]

    # Count workers with active assignments (more reliable than heartbeat — browser clients don't heartbeat)
    active_workers = conn.execute(
        "SELECT COUNT(DISTINCT worker_id) FROM assignments WHERE status = 'assigned' AND deadline > ?",
        (time.time(),)
    ).fetchone()[0]

    total_compute = conn.execute("SELECT COALESCE(SUM(compute_seconds), 0) FROM workers").fetchone()[0]
    total_discoveries = conn.execute("SELECT COUNT(*) FROM discoveries").fetchone()[0]
    total_confirmed = conn.execute(
        "SELECT COUNT(*) FROM discoveries d JOIN jobs j ON d.job_id = j.id WHERE j.verified = 1"
    ).fetchone()[0]

    # Current lambda range being worked
    current_lambda = conn.execute(
        "SELECT lambda_val FROM jobs WHERE status = 'assigned' ORDER BY lambda_val ASC LIMIT 1"
    ).fetchone()
    current_lam = current_lambda['lambda_val'] if current_lambda else LAMBDA_START

    # Jobs per hour (last hour)
    hour_ago = time.time() - 3600
    recent_completions = conn.execute(
        "SELECT COUNT(*) FROM assignments WHERE status = 'completed' AND completed_at > ?",
        (hour_ago,)
    ).fetchone()[0]

    pct = (completed / total * 100) if total > 0 else 0
    eta_hours = ((total - completed) / recent_completions) if recent_completions > 0 else float('inf')

    # Agent mode stats (safe — tables exist after init_db)
    try:
        total_hints = conn.execute("SELECT COUNT(*) FROM hints").fetchone()[0]
        total_hypotheses = conn.execute("SELECT COUNT(*) FROM hypotheses").fetchone()[0]
        hypotheses_confirmed = conn.execute("SELECT COUNT(*) FROM hypotheses WHERE status = 'confirmed'").fetchone()[0]
        hypotheses_refuted = conn.execute("SELECT COUNT(*) FROM hypotheses WHERE status = 'refuted'").fetchone()[0]
        priority_jobs = conn.execute("SELECT COUNT(*) FROM jobs WHERE priority > 0").fetchone()[0]
        active_agents = conn.execute(
            "SELECT COUNT(*) FROM workers WHERE worker_type = 'agent' AND last_heartbeat > ? AND status = 'active'",
            (time.time() - 300,)
        ).fetchone()[0]
    except sqlite3.OperationalError:
        total_hints = total_hypotheses = hypotheses_confirmed = hypotheses_refuted = priority_jobs = active_agents = 0

    # Per-type stats
    per_type = {}
    for wtype in list(JOB_PRIORITY.keys()):
        t_total = conn.execute("SELECT COUNT(*) FROM jobs WHERE job_type = ?", (wtype,)).fetchone()[0]
        t_done = conn.execute("SELECT COUNT(*) FROM jobs WHERE job_type = ? AND quorum_received > 0", (wtype,)).fetchone()[0]
        t_compute = conn.execute("""
            SELECT COALESCE(SUM(r.compute_seconds), 0) FROM results r
            JOIN jobs j ON r.job_id = j.id WHERE j.job_type = ?
        """, (wtype,)).fetchone()[0]
        per_type[wtype] = {
            "total": t_total, "completed": t_done,
            "pct": round(t_done / max(t_total, 1) * 100, 2),
            "compute_hours": round(t_compute / 3600, 2),
        }

    return {
        "total_jobs": total,
        "completed": completed,
        "verified": verified,
        "pending": pending,
        "assigned": assigned,
        "disputed": disputed,
        "percent_complete": round(pct, 4),
        "active_workers": active_workers,
        "active_agents": active_agents,
        "total_compute_hours": round(total_compute / 3600, 2),
        "total_discoveries": total_discoveries,
        "total_confirmed": total_confirmed,
        "per_type": per_type,
        "total_hints": total_hints,
        "total_hypotheses": total_hypotheses,
        "hypotheses_confirmed": hypotheses_confirmed,
        "hypotheses_refuted": hypotheses_refuted,
        "priority_jobs": priority_jobs,
        "current_lambda": round(current_lam, 6),
        "jobs_per_hour": recent_completions,
        "eta_hours": round(eta_hours, 1) if eta_hours != float('inf') else None,
        "lambda_range": [LAMBDA_START, LAMBDA_END],
        "lambda_step": _lambda_step_for_level(CURRENT_LEVEL),
        "current_level": CURRENT_LEVEL,
    }

@app.get("/progress")
def get_progress():
    conn = get_db()
    stats = _get_progress_stats(conn)
    # conn reused (shared connection)
    return stats

@app.get("/progress/heatmap")
def progress_heatmap(blocks: int = 20):
    """Return per-block completion for Menger sponge visualization."""
    blocks = min(max(blocks, 8), 160000)
    conn = get_db()
    total_jobs = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    block_size = max(total_jobs, 1) / blocks

    # Single-query approach: get all completed/assigned job counts by block
    done_counts = [0] * blocks
    active_counts = [0] * blocks
    disc_counts = [0] * blocks

    # Completed jobs
    for row in conn.execute(
        "SELECT id, status FROM jobs WHERE status IN ('verified','completed','assigned')"
    ).fetchall():
        b = min(int(row['id'] / block_size), blocks - 1)
        if row['status'] in ('verified', 'completed'):
            done_counts[b] += 1
        else:
            active_counts[b] += 1

    # Discoveries
    for row in conn.execute("SELECT job_id FROM discoveries").fetchall():
        b = min(int(row['job_id'] / block_size), blocks - 1)
        disc_counts[b] += 1

    # conn reused (shared connection)

    result = []
    for i in range(blocks):
        total = int((i + 1) * block_size) - int(i * block_size)
        result.append({
            "block": i,
            "done": done_counts[i],
            "active": active_counts[i],
            "total": total,
            "pct": round(done_counts[i] / max(total, 1) * 100, 1),
            "discoveries": disc_counts[i],
        })
    return result

@app.get("/workers")
def list_workers():
    conn = get_db()
    rows = conn.execute("""
        SELECT id, name, jobs_completed, compute_seconds, discoveries,
               last_heartbeat, gpu_info, status
        FROM workers ORDER BY jobs_completed DESC
    """).fetchall()
    # conn reused (shared connection)
    now = time.time()
    return [
        {**dict(r), "online": (now - r['last_heartbeat']) < 300}
        for r in rows
    ]

@app.get("/discoveries")
def list_discoveries():
    conn = get_db()
    rows = conn.execute("""
        SELECT d.*, w.name as worker_name, j.verified as job_verified
        FROM discoveries d
        LEFT JOIN workers w ON d.worker_id = w.id
        LEFT JOIN jobs j ON d.job_id = j.id
        ORDER BY d.discovered_at DESC
        LIMIT 100
    """).fetchall()
    # conn reused (shared connection)
    return [dict(r) for r in rows]

@app.get("/my/discoveries")
def my_discoveries(x_api_key: str = Header()):
    """Get discoveries for the authenticated worker."""
    conn = get_db()
    worker = verify_worker(x_api_key, conn)
    if not worker:
        raise HTTPException(401, "Invalid API key")
    rows = conn.execute("""
        SELECT d.id, d.job_id, d.lambda_val, d.constant_name, d.ratio_value,
               d.discovered_at, d.verified, d.param_tier
        FROM discoveries d
        WHERE d.worker_id = ?
        ORDER BY d.discovered_at DESC
    """, (worker['id'],)).fetchall()
    # conn reused (shared connection)
    return {
        "worker": worker['name'] or worker['id'][:8],
        "total": len(rows),
        "discoveries": [dict(r) for r in rows],
    }

@app.get("/my/stats")
def my_stats(x_api_key: str = Header()):
    """Get full stats for the authenticated worker."""
    conn = get_db()
    worker = verify_worker(x_api_key, conn)
    if not worker:
        raise HTTPException(401, "Invalid API key")
    # Count discoveries
    disc_count = conn.execute(
        "SELECT COUNT(*) FROM discoveries WHERE worker_id = ?", (worker['id'],)
    ).fetchone()[0]
    # Count verified discoveries
    verified_count = conn.execute(
        "SELECT COUNT(*) FROM discoveries WHERE worker_id = ? AND verified = 1", (worker['id'],)
    ).fetchone()[0]
    # Recent results
    recent = conn.execute("""
        SELECT r.job_id, r.found_constants, r.compute_seconds, r.submitted_at,
               r.n_vertices, r.param_tier
        FROM results r WHERE r.worker_id = ?
        ORDER BY r.submitted_at DESC LIMIT 20
    """, (worker['id'],)).fetchall()
    # W wallet info
    _ensure_wallet(conn, worker['id'])
    wallet = conn.execute("SELECT balance, total_earned FROM wallets WHERE worker_id = ?", (worker['id'],)).fetchone()
    # conn reused (shared connection)
    return {
        "worker": worker['name'] or worker['id'][:8],
        "jobs_completed": worker['jobs_completed'],
        "compute_hours": round(worker['compute_seconds'] / 3600, 2),
        "discoveries": disc_count,
        "verified_discoveries": verified_count,
        "trust_score": round(worker['trust_score'], 2),
        "w_balance": round(wallet['balance'], 4) if wallet else 0,
        "w_earned": round(wallet['total_earned'], 4) if wallet else 0,
        "recent_jobs": [dict(r) for r in recent],
    }

# ── Leaderboard ──

@app.get("/leaderboard")
def leaderboard():
    conn = get_db()
    rows = conn.execute("""
        SELECT wk.id, wk.name, wk.jobs_completed, wk.compute_seconds, wk.discoveries,
               wk.last_heartbeat, wk.gpu_info, wk.trust_score, wk.canaries_passed, wk.canaries_failed,
               wk.status,
               COALESCE(w.balance, 0) as w_balance
        FROM workers wk
        LEFT JOIN wallets w ON wk.id = w.worker_id
        WHERE wk.status NOT IN ('banned', 'flagged')
        ORDER BY wk.jobs_completed DESC, wk.registered_at ASC
        LIMIT 50
    """).fetchall()
    # Get set of workers with active assignments (more reliable than heartbeat)
    active_ids = {r['worker_id'] for r in conn.execute(
        "SELECT DISTINCT worker_id FROM assignments WHERE status = 'assigned' AND deadline > ?",
        (time.time(),)
    ).fetchall()}
    # conn reused (shared connection)
    now = time.time()
    return [{
        "rank": i + 1,
        "name": r['name'] or r['id'][:8],
        "jobs": r['jobs_completed'],
        "compute_hours": round(r['compute_seconds'] / 3600, 2),
        "discoveries": r['discoveries'],
        "online": r['id'] in active_ids or (now - r['last_heartbeat']) < 300,
        "gpu": r['gpu_info'],
        "trust": round(r['trust_score'], 2),
        "w_balance": round(r['w_balance'], 2),
    } for i, r in enumerate(rows)]

# ── Active Jobs ──

@app.get("/active")
def active_jobs():
    conn = get_db()
    rows = conn.execute("""
        SELECT a.job_id, j.lambda_val, a.assigned_at, a.deadline,
               w.name as worker_name, w.gpu_info
        FROM assignments a
        JOIN jobs j ON a.job_id = j.id
        JOIN workers w ON a.worker_id = w.id
        WHERE a.status = 'assigned'
        ORDER BY a.assigned_at DESC
        LIMIT 20
    """).fetchall()
    # conn reused (shared connection)
    now = time.time()
    return [{
        "job_id": r['job_id'],
        "lambda": r['lambda_val'],
        "worker": r['worker_name'] or "anonymous",
        "gpu": r['gpu_info'] or "CPU",
        "elapsed": round(now - r['assigned_at']),
        "deadline_remaining": round(r['deadline'] - now),
    } for r in rows]

# ── Chat ──

chat_connections: dict = {}  # WebSocket -> username mapping

@app.get("/chat/history")
def chat_history(limit: int = 50):
    """Get recent chat messages."""
    conn = get_db()
    rows = conn.execute(
        "SELECT username, content, sent_at, msg_type FROM messages ORDER BY sent_at DESC LIMIT ?",
        (limit,)
    ).fetchall()
    # conn reused (shared connection)
    return [{"username": r['username'], "content": r['content'],
             "time": r['sent_at'], "type": r['msg_type']} for r in reversed(rows)]

@app.post("/chat/send")
def chat_send(x_api_key: str = Header(), message: dict = {}):
    """Send a chat message (authenticated workers)."""
    conn = get_db()
    worker = verify_worker(x_api_key, conn)
    if not worker:
        raise HTTPException(401, "Invalid API key")
    content = str(message.get('content', '')).strip()[:500]  # 500 char limit
    if not content:
        raise HTTPException(400, "Empty message")
    now = time.time()
    conn.execute("INSERT INTO messages (username, content, sent_at) VALUES (?,?,?)",
                 (worker['name'], content, now))
    conn.commit()
    # conn reused (shared connection)
    # Broadcast to WebSocket clients
    msg = json.dumps({"username": worker['name'], "content": content, "time": now, "type": "chat"})
    for ws in list(chat_connections.keys()):
        try:
            import asyncio
            asyncio.create_task(ws.send_text(msg))
        except Exception:
            chat_connections.pop(ws, None)
    return {"status": "sent"}

@app.get("/chat/online")
def chat_online():
    """Who's in chat and who's computing."""
    chat_users = sorted(set(v for v in chat_connections.values() if v != "anonymous"))
    conn = get_db()
    now = time.time()
    # Workers active in last 5 minutes
    rows = conn.execute(
        "SELECT wk.name, COALESCE(w.balance, 0) as w_balance FROM workers wk LEFT JOIN wallets w ON wk.id = w.worker_id WHERE wk.last_heartbeat > ?", (now - 300,)
    ).fetchall()
    # conn reused (shared connection)
    computing = [{"name": r['name'], "w_balance": round(r['w_balance'], 2)} for r in rows]
    return {"chat": chat_users, "computing": computing}

async def broadcast_presence():
    """Send updated user list to all connected clients."""
    chat_users = sorted(set(chat_connections.values()))
    msg = json.dumps({"type": "presence", "chat_users": chat_users})
    for ws in list(chat_connections.keys()):
        try:
            await ws.send_text(msg)
        except Exception:
            chat_connections.pop(ws, None)

@app.websocket("/chat/ws")
async def chat_ws(websocket: WebSocket):
    """WebSocket for live chat updates."""
    await websocket.accept()
    chat_connections[websocket] = "anonymous"
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                api_key = msg.get('api_key', '')
                content = str(msg.get('content', '')).strip()[:500]
                if api_key:
                    conn = get_db()
                    worker = verify_worker(api_key, conn)
                    if worker:
                        # Track username for presence
                        if chat_connections.get(websocket) != worker['name']:
                            chat_connections[websocket] = worker['name']
                            await broadcast_presence()
                        if content:
                            now = time.time()
                            conn.execute("INSERT INTO messages (username, content, sent_at) VALUES (?,?,?)",
                                         (worker['name'], content, now))
                            conn.commit()
                            broadcast = json.dumps({"username": worker['name'], "content": content, "time": now, "type": "chat"})
                            for ws in list(chat_connections.keys()):
                                try:
                                    await ws.send_text(broadcast)
                                except Exception:
                                    chat_connections.pop(ws, None)
                    # conn reused (shared connection)
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        chat_connections.pop(websocket, None)
        await broadcast_presence()

# ── Status (public) ──

@app.get("/api/status")
def status():
    conn = get_db()
    stats = _get_progress_stats(conn)
    # conn reused (shared connection)
    return {
        "name": "W@Home Hive — Akataleptos Distributed Spectral Search",
        "version": "2.0",
        "status": "online",
        **stats
    }

@app.get("/api/falsification")
def api_falsification():
    """Run exhaustive threshold scan and return results as JSON."""
    from itertools import product as iterproduct
    results = []
    for b in range(3, 22, 2):  # odd bases 3-21
        d = 3
        n_total = b ** d
        for P in range(1, d + 1):
            r = 0
            for coords in iterproduct(range(b), repeat=d):
                center_count = sum(1 for c in coords if c == (b-1)//2)
                if center_count >= P:
                    r += 1
            k = n_total - r
            S = b + P
            Delta = S**2 - 4*P
            preds = eval_formulas(b, d, S, P, r, k)
            n_match = 0
            matched = []
            for name, pred in preds.items():
                target = TARGETS.get(name)
                if target and abs(pred - target) / target < MATCH_TOL:
                    n_match += 1
                    matched.append({"name": name, "predicted": round(pred, 6), "target": target,
                                     "error_pct": round(abs(pred - target) / target * 100, 4)})
            results.append({
                "b": b, "P": P, "S": S, "Delta": Delta, "k": k, "r": r,
                "n_matched": n_match, "matches": matched,
                "is_menger": (b == 3 and P == 2),
            })
    return {"results": results, "total_configs": len(results),
            "max_match": max(r["n_matched"] for r in results),
            "menger_match": next(r["n_matched"] for r in results if r["is_menger"])}

@app.get("/api/clock")
def api_clock():
    """Spectral decimation convergence data — the Menger Countdown Clock.

    If physical constants come from Menger spectral invariants, residuals between
    predicted and measured values decrease geometrically with iteration depth.
    The convergence ratio P/k = 2/20 = 0.1 means each level refines by 10x.
    """
    # Known spectral dimensions at each level (computed)
    d_S_levels = [1.32, 1.88, 2.14]  # L1, L2, L3
    d_H = np.log(20) / np.log(3)     # ~2.727 = Hausdorff dimension (limit)

    # Spectral gap convergence
    gap_ratio_levels = [0.333, 0.143, 0.103]  # approaching P/k = 0.1

    # Intra-gap convergence (L1-L3)
    intra_gap_levels = [0.16, 0.0016, 1.6e-5]  # approaching P^4/r

    # Predicted values at L4 (extrapolated via P/k geometric correction)
    d_S_L4_pred = d_H - (d_H - d_S_levels[-1]) * 0.1  # ~2.67
    gap_ratio_L4_pred = 0.1 + (gap_ratio_levels[-1] - 0.1) * 0.1

    # Eigenvalue-2 multiplicity tower: m(n) = (18^n + 153*4^n + 1155) / 357
    mult_tower = []
    for n in range(1, 6):
        m = (18**n + 153 * 4**n + 1155) / 357
        mult_tower.append({"level": n, "multiplicity": int(round(m))})
    # Known: [5, 11, 47, 407, 5735]

    # Convergence rates
    P_over_k = 2 / 20  # = 0.1 — the geometric correction ratio

    # Per-level prediction quality (relative error for fine structure constant)
    alpha_residuals = [
        {"level": 1, "residual_pct": 12.4, "note": "L1: raw algebraic formula"},
        {"level": 2, "residual_pct": 1.2,  "note": "L2: first spectral decimation correction"},
        {"level": 3, "residual_pct": 0.12, "note": "L3: second correction"},
        {"level": 4, "residual_pct": 0.012, "note": "L4: predicted (P/k geometric decay)"},
    ]

    return {
        "hausdorff_dim": round(d_H, 6),
        "spectral_dim_convergence": [
            {"level": i+1, "d_S": d_S_levels[i], "residual": round(d_H - d_S_levels[i], 4)}
            for i in range(len(d_S_levels))
        ] + [{"level": 4, "d_S": round(d_S_L4_pred, 4), "residual": round(d_H - d_S_L4_pred, 4), "predicted": True}],
        "gap_ratio_convergence": [
            {"level": i+1, "ratio": gap_ratio_levels[i]}
            for i in range(len(gap_ratio_levels))
        ] + [{"level": 4, "ratio": round(gap_ratio_L4_pred, 6), "predicted": True}],
        "intra_gap_convergence": [
            {"level": i+1, "value": intra_gap_levels[i]}
            for i in range(len(intra_gap_levels))
        ],
        "multiplicity_tower": mult_tower,
        "alpha_residuals": alpha_residuals,
        "convergence_ratio": P_over_k,
        "target_ratio": "P/k = 2/20 = 0.1",
        "conclusion": "Each Menger level refines predictions by ~10x. L5 computation would resolve fine structure to 0.001%.",
    }

@app.get("/api/work-types")
def api_work_types():
    """Return available work types and per-type stats."""
    conn = get_db()

    types_info = {}
    for wtype in list(JOB_PRIORITY.keys()):
        total = conn.execute("SELECT COUNT(*) FROM jobs WHERE job_type = ?", (wtype,)).fetchone()[0]
        completed = conn.execute("SELECT COUNT(*) FROM jobs WHERE job_type = ? AND quorum_received > 0", (wtype,)).fetchone()[0]
        verified = conn.execute("SELECT COUNT(*) FROM jobs WHERE job_type = ? AND verified = 1", (wtype,)).fetchone()[0]
        compute_secs = conn.execute("""
            SELECT COALESCE(SUM(r.compute_seconds), 0) FROM results r
            JOIN jobs j ON r.job_id = j.id WHERE j.job_type = ?
        """, (wtype,)).fetchone()[0]
        discoveries = conn.execute("""
            SELECT COUNT(*) FROM discoveries d
            JOIN jobs j ON d.job_id = j.id WHERE j.job_type = ?
        """, (wtype,)).fetchone()[0]

        types_info[wtype] = {
            "total_jobs": total,
            "completed": completed,
            "verified": verified,
            "compute_hours": round(compute_secs / 3600, 2),
            "discoveries": discoveries,
            "percent_complete": round(completed / max(total, 1) * 100, 2),
        }

    # conn reused (shared connection)

    return {
        "types": types_info,
        "descriptions": {
            "eigenvalue": "Lambda sweep — eigenvalue search for physical constants in W-operator spectra",
            "falsification": "Fractal falsification — test random 3D fractals against the 13 Menger predictions",
            "clock": "Countdown clock — spectral decimation convergence across Menger iteration levels",
        },
        "themes": {
            "eigenvalue": {"primary": "#a78bfa", "secondary": "#6dd5ed", "label": "Eigenvalue Search"},
            "falsification": {"primary": "#22c55e", "secondary": "#ffd06a", "label": "Fractal Falsification"},
            "clock": {"primary": "#6dd5ed", "secondary": "#e0e0f0", "label": "Countdown Clock"},
        },
    }

# ═══════════════════════════════════════════════════════════
# W Economy Endpoints
# ═══════════════════════════════════════════════════════════

class TransferRequest(BaseModel):
    to_name: str
    amount: float
    memo: str = ""

class StakeRequest(BaseModel):
    job_type: str
    amount: float

class UnstakeRequest(BaseModel):
    stake_id: int

@app.get("/wallet")
def get_wallet(x_api_key: str = Header()):
    """Get authenticated worker's W wallet."""
    conn = get_db()
    worker = verify_worker(x_api_key, conn)
    if not worker:
        raise HTTPException(401, "Invalid API key")
    _ensure_wallet(conn, worker['id'])
    wallet = conn.execute("SELECT * FROM wallets WHERE worker_id = ?", (worker['id'],)).fetchone()
    # Recent transactions
    txns = conn.execute("""
        SELECT id, from_id, to_id, amount, tx_type, memo, timestamp, block_height
        FROM transactions WHERE from_id = ? OR to_id = ?
        ORDER BY timestamp DESC LIMIT 50
    """, (worker['id'], worker['id'])).fetchall()
    # Active stakes
    stakes = conn.execute(
        "SELECT id, job_type, amount, staked_at FROM stakes WHERE worker_id = ?",
        (worker['id'],)
    ).fetchall()
    # conn reused (shared connection)
    return {
        "worker_id": worker['id'],
        "name": worker['name'],
        "balance": round(wallet['balance'], 4) if wallet else 0,
        "total_earned": round(wallet['total_earned'], 4) if wallet else 0,
        "total_sent": round(wallet['total_sent'], 4) if wallet else 0,
        "total_received": round(wallet['total_received'], 4) if wallet else 0,
        "transactions": [dict(t) for t in txns],
        "stakes": [dict(s) for s in stakes],
    }

@app.get("/wallet/{name}")
def get_wallet_public(name: str):
    """Public view of a worker's W balance (no transaction history)."""
    conn = get_db()
    worker = conn.execute("SELECT id FROM workers WHERE name = ?", (name,)).fetchone()
    if not worker:
        raise HTTPException(404, "Worker not found")
    wallet = conn.execute("SELECT balance, total_earned FROM wallets WHERE worker_id = ?", (worker['id'],)).fetchone()
    # conn reused (shared connection)
    return {
        "name": name,
        "balance": round(wallet['balance'], 4) if wallet else 0,
        "total_earned": round(wallet['total_earned'], 4) if wallet else 0,
    }

@app.post("/transfer")
def transfer_w(req: TransferRequest, request: Request, x_api_key: str = Header()):
    """Transfer W tokens to another worker."""
    ip = _get_client_ip(request)
    _check_rate(ip, "transfer")
    if req.amount <= 0:
        raise HTTPException(400, "Amount must be positive")
    conn = get_db()
    worker = verify_worker(x_api_key, conn)
    if not worker:
        raise HTTPException(401, "Invalid API key")
    # Find recipient
    recipient = conn.execute("SELECT id FROM workers WHERE name = ?", (req.to_name,)).fetchone()
    if not recipient:
        raise HTTPException(404, f"Recipient '{req.to_name}' not found")
    if recipient['id'] == worker['id']:
        raise HTTPException(400, "Cannot transfer to yourself")
    # Check balance
    _ensure_wallet(conn, worker['id'])
    wallet = conn.execute("SELECT balance FROM wallets WHERE worker_id = ?", (worker['id'],)).fetchone()
    if wallet['balance'] < req.amount:
        raise HTTPException(400, f"Insufficient balance ({wallet['balance']:.4f} W)")
    # Execute transfer
    _ensure_wallet(conn, recipient['id'])
    now = time.time()
    conn.execute("UPDATE wallets SET balance = balance - ?, total_sent = total_sent + ? WHERE worker_id = ?",
                 (req.amount, req.amount, worker['id']))
    conn.execute("UPDATE wallets SET balance = balance + ?, total_received = total_received + ? WHERE worker_id = ?",
                 (req.amount, req.amount, recipient['id']))
    conn.execute(
        "INSERT INTO transactions (from_id, to_id, amount, tx_type, memo, timestamp) VALUES (?,?,?,?,?,?)",
        (worker['id'], recipient['id'], req.amount, "transfer", req.memo[:200], now)
    )
    conn.commit()
    new_bal = conn.execute("SELECT balance FROM wallets WHERE worker_id = ?", (worker['id'],)).fetchone()['balance']
    # conn reused (shared connection)
    return {"status": "transferred", "amount": req.amount, "to": req.to_name, "new_balance": round(new_bal, 4)}

@app.post("/stake")
def stake_w(req: StakeRequest, request: Request, x_api_key: str = Header()):
    """Stake W tokens on a job type to boost priority."""
    ip = _get_client_ip(request)
    _check_rate(ip, "stake")
    if req.amount <= 0:
        raise HTTPException(400, "Amount must be positive")
    if req.job_type not in W_REWARDS:
        raise HTTPException(400, f"Invalid job type. Choose from: {list(W_REWARDS.keys())}")
    conn = get_db()
    worker = verify_worker(x_api_key, conn)
    if not worker:
        raise HTTPException(401, "Invalid API key")
    _ensure_wallet(conn, worker['id'])
    wallet = conn.execute("SELECT balance FROM wallets WHERE worker_id = ?", (worker['id'],)).fetchone()
    if wallet['balance'] < req.amount:
        raise HTTPException(400, f"Insufficient balance ({wallet['balance']:.4f} W)")
    now = time.time()
    conn.execute("UPDATE wallets SET balance = balance - ? WHERE worker_id = ?", (req.amount, worker['id']))
    conn.execute("INSERT INTO stakes (worker_id, job_type, amount, staked_at) VALUES (?,?,?,?)",
                 (worker['id'], req.job_type, req.amount, now))
    conn.execute(
        "INSERT INTO transactions (from_id, to_id, amount, tx_type, memo, timestamp) VALUES (?,?,?,?,?,?)",
        (worker['id'], "stake_pool", req.amount, "stake", f"staked on {req.job_type}", now)
    )
    conn.commit()
    new_bal = conn.execute("SELECT balance FROM wallets WHERE worker_id = ?", (worker['id'],)).fetchone()['balance']
    # conn reused (shared connection)
    return {"status": "staked", "amount": req.amount, "job_type": req.job_type, "new_balance": round(new_bal, 4)}

@app.post("/unstake")
def unstake_w(req: UnstakeRequest, request: Request, x_api_key: str = Header()):
    """Unstake W tokens (must wait lockout period)."""
    ip = _get_client_ip(request)
    _check_rate(ip, "stake")
    conn = get_db()
    worker = verify_worker(x_api_key, conn)
    if not worker:
        raise HTTPException(401, "Invalid API key")
    stake = conn.execute("SELECT * FROM stakes WHERE id = ? AND worker_id = ?", (req.stake_id, worker['id'])).fetchone()
    if not stake:
        raise HTTPException(404, "Stake not found")
    now = time.time()
    if now - stake['staked_at'] < W_STAKE_LOCKOUT:
        remaining = int(W_STAKE_LOCKOUT - (now - stake['staked_at']))
        raise HTTPException(400, f"Lockout: {remaining}s remaining (24h minimum)")
    # Return staked amount
    conn.execute("UPDATE wallets SET balance = balance + ? WHERE worker_id = ?", (stake['amount'], worker['id']))
    conn.execute("DELETE FROM stakes WHERE id = ?", (stake['id'],))
    conn.execute(
        "INSERT INTO transactions (from_id, to_id, amount, tx_type, memo, timestamp) VALUES (?,?,?,?,?,?)",
        ("stake_pool", worker['id'], stake['amount'], "unstake", f"unstaked from {stake['job_type']}", now)
    )
    conn.commit()
    new_bal = conn.execute("SELECT balance FROM wallets WHERE worker_id = ?", (worker['id'],)).fetchone()['balance']
    # conn reused (shared connection)
    return {"status": "unstaked", "amount": stake['amount'], "new_balance": round(new_bal, 4)}

@app.get("/api/economy")
def economy_stats():
    """Public W economy overview."""
    conn = get_db()
    # Total supply
    total_minted = conn.execute("SELECT COALESCE(SUM(amount), 0) FROM transactions WHERE tx_type = 'mint'").fetchone()[0]
    total_staked = conn.execute("SELECT COALESCE(SUM(amount), 0) FROM stakes").fetchone()[0]
    total_transferred = conn.execute("SELECT COALESCE(SUM(amount), 0) FROM transactions WHERE tx_type = 'transfer'").fetchone()[0]
    n_wallets = conn.execute("SELECT COUNT(*) FROM wallets WHERE total_earned > 0").fetchone()[0]
    n_txns = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    # Top holders
    top = conn.execute("""
        SELECT w.worker_id, wk.name, w.balance, w.total_earned
        FROM wallets w JOIN workers wk ON w.worker_id = wk.id
        WHERE w.total_earned > 0
        ORDER BY w.balance DESC LIMIT 20
    """).fetchall()
    # Staking breakdown
    stake_breakdown = {}
    for jt in W_REWARDS:
        s = conn.execute("SELECT COALESCE(SUM(amount), 0) FROM stakes WHERE job_type = ?", (jt,)).fetchone()[0]
        stake_breakdown[jt] = round(s, 4)
    # Recent transactions
    recent_txns = conn.execute("""
        SELECT t.id, t.from_id, t.to_id, t.amount, t.tx_type, t.memo, t.timestamp, t.block_height,
               COALESCE(wf.name, t.from_id) as from_name,
               COALESCE(wt.name, t.to_id) as to_name
        FROM transactions t
        LEFT JOIN workers wf ON t.from_id = wf.id
        LEFT JOIN workers wt ON t.to_id = wt.id
        ORDER BY t.timestamp DESC LIMIT 50
    """).fetchall()
    # Chain stats
    tip = conn.execute("SELECT height, block_hash, timestamp FROM blocks ORDER BY height DESC LIMIT 1").fetchone()
    # Energy estimation
    # Watts per device type (compute load estimate)
    WATTS = {"WebAssembly (Browser)": 15, "numpy dense": 45, "CPU": 45,
             "CUDA (CuPy)": 150, "RTX 4070 Laptop 8GB": 80}
    DEFAULT_WATTS = 45
    total_kwh = 0.0
    workers_energy = conn.execute("SELECT gpu_info, compute_seconds FROM workers WHERE compute_seconds > 0").fetchall()
    for w in workers_energy:
        gpu = w['gpu_info'] or ''
        watts = DEFAULT_WATTS
        for key, val in WATTS.items():
            if key.lower() in gpu.lower():
                watts = val
                break
        total_kwh += (w['compute_seconds'] / 3600) * (watts / 1000)

    # Comparisons
    co2_kg = total_kwh * 0.417  # US grid avg kg CO2/kWh
    coal_kg = total_kwh * 0.453  # kg coal per kWh
    btc_per_tx_kwh = 1449  # Bitcoin kWh per transaction
    verified_jobs = conn.execute("SELECT COUNT(*) FROM jobs WHERE verified = 1").fetchone()[0]
    total_discoveries = conn.execute("SELECT COUNT(*) FROM discoveries").fetchone()[0]

    # conn reused (shared connection)
    return {
        "total_supply": round(total_minted, 4),
        "circulating": round(total_minted - total_staked, 4),
        "total_staked": round(total_staked, 4),
        "total_transferred": round(total_transferred, 4),
        "active_wallets": n_wallets,
        "total_transactions": n_txns,
        "top_holders": [{"name": r['name'] or r['worker_id'][:8], "balance": round(r['balance'], 4), "earned": round(r['total_earned'], 4)} for r in top],
        "stake_breakdown": stake_breakdown,
        "recent_transactions": [dict(t) for t in recent_txns],
        "chain": {
            "height": tip['height'] if tip else 0,
            "latest_hash": tip['block_hash'][:16] + "..." if tip else "genesis",
            "latest_time": tip['timestamp'] if tip else 0,
        },
        "rewards": {"all_types": W_BASE_RATE, "model": "equal_share"},
        "economy_model": {
            "type": "equal_share_profit_pool",
            "base_rate_per_result": W_BASE_RATE,
            "session_pool_base": W_SESSION_POOL_BASE,
            "session_pool_per_worker": W_SESSION_POOL_PER_WORKER,
            "min_turnaround": MIN_TURNAROUND,
            "description": "Flat rate per result. Session bonus pool split equally on level completion. High tide rises all ships.",
        },
        "energy": {
            "total_kwh": round(total_kwh, 4),
            "co2_kg": round(co2_kg, 4),
            "coal_kg": round(coal_kg, 4),
            "verified_jobs": verified_jobs,
            "kwh_per_verification": round(total_kwh / max(verified_jobs, 1), 6),
            "btc_equivalent_tx": round(total_kwh / btc_per_tx_kwh, 6) if total_kwh > 0 else 0,
            "w_per_kwh": round(total_minted / max(total_kwh, 0.0001), 2),
            "discoveries_per_kwh": round(total_discoveries / max(total_kwh, 0.0001), 1) if total_kwh > 0 else 0,
        },
    }

@app.get("/api/chain")
def chain_info():
    """Blockchain info — recent blocks with embedded science."""
    conn = get_db()
    blocks = conn.execute("""
        SELECT height, prev_hash, merkle_root, timestamp, n_transactions,
               total_minted, block_hash, science_hash, science_payload
        FROM blocks ORDER BY height DESC LIMIT 50
    """).fetchall()
    tip_height, _ = _get_chain_tip(conn)
    pending = conn.execute("SELECT COUNT(*) FROM transactions WHERE block_height = 0").fetchone()[0]

    block_list = []
    for b in blocks:
        bd = dict(b)
        # Parse science payload back to JSON for API consumers
        if bd.get('science_payload'):
            try:
                bd['science'] = json.loads(bd['science_payload'])
            except Exception:
                bd['science'] = None
            del bd['science_payload']  # don't double-send raw JSON string
        else:
            bd['science'] = None
        block_list.append(bd)

    return {
        "chain_height": tip_height,
        "pending_transactions": pending,
        "blocks": block_list,
    }

@app.get("/api/chain/{height}")
def chain_block(height: int):
    """Get a specific block with full science payload — the TOE at that moment."""
    conn = get_db()
    block = conn.execute("SELECT * FROM blocks WHERE height = ?", (height,)).fetchone()
    if not block:
        raise HTTPException(404, f"Block #{height} not found")
    bd = dict(block)
    if bd.get('science_payload'):
        try:
            bd['science'] = json.loads(bd['science_payload'])
        except Exception:
            bd['science'] = None
    # Get transactions in this block
    txns = conn.execute(
        "SELECT * FROM transactions WHERE block_height = ? ORDER BY timestamp ASC", (height,)
    ).fetchall()
    return {
        "block": bd,
        "transactions": [dict(t) for t in txns],
    }

@app.get("/api/genesis")
def genesis_block():
    """Read the genesis block — the Codex of the Fold, permanently embedded in Block #0."""
    conn = get_db()
    block = conn.execute("SELECT * FROM blocks WHERE height = 0").fetchone()
    tx = conn.execute("SELECT * FROM transactions WHERE block_height = 0 AND tx_type = 'genesis'").fetchone()
    if not block:
        return {"error": "Genesis block not yet created"}
    return {
        "block": dict(block),
        "codex": tx['memo'] if tx else GENESIS_MESSAGE.strip(),
        "codex_hash": hashlib.sha256(GENESIS_MESSAGE.encode()).hexdigest(),
        "message": "Nothing real is lost. Nothing false is crowned.",
    }

# ── Landing Page ──

@app.get("/", response_class=HTMLResponse)
def landing():
    return LANDING_HTML

# ═══════════════════════════════════════════════════════════
# Admin endpoints — protected by SERVER_SECRET
# ═══════════════════════════════════════════════════════════

def _verify_admin(x_admin_key: str):
    if not x_admin_key or x_admin_key != SERVER_SECRET:
        raise HTTPException(403, "Invalid admin key")

@app.post("/admin/ban")
def admin_ban(request: Request, x_admin_key: str = Header(default="")):
    """Ban a worker by ID or IP. Body: {"worker_id": "...", "ip": "...", "reason": "..."}"""
    _verify_admin(x_admin_key)
    import json as _json
    # Can't use pydantic model easily, just parse body
    # Accept worker_id and/or ip
    return JSONResponse({"error": "use /admin/ban-worker or /admin/ban-ip"}, 400)

@app.post("/admin/ban-worker/{worker_id}")
def admin_ban_worker(worker_id: str, reason: str = "", x_admin_key: str = Header(default="")):
    _verify_admin(x_admin_key)
    conn = get_db()
    conn.execute("UPDATE workers SET status = 'banned' WHERE id = ?", (worker_id,))
    conn.execute("INSERT INTO bans (worker_id, reason, banned_at) VALUES (?,?,?)",
                 (worker_id, reason, time.time()))
    # Re-queue all their unverified results
    conn.execute("""
        UPDATE jobs SET quorum_received = quorum_received - 1, status = 'pending'
        WHERE id IN (
            SELECT job_id FROM results WHERE worker_id = ?
            AND job_id NOT IN (SELECT id FROM jobs WHERE verified = 1)
        )
    """, (worker_id,))
    conn.execute("""
        DELETE FROM results WHERE worker_id = ?
        AND job_id NOT IN (SELECT id FROM jobs WHERE verified = 1)
    """, (worker_id,))
    conn.commit()
    name = conn.execute("SELECT name FROM workers WHERE id = ?", (worker_id,)).fetchone()
    # conn reused (shared connection)
    return {"status": "banned", "worker_id": worker_id, "name": name[0] if name else "unknown"}

@app.post("/admin/ban-ip/{ip}")
def admin_ban_ip(ip: str, reason: str = "", x_admin_key: str = Header(default="")):
    _verify_admin(x_admin_key)
    conn = get_db()
    conn.execute("INSERT INTO bans (ip, reason, banned_at) VALUES (?,?,?)",
                 (ip, reason, time.time()))
    conn.commit()
    # conn reused (shared connection)
    return {"status": "banned", "ip": ip}

@app.post("/admin/unban-worker/{worker_id}")
def admin_unban_worker(worker_id: str, x_admin_key: str = Header(default="")):
    _verify_admin(x_admin_key)
    conn = get_db()
    conn.execute("UPDATE workers SET status = 'active' WHERE id = ?", (worker_id,))
    conn.execute("DELETE FROM bans WHERE worker_id = ?", (worker_id,))
    conn.commit()
    # conn reused (shared connection)
    return {"status": "unbanned", "worker_id": worker_id}

@app.post("/admin/unban-ip/{ip}")
def admin_unban_ip(ip: str, x_admin_key: str = Header(default="")):
    _verify_admin(x_admin_key)
    conn = get_db()
    conn.execute("DELETE FROM bans WHERE ip = ?", (ip,))
    conn.commit()
    # conn reused (shared connection)
    return {"status": "unbanned", "ip": ip}

@app.get("/admin/audit")
def admin_audit(x_admin_key: str = Header(default=""), limit: int = 50):
    """View recent activity: workers, IPs, trust scores, flagged accounts."""
    _verify_admin(x_admin_key)
    conn = get_db()
    workers = [dict(r) for r in conn.execute("""
        SELECT id, name, status, trust_score, jobs_completed, canaries_passed, canaries_failed,
               registered_at, last_heartbeat
        FROM workers ORDER BY last_heartbeat DESC LIMIT ?
    """, (limit,)).fetchall()]
    # Attach known IPs to each worker
    for w in workers:
        ips = conn.execute(
            "SELECT DISTINCT ip FROM ip_log WHERE worker_id = ? ORDER BY timestamp DESC LIMIT 10",
            (w['id'],)).fetchall()
        w['known_ips'] = [r[0] for r in ips]
    flagged = [w for w in workers if w['status'] in ('flagged', 'banned') or w['trust_score'] < 0.5]
    bans = [dict(r) for r in conn.execute("SELECT * FROM bans ORDER BY banned_at DESC").fetchall()]
    # IP collision report: IPs with multiple worker accounts
    collisions = [dict(r) for r in conn.execute("""
        SELECT ip, COUNT(DISTINCT worker_id) as n_workers, GROUP_CONCAT(DISTINCT worker_id) as worker_ids
        FROM ip_log GROUP BY ip HAVING n_workers > 1 ORDER BY n_workers DESC LIMIT 20
    """).fetchall()]
    # conn reused (shared connection)
    return {
        "workers": workers,
        "flagged": flagged,
        "bans": bans,
        "ip_collisions": collisions,
    }

@app.post("/admin/revoke-key/{worker_id}")
def admin_revoke_keys(worker_id: str, x_admin_key: str = Header(default="")):
    """Revoke all API keys for a worker (forces re-login)."""
    _verify_admin(x_admin_key)
    conn = get_db()
    conn.execute("DELETE FROM api_keys WHERE worker_id = ?", (worker_id,))
    conn.execute("UPDATE workers SET api_key_hash = 'revoked' WHERE id = ?", (worker_id,))
    conn.commit()
    # conn reused (shared connection)
    return {"status": "keys_revoked", "worker_id": worker_id}

# ═══════════════════════════════════════════════════════════
# Agent Mode — Smart Tier Endpoints
# ═══════════════════════════════════════════════════════════

PRIORITY_JOB_BASE = 10000000  # IDs for agent-generated jobs (high range)

class HintSubmit(BaseModel):
    hint_type: str
    lambda_center: float = 0.0
    lambda_width: float = 0.0
    confidence: float = 0.0
    constants_involved: List[str] = []
    observation: str = ""
    requested_resolution: float = 0.0

@app.post("/api/hint")
def submit_hint(hint: HintSubmit, request: Request, x_api_key: str = Header()):
    """Submit adaptive sampling hint. Creates priority jobs in hinted region."""
    ip = _get_client_ip(request)
    _check_rate(ip, "hint")
    conn = get_db()
    if _is_banned(conn, ip):
        raise HTTPException(403, "Access denied")
    worker = verify_worker(x_api_key, conn)
    if not worker:
        raise HTTPException(401, "Invalid API key")

    now = time.time()
    conn.execute(
        "INSERT INTO hints (worker_id, hint_type, lambda_center, lambda_width, confidence, constants_involved, observation, requested_resolution, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (worker['id'], hint.hint_type, hint.lambda_center, hint.lambda_width,
         hint.confidence, json.dumps(hint.constants_involved), hint.observation,
         hint.requested_resolution, now)
    )
    hint_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    # Generate priority jobs in hinted lambda region
    jobs_created = 0
    if hint.lambda_center > 0 and hint.lambda_width > 0:
        resolution = hint.requested_resolution if hint.requested_resolution > 0 else _lambda_step_for_level(CURRENT_LEVEL)
        lam_start = hint.lambda_center - hint.lambda_width / 2
        lam_end = hint.lambda_center + hint.lambda_width / 2
        max_jobs = 100  # cap per hint
        lam = lam_start
        # Find next available ID
        max_id = conn.execute("SELECT COALESCE(MAX(id), ?) FROM jobs WHERE id >= ?",
                              (PRIORITY_JOB_BASE - 1, PRIORITY_JOB_BASE)).fetchone()[0]
        next_id = max(max_id + 1, PRIORITY_JOB_BASE)

        while lam <= lam_end and jobs_created < max_jobs:
            # Check if a job already exists near this lambda
            existing = conn.execute(
                "SELECT id FROM jobs WHERE ABS(lambda_val - ?) < ?",
                (lam, resolution / 2)
            ).fetchone()
            if not existing:
                conn.execute(
                    "INSERT OR IGNORE INTO jobs (id, lambda_val, status, quorum_target, quorum_received, verified, created_at, priority, source_hint_id) VALUES (?,?,?,?,?,?,?,?,?)",
                    (next_id, round(lam, 10), 'pending', QUORUM_SIZE, 0, 0, now, 1, hint_id)
                )
                next_id += 1
                jobs_created += 1
            lam += resolution

    conn.execute("UPDATE hints SET jobs_created = ? WHERE id = ?", (jobs_created, hint_id))
    conn.commit()
    # conn reused (shared connection)

    return {"status": "accepted", "hint_id": hint_id, "jobs_created": jobs_created}


class HypothesisSubmit(BaseModel):
    hypothesis: str
    test_lambdas: List[float] = []
    prediction: str = ""
    falsifiable: bool = True

@app.post("/api/hypothesis")
def submit_hypothesis(hyp: HypothesisSubmit, request: Request, x_api_key: str = Header()):
    """Submit testable hypothesis. Creates highest-priority jobs for test lambdas."""
    ip = _get_client_ip(request)
    _check_rate(ip, "hypothesis")
    conn = get_db()
    if _is_banned(conn, ip):
        raise HTTPException(403, "Access denied")
    worker = verify_worker(x_api_key, conn)
    if not worker:
        raise HTTPException(401, "Invalid API key")

    now = time.time()
    conn.execute(
        "INSERT INTO hypotheses (worker_id, hypothesis, test_lambdas, prediction, falsifiable, created_at) VALUES (?,?,?,?,?,?)",
        (worker['id'], hyp.hypothesis, json.dumps(hyp.test_lambdas),
         hyp.prediction, 1 if hyp.falsifiable else 0, now)
    )
    hyp_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    # Create priority-2 jobs for each test lambda
    jobs_created = 0
    jobs_existing = 0
    max_id = conn.execute("SELECT COALESCE(MAX(id), ?) FROM jobs WHERE id >= ?",
                          (PRIORITY_JOB_BASE - 1, PRIORITY_JOB_BASE)).fetchone()[0]
    next_id = max(max_id + 1, PRIORITY_JOB_BASE)

    for lam in hyp.test_lambdas[:20]:  # cap at 20 test lambdas per hypothesis
        existing = conn.execute(
            "SELECT id FROM jobs WHERE ABS(lambda_val - ?) < ?",
            (lam, _lambda_step_for_level(CURRENT_LEVEL) / 2)
        ).fetchone()
        if existing:
            jobs_existing += 1
        else:
            conn.execute(
                "INSERT OR IGNORE INTO jobs (id, lambda_val, status, quorum_target, quorum_received, verified, created_at, priority, source_hypothesis_id) VALUES (?,?,?,?,?,?,?,?,?)",
                (next_id, round(lam, 10), 'pending', QUORUM_SIZE, 0, 0, now, 2, hyp_id)
            )
            next_id += 1
            jobs_created += 1

    conn.commit()
    # conn reused (shared connection)

    return {"status": "accepted", "hypothesis_id": hyp_id, "jobs_created": jobs_created, "jobs_existing": jobs_existing}


class ObservationSubmit(BaseModel):
    text: str

@app.post("/api/observe")
def submit_observation(obs: ObservationSubmit, request: Request, x_api_key: str = Header()):
    """Write to the shared observation blackboard."""
    ip = _get_client_ip(request)
    _check_rate(ip, "observe")
    conn = get_db()
    if _is_banned(conn, ip):
        raise HTTPException(403, "Access denied")
    worker = verify_worker(x_api_key, conn)
    if not worker:
        raise HTTPException(401, "Invalid API key")

    text = obs.text[:2000]  # cap length
    now = time.time()
    conn.execute(
        "INSERT INTO observations (worker_id, worker_name, content, created_at) VALUES (?,?,?,?)",
        (worker['id'], worker.get('name', ''), text, now)
    )
    obs_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.commit()
    # conn reused (shared connection)

    return {"status": "logged", "observation_id": obs_id}


@app.get("/api/hints")
def list_hints(limit: int = 50, status: str = ""):
    """List recent hints (public)."""
    conn = get_db()
    if status:
        rows = conn.execute(
            "SELECT h.*, w.name as worker_name FROM hints h LEFT JOIN workers w ON h.worker_id = w.id WHERE h.status = ? ORDER BY h.created_at DESC LIMIT ?",
            (status, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT h.*, w.name as worker_name FROM hints h LEFT JOIN workers w ON h.worker_id = w.id ORDER BY h.created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
    # conn reused (shared connection)
    return [dict(r) for r in rows]


@app.get("/api/hypotheses")
def list_hypotheses(limit: int = 50, status: str = ""):
    """List hypotheses and their status (public)."""
    conn = get_db()
    if status:
        rows = conn.execute(
            "SELECT h.*, w.name as worker_name FROM hypotheses h LEFT JOIN workers w ON h.worker_id = w.id WHERE h.status = ? ORDER BY h.created_at DESC LIMIT ?",
            (status, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT h.*, w.name as worker_name FROM hypotheses h LEFT JOIN workers w ON h.worker_id = w.id ORDER BY h.created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
    # conn reused (shared connection)
    return [dict(r) for r in rows]


@app.get("/api/observations")
def list_observations(limit: int = 100, since: float = 0):
    """Read the shared observation blackboard."""
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM observations WHERE created_at > ? ORDER BY created_at DESC LIMIT ?",
        (since, limit)
    ).fetchall()
    # conn reused (shared connection)
    return [dict(r) for r in rows]


# ── Dashboard ──

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD_HTML

# ── Chat ──

@app.get("/chat", response_class=HTMLResponse)
def chat_page():
    return CHAT_HTML

# ── Economy ──

@app.get("/economy", response_class=HTMLResponse)
def economy_page():
    return ECONOMY_HTML

# ═══════════════════════════════════════════════════════════
# Embedded Dashboard
# ═══════════════════════════════════════════════════════════

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>W@Home Hive — Dashboard</title>
<style>
  :root {
    --bg: #0a0a10; --bg2: #12121e; --bg3: #1a1a2e;
    --cyan: #a0f0ff; --gold: #ffd06a; --violet: #c4a0ff;
    --green: #80ffaa; --red: #ff6b6b; --text: #c8c8d8;
    --dim: #555568; --border: #2a2a3a; --mono: 'JetBrains Mono', monospace;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: var(--bg); color: var(--text); font-family: var(--mono);
    min-height: 100vh; padding: 1.5em;
  }
  .header {
    text-align: center; padding: 2em 0 1em;
    border-bottom: 1px solid var(--border); margin-bottom: 2em;
  }
  .header h1 { color: var(--cyan); font-size: 1.8em; letter-spacing: 0.08em; }
  .header .sub { color: var(--dim); font-size: 0.85em; margin-top: 0.3em; }
  .grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1.2em; max-width: 1200px; margin: 0 auto;
  }
  .card {
    background: var(--bg2); border: 1px solid var(--border);
    border-radius: 8px; padding: 1.3em;
  }
  .card h2 {
    color: var(--violet); font-size: 0.85em; letter-spacing: 0.06em;
    text-transform: uppercase; margin-bottom: 1em;
  }
  .big-num { font-size: 2.2em; color: var(--cyan); font-weight: bold; }
  .big-num .unit { font-size: 0.4em; color: var(--dim); margin-left: 0.3em; }
  .stat-row {
    display: flex; justify-content: space-between; padding: 0.4em 0;
    border-bottom: 1px solid var(--border); font-size: 0.85em;
  }
  .stat-row:last-child { border-bottom: none; }
  .stat-label { color: var(--dim); }
  .stat-val { color: var(--text); }
  .stat-val.green { color: var(--green); }
  .stat-val.gold { color: var(--gold); }
  .stat-val.red { color: var(--red); }

  /* Lambda sweep bar */
  .sweep-container { grid-column: 1 / -1; }
  .sweep-bar {
    height: 32px; background: var(--bg3); border-radius: 4px;
    position: relative; overflow: hidden; margin: 1em 0;
  }
  .sweep-fill {
    height: 100%; border-radius: 4px;
    background: linear-gradient(90deg, var(--violet), var(--cyan));
    transition: width 1s ease;
  }
  .sweep-marker {
    position: absolute; top: 0; height: 100%; width: 2px;
    background: var(--gold); box-shadow: 0 0 8px var(--gold);
  }
  .sweep-labels {
    display: flex; justify-content: space-between;
    font-size: 0.75em; color: var(--dim); margin-top: 0.3em;
  }
  .sweep-pct {
    position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
    font-size: 0.9em; color: var(--text); text-shadow: 0 0 4px var(--bg);
    font-weight: bold;
  }

  /* Workers table */
  .workers-table { width: 100%; border-collapse: collapse; font-size: 0.8em; }
  .workers-table th {
    text-align: left; padding: 0.5em; color: var(--cyan);
    border-bottom: 1px solid var(--border); font-weight: normal;
  }
  .workers-table td { padding: 0.5em; border-bottom: 1px solid var(--border); }
  .workers-table tr:hover { background: var(--bg3); }
  .online-dot {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    margin-right: 0.5em;
  }
  .online-dot.on { background: var(--green); box-shadow: 0 0 6px var(--green); }
  .online-dot.off { background: var(--dim); }

  /* Discovery feed */
  .discovery {
    background: var(--bg3); border-left: 3px solid var(--gold);
    padding: 0.8em 1em; margin-bottom: 0.6em; border-radius: 0 4px 4px 0;
    font-size: 0.85em;
  }
  .discovery .const-name { color: var(--gold); font-weight: bold; }
  .discovery .lambda-val { color: var(--cyan); }
  .discovery .time-ago { color: var(--dim); font-size: 0.8em; }

  /* Live pulse */
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
  .live { animation: pulse 2s ease infinite; }
  .live-dot {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    background: var(--green); margin-right: 0.5em;
    box-shadow: 0 0 6px var(--green); animation: pulse 2s ease infinite;
  }

  .depth-btn {
    background: var(--bg3); color: var(--dim); border: 1px solid var(--border);
    padding: 0.3em 0.7em; border-radius: 4px; cursor: pointer;
    font-family: var(--mono); font-size: 0.75em;
  }
  .depth-btn:hover { border-color: var(--violet); color: var(--text); }
  .depth-btn.active { background: var(--violet); color: var(--bg); border-color: var(--violet); }
  .axiom {
    text-align: center; color: var(--dim); font-size: 0.8em;
    margin-top: 3em; padding-top: 1.5em; border-top: 1px solid var(--border);
  }
</style>
</head>
<body>

<div class="header">
  <h1>W@HOME HIVE</h1>
  <div class="sub">Akataleptos Distributed Spectral Search</div>
  <div class="sub" style="margin-top: 0.5em;">
    <span class="live-dot"></span>
    <span id="status-text">Connecting...</span>
    &nbsp;&nbsp;|&nbsp;&nbsp;
    <a href="/" style="color:#a78bfa;">Home</a> &nbsp;
    <a href="/chat" style="color:#a78bfa;">Chat</a> &nbsp;
    <a href="/economy" style="color:#ffd06a;">W Economy</a>
  </div>
</div>

<div class="grid">

  <!-- Lambda Sweep Progress -->
  <div class="card sweep-container">
    <h2>Lambda Sweep Progress</h2>
    <div class="sweep-bar">
      <div class="sweep-fill" id="sweep-fill" style="width: 0%"></div>
      <div class="sweep-marker" id="sweep-marker" style="left: 0%"></div>
      <div class="sweep-pct" id="sweep-pct">0%</div>
    </div>
    <div class="sweep-labels">
      <span>&lambda; = 0.400000</span>
      <span id="current-lambda">&lambda; = ?</span>
      <span>&lambda; = 0.600000</span>
    </div>
  </div>

  <!-- Stats -->
  <div class="card">
    <h2>Computation</h2>
    <div class="big-num" id="jobs-done">0<span class="unit">/ <span id="jobs-total">200,000</span></span></div>
    <div style="margin-top: 1em;">
      <div class="stat-row"><span class="stat-label">Verified</span><span class="stat-val green" id="verified">0</span></div>
      <div class="stat-row"><span class="stat-label">In Progress</span><span class="stat-val" id="assigned">0</span></div>
      <div class="stat-row"><span class="stat-label">Pending</span><span class="stat-val" id="pending">0</span></div>
      <div class="stat-row"><span class="stat-label">Disputed</span><span class="stat-val red" id="disputed">0</span></div>
    </div>
  </div>

  <div class="card">
    <h2>Volunteers</h2>
    <div class="big-num" id="workers-online">0<span class="unit">online</span></div>
    <div style="margin-top: 1em;">
      <div class="stat-row"><span class="stat-label">Jobs/Hour</span><span class="stat-val gold" id="jobs-hr">0</span></div>
      <div class="stat-row"><span class="stat-label">Compute Time</span><span class="stat-val" id="compute-hrs">0h</span></div>
      <div class="stat-row"><span class="stat-label">ETA</span><span class="stat-val" id="eta">--</span></div>
    </div>
  </div>

  <div class="card">
    <h2>Hits</h2>
    <div class="big-num" id="total-discoveries">0<span class="unit">hits</span></div>
    <div style="font-size: 0.9em; color: var(--green); margin-top: 0.3em;" id="confirmed-count"></div>
    <div id="discovery-feed" style="margin-top: 1em; max-height: 200px; overflow-y: auto;"></div>
  </div>

  <!-- Per-type Stats -->
  <div class="card" style="grid-column: 1 / -1;">
    <h2>Work Types</h2>
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1em;">
      <div style="background: var(--bg3); border-radius: 6px; padding: 1em; border-left: 3px solid var(--violet);">
        <div style="color: var(--violet); font-size: 0.75em; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.5em;">Eigenvalue Search</div>
        <div style="font-size: 1.4em; color: var(--cyan);" id="wt-eig-pct">0%</div>
        <div style="font-size: 0.75em; color: var(--dim); margin-top: 0.3em;"><span id="wt-eig-done">0</span> jobs &middot; <span id="wt-eig-hrs">0</span>h</div>
        <div style="height: 4px; background: var(--bg); border-radius: 2px; margin-top: 0.5em; overflow: hidden;">
          <div id="wt-eig-bar" style="height: 100%; background: linear-gradient(90deg, var(--violet), var(--cyan)); width: 0%; transition: width 1s;"></div>
        </div>
      </div>
      <div style="background: var(--bg3); border-radius: 6px; padding: 1em; border-left: 3px solid var(--green);">
        <div style="color: var(--green); font-size: 0.75em; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.5em;">Fractal Falsification</div>
        <div style="font-size: 1.4em; color: var(--green);" id="wt-fals-pct">0%</div>
        <div style="font-size: 0.75em; color: var(--dim); margin-top: 0.3em;"><span id="wt-fals-done">0</span> jobs &middot; <span id="wt-fals-hrs">0</span>h</div>
        <div style="height: 4px; background: var(--bg); border-radius: 2px; margin-top: 0.5em; overflow: hidden;">
          <div id="wt-fals-bar" style="height: 100%; background: linear-gradient(90deg, #22c55e, var(--gold)); width: 0%; transition: width 1s;"></div>
        </div>
      </div>
      <div style="background: var(--bg3); border-radius: 6px; padding: 1em; border-left: 3px solid var(--cyan);">
        <div style="color: var(--cyan); font-size: 0.75em; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.5em;">Countdown Clock</div>
        <div style="font-size: 1.4em; color: var(--cyan);" id="wt-clock-pct">0%</div>
        <div style="font-size: 0.75em; color: var(--dim); margin-top: 0.3em;"><span id="wt-clock-done">0</span> jobs &middot; <span id="wt-clock-hrs">0</span>h</div>
        <div style="height: 4px; background: var(--bg); border-radius: 2px; margin-top: 0.5em; overflow: hidden;">
          <div id="wt-clock-bar" style="height: 100%; background: linear-gradient(90deg, var(--cyan), #e0e0f0); width: 0%; transition: width 1s;"></div>
        </div>
      </div>
    </div>
  </div>

  <!-- W Economy -->
  <div class="card">
    <h2 style="color: var(--gold);">W Economy</h2>
    <div style="display: flex; gap: 1.5em; flex-wrap: wrap;">
      <div>
        <div style="color: var(--dim); font-size: 0.75em;">Total Supply</div>
        <div style="font-size: 1.6em; color: var(--gold);" id="w-supply">0</div>
      </div>
      <div>
        <div style="color: var(--dim); font-size: 0.75em;">Staked</div>
        <div style="font-size: 1.6em; color: var(--violet);" id="w-staked">0</div>
      </div>
      <div>
        <div style="color: var(--dim); font-size: 0.75em;">Wallets</div>
        <div style="font-size: 1.6em; color: var(--cyan);" id="w-wallets">0</div>
      </div>
      <div>
        <div style="color: var(--dim); font-size: 0.75em;">Chain Height</div>
        <div style="font-size: 1.6em; color: var(--green);" id="w-height">0</div>
      </div>
    </div>
    <div style="margin-top: 1em; text-align: right;">
      <a href="/economy" style="color: var(--gold); font-size: 0.8em;">Full Economy →</a>
    </div>
  </div>

  <!-- Menger Sponge Progress Viz -->
  <div class="card" style="grid-column: 1 / -1; display: flex; flex-direction: column; align-items: center;">
    <h2>Spectral Search Map</h2>
    <canvas id="sponge-canvas" width="800" height="550" style="max-width: 100%; cursor: grab;"></canvas>
    <div style="display: flex; gap: 1em; margin-top: 0.8em; align-items: center; flex-wrap: wrap; justify-content: center;">
      <div style="display: flex; gap: 0.3em;">
        <button onclick="setDepth(1)" id="btn-L1" class="depth-btn">L1</button>
        <button onclick="setDepth(2)" id="btn-L2" class="depth-btn active">L2</button>
        <button onclick="setDepth(3)" id="btn-L3" class="depth-btn">L3</button>
        <button onclick="setDepth(4)" id="btn-L4" class="depth-btn" title="160,000 cubes — for the brave">L4</button>
      </div>
      <div style="display: flex; gap: 0.3em;">
        <button onclick="setRender('full')" id="btn-full" class="depth-btn active" title="Full quality">HD</button>
        <button onclick="setRender('fast')" id="btn-fast" class="depth-btn" title="Fast — no strokes/glow">Fast</button>
        <button onclick="setRender('off')" id="btn-off" class="depth-btn" title="Hide visualization">Off</button>
      </div>
      <span style="font-size: 0.7em; color: var(--dim);" id="cube-count">400 cubes</span>
      <span style="font-size: 0.7em; color: var(--dim);" id="fps-counter"></span>
      <span style="font-size: 0.7em; margin-left: 1em;">
        <span style="color: #1a1a2e;">&#9632;</span> Pending
        <span style="color: #6a4aaa; margin-left: 0.5em;">&#9632;</span> Computing
        <span style="color: var(--violet); margin-left: 0.5em;">&#9632;</span> Partial
        <span style="color: var(--cyan); margin-left: 0.5em;">&#9632;</span> Complete
        <span style="color: var(--gold); margin-left: 0.5em;">&#9632;</span> Hit
      </span>
    </div>
  </div>

  <!-- Active Computation -->
  <div class="card" style="grid-column: 1 / -1;">
    <h2>Active Computation</h2>
    <table class="workers-table">
      <thead><tr>
        <th>Job</th><th>&lambda;</th><th>Worker</th><th>GPU</th><th>Elapsed</th><th>Status</th>
      </tr></thead>
      <tbody id="active-body">
        <tr><td colspan="6" style="color: var(--dim);">Waiting for workers...</td></tr>
      </tbody>
    </table>
  </div>

  <!-- Leaderboard -->
  <div class="card" style="grid-column: 1 / -1;">
    <h2>Leaderboard</h2>
    <table class="workers-table">
      <thead><tr>
        <th></th><th>Volunteer</th><th>Jobs</th><th>Compute</th><th>Hits</th><th>W</th><th>GPU</th>
      </tr></thead>
      <tbody id="leaderboard-body"></tbody>
    </table>
  </div>
</div>

<div class="axiom">1 = 0 = &infin; &mdash; Akataleptos</div>

<script>
const API = window.location.origin;

function timeAgo(ts) {
  const s = Math.floor(Date.now()/1000 - ts);
  if (s < 60) return s + 's ago';
  if (s < 3600) return Math.floor(s/60) + 'm ago';
  if (s < 86400) return Math.floor(s/3600) + 'h ago';
  return Math.floor(s/86400) + 'd ago';
}

async function update() {
  try {
    const [prog, disc, lb, active, econ] = await Promise.all([
      fetch(API + '/progress').then(r => r.json()),
      fetch(API + '/discoveries').then(r => r.json()),
      fetch(API + '/leaderboard').then(r => r.json()),
      fetch(API + '/active').then(r => r.json()),
      fetch(API + '/api/economy').then(r => r.json()).catch(() => null),
    ]);

    document.getElementById('status-text').textContent = 'Live';
    document.getElementById('sweep-fill').style.width = prog.percent_complete + '%';
    document.getElementById('sweep-pct').textContent = prog.percent_complete.toFixed(2) + '%';

    const markerPct = ((prog.current_lambda - 0.4) / 0.2) * 100;
    document.getElementById('sweep-marker').style.left = markerPct + '%';
    document.getElementById('current-lambda').textContent = '\\u03bb = ' + prog.current_lambda.toFixed(6);

    document.getElementById('jobs-done').innerHTML = prog.completed.toLocaleString() +
      '<span class="unit">/ <span>' + prog.total_jobs.toLocaleString() + '</span></span>';
    document.getElementById('verified').textContent = prog.verified.toLocaleString();
    document.getElementById('assigned').textContent = prog.assigned.toLocaleString();
    document.getElementById('pending').textContent = prog.pending.toLocaleString();
    document.getElementById('disputed').textContent = prog.disputed.toLocaleString();

    document.getElementById('workers-online').innerHTML = prog.active_workers +
      '<span class="unit">online</span>';
    document.getElementById('jobs-hr').textContent = prog.jobs_per_hour.toLocaleString();
    document.getElementById('compute-hrs').textContent = prog.total_compute_hours + 'h';
    document.getElementById('eta').textContent = prog.eta_hours ? prog.eta_hours + 'h' : '--';

    document.getElementById('total-discoveries').innerHTML = prog.total_discoveries +
      '<span class="unit">hits</span>';
    const confEl = document.getElementById('confirmed-count');
    confEl.textContent = prog.total_confirmed ? prog.total_confirmed + ' confirmed' : '';

    // Per-type stats
    if (prog.per_type) {
      const pt = prog.per_type;
      if (pt.eigenvalue) {
        document.getElementById('wt-eig-pct').textContent = pt.eigenvalue.pct + '%';
        document.getElementById('wt-eig-done').textContent = pt.eigenvalue.completed.toLocaleString();
        document.getElementById('wt-eig-hrs').textContent = pt.eigenvalue.compute_hours;
        document.getElementById('wt-eig-bar').style.width = pt.eigenvalue.pct + '%';
      }
      if (pt.falsification) {
        document.getElementById('wt-fals-pct').textContent = pt.falsification.pct + '%';
        document.getElementById('wt-fals-done').textContent = pt.falsification.completed.toLocaleString();
        document.getElementById('wt-fals-hrs').textContent = pt.falsification.compute_hours;
        document.getElementById('wt-fals-bar').style.width = pt.falsification.pct + '%';
      }
      if (pt.clock) {
        document.getElementById('wt-clock-pct').textContent = pt.clock.pct + '%';
        document.getElementById('wt-clock-done').textContent = pt.clock.completed.toLocaleString();
        document.getElementById('wt-clock-hrs').textContent = pt.clock.compute_hours;
        document.getElementById('wt-clock-bar').style.width = pt.clock.pct + '%';
      }
    }

    // W Economy card
    if (econ) {
      document.getElementById('w-supply').textContent = econ.total_supply.toFixed(2) + ' W';
      document.getElementById('w-staked').textContent = econ.total_staked.toFixed(2) + ' W';
      document.getElementById('w-wallets').textContent = econ.active_wallets;
      document.getElementById('w-height').textContent = econ.chain ? econ.chain.height : 0;
    }

    // Hit feed
    const feed = document.getElementById('discovery-feed');
    feed.innerHTML = disc.slice(0, 10).map(d => {
      const tier = d.param_tier === 'mobile' ? ' <span style="color:#7070a0;font-size:0.8em;">[scout]</span>' : ' <span style="color:var(--green);font-size:0.8em;">[HD]</span>';
      const conf = d.job_verified ? ' <span style="color:var(--gold);font-size:0.8em;font-weight:bold;">[CONFIRMED]</span>' : '';
      return '<div class="discovery">' +
        '<span class="const-name">' + d.constant_name + '</span> ' +
        'at <span class="lambda-val">&lambda;=' + d.lambda_val.toFixed(6) + '</span> ' +
        '(ratio=' + d.ratio_value.toFixed(5) + ') ' + tier + conf +
        '<span class="time-ago">' + timeAgo(d.discovered_at) + '</span>' +
      '</div>';
    }).join('') || '<div style="color: var(--dim); font-size: 0.85em;">No hits yet...</div>';

    // Active computation
    const actBody = document.getElementById('active-body');
    function fmtElapsed(s) {
      if (s < 60) return s + 's';
      if (s < 3600) return Math.floor(s/60) + 'm ' + (s%60) + 's';
      return Math.floor(s/3600) + 'h ' + Math.floor((s%3600)/60) + 'm';
    }
    actBody.innerHTML = active.map(a =>
      '<tr>' +
        '<td style="color: var(--cyan);">#' + a.job_id + '</td>' +
        '<td style="color: var(--gold);">' + a.lambda.toFixed(6) + '</td>' +
        '<td>' + a.worker + '</td>' +
        '<td style="color: var(--dim);">' + a.gpu + '</td>' +
        '<td>' + fmtElapsed(a.elapsed) + '</td>' +
        '<td><span class="live" style="color: var(--green);">computing</span></td>' +
      '</tr>'
    ).join('') || '<tr><td colspan="6" style="color: var(--dim);">No active jobs</td></tr>';

    // Leaderboard
    const lbBody = document.getElementById('leaderboard-body');
    lbBody.innerHTML = lb.map(w =>
      '<tr>' +
        '<td><span class="online-dot ' + (w.online ? 'on' : 'off') + '"></span></td>' +
        '<td>' + w.name + '</td>' +
        '<td>' + w.jobs.toLocaleString() + '</td>' +
        '<td>' + w.compute_hours + 'h</td>' +
        '<td style="color: var(--gold);">' + w.discoveries + '</td>' +
        '<td style="color: #ffd06a;">' + (w.w_balance || 0).toFixed(2) + '</td>' +
        '<td style="color: var(--dim);">' + (w.gpu || 'CPU') + '</td>' +
      '</tr>'
    ).join('') || '<tr><td colspan="7" style="color: var(--dim);">No volunteers yet</td></tr>';

  } catch(e) {
    document.getElementById('status-text').textContent = 'Disconnected';
  }
}

// ── Menger Sponge Visualization (Rotatable) ──
const spongeCanvas = document.getElementById('sponge-canvas');
const sCtx = spongeCanvas.getContext('2d');

// Menger sponge generator — variable depth
function isMengerCell(x, y, z) {
  while (x > 0 || y > 0 || z > 0) {
    if (((x%3===1)?1:0) + ((y%3===1)?1:0) + ((z%3===1)?1:0) >= 2) return false;
    x = Math.floor(x/3); y = Math.floor(y/3); z = Math.floor(z/3);
  }
  return true;
}
let mengerCubes = [];
let currentDepth = 2;
let cubeSize = 0.32;
const depthConfig = {
  1: {N: 3, div: 1, size: 0.88, scale: 52},
  2: {N: 9, div: 3, size: 0.32, scale: 105},
  3: {N: 27, div: 9, size: 0.105, scale: 105},
  4: {N: 81, div: 27, size: 0.035, scale: 105},
};

function buildSponge(depth) {
  const cfg = depthConfig[depth];
  const N = cfg.N;
  const half = (N - 1) / 2;
  mengerCubes = [];
  for (let x = 0; x < N; x++)
    for (let y = 0; y < N; y++)
      for (let z = 0; z < N; z++) {
        if (!isMengerCell(x, y, z)) continue;
        mengerCubes.push({x: (x - half) / cfg.div, y: (y - half) / cfg.div, z: (z - half) / cfg.div});
      }
  cubeSize = cfg.size;
  currentScale = cfg.scale;
  document.getElementById('cube-count').textContent = mengerCubes.length + ' cubes';
}

let currentScale = 105;

function setDepth(d) {
  currentDepth = d;
  buildSponge(d);
  document.querySelectorAll('.depth-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('btn-L' + d).classList.add('active');
  updateSponge();
}

let renderMode = 'full'; // 'full', 'fast', 'off'
let lastFrameTime = performance.now();
let frameCount = 0;
let fpsDisplay = 0;

function setRender(mode) {
  renderMode = mode;
  document.querySelectorAll('#btn-full,#btn-fast,#btn-off').forEach(b => b.classList.remove('active'));
  document.getElementById('btn-' + mode).classList.add('active');
  const canvas = document.getElementById('sponge-canvas');
  if (mode === 'off') {
    canvas.style.display = 'none';
    if (animFrame) cancelAnimationFrame(animFrame);
    animFrame = null;
  } else {
    canvas.style.display = '';
    if (!animFrame) drawSponge();
  }
}

buildSponge(2); // default L2

// Rotation state
let rotY = 0.6;   // Y-axis rotation (horizontal drag)
let rotX = -0.4;  // X-axis tilt (vertical drag)
let autoRotate = true;
let dragging = false;
let lastMouse = {x: 0, y: 0};
let animFrame = null;

spongeCanvas.addEventListener('mousedown', e => {
  dragging = true; autoRotate = false;
  lastMouse = {x: e.clientX, y: e.clientY};
});
spongeCanvas.addEventListener('touchstart', e => {
  dragging = true; autoRotate = false;
  lastMouse = {x: e.touches[0].clientX, y: e.touches[0].clientY};
  e.preventDefault();
}, {passive: false});
window.addEventListener('mouseup', () => dragging = false);
window.addEventListener('touchend', () => dragging = false);
window.addEventListener('mousemove', e => {
  if (!dragging) return;
  rotY += (e.clientX - lastMouse.x) * 0.01;
  rotX += (e.clientY - lastMouse.y) * 0.01;
  rotX = Math.max(-1.2, Math.min(1.2, rotX));
  lastMouse = {x: e.clientX, y: e.clientY};
});
window.addEventListener('touchmove', e => {
  if (!dragging) return;
  rotY += (e.touches[0].clientX - lastMouse.x) * 0.01;
  rotX += (e.touches[0].clientY - lastMouse.y) * 0.01;
  rotX = Math.max(-1.2, Math.min(1.2, rotX));
  lastMouse = {x: e.touches[0].clientX, y: e.touches[0].clientY};
}, {passive: false});
// Double-click to resume auto-rotate
spongeCanvas.addEventListener('dblclick', () => autoRotate = true);

function rotate3D(x, y, z) {
  // Y-axis rotation
  let x1 = x * Math.cos(rotY) - z * Math.sin(rotY);
  let z1 = x * Math.sin(rotY) + z * Math.cos(rotY);
  // X-axis tilt
  let y1 = y * Math.cos(rotX) - z1 * Math.sin(rotX);
  let z2 = y * Math.sin(rotX) + z1 * Math.cos(rotX);
  return {x: x1, y: y1, z: z2};
}

function project(x, y, z) {
  const r = rotate3D(x, y, z);
  const scale = currentScale;
  const perspective = 6;
  const f = perspective / (perspective + r.z);
  return {
    x: r.x * scale * f + spongeCanvas.width / 2,
    y: -r.y * scale * f + spongeCanvas.height / 2,
    depth: r.z
  };
}

function drawIsoCube(ctx, cx, cy, cz, size, color, glow, pulse) {
  const s = size / 2;
  const c = [
    project(cx-s, cy-s, cz-s), project(cx+s, cy-s, cz-s),
    project(cx+s, cy-s, cz+s), project(cx-s, cy-s, cz+s),
    project(cx-s, cy+s, cz-s), project(cx+s, cy+s, cz-s),
    project(cx+s, cy+s, cz+s), project(cx-s, cy+s, cz+s),
  ];

  let r, g, b;
  if (color.startsWith('#')) {
    r = parseInt(color.slice(1,3),16);
    g = parseInt(color.slice(3,5),16);
    b = parseInt(color.slice(5,7),16);
  } else { r = 26; g = 26; b = 46; }

  let pm = 1;
  if (pulse) pm = 0.5 + 0.5 * Math.sin(Date.now() / 400);

  const faces = [
    {verts: [4,5,6,7], shade: 1.0},
    {verts: [0,1,5,4], shade: 0.75},
    {verts: [1,2,6,5], shade: 0.85},
    {verts: [3,0,4,7], shade: 0.65},
    {verts: [2,3,7,6], shade: 0.7},
    {verts: [0,3,2,1], shade: 0.5},
  ];

  const fast = renderMode === 'fast';
  faces.forEach(f => {
    const v = f.verts;
    const ax = c[v[1]].x - c[v[0]].x, ay = c[v[1]].y - c[v[0]].y;
    const bx = c[v[2]].x - c[v[0]].x, by = c[v[2]].y - c[v[0]].y;
    if (ax * by - ay * bx <= 0) return;

    const sh = f.shade;
    if (glow && !fast) { ctx.shadowColor = color; ctx.shadowBlur = 10; }
    ctx.fillStyle = `rgb(${Math.floor(r*sh*pm)},${Math.floor(g*sh*pm)},${Math.floor(b*sh*pm)})`;
    ctx.beginPath();
    ctx.moveTo(c[v[0]].x, c[v[0]].y);
    for (let i = 1; i < v.length; i++) ctx.lineTo(c[v[i]].x, c[v[i]].y);
    ctx.closePath();
    ctx.fill();
    if (!fast) {
      ctx.strokeStyle = 'rgba(100,100,140,0.25)';
      ctx.lineWidth = 0.5;
      ctx.stroke();
    }
    ctx.shadowBlur = 0;
  });
}

function pctToColor(pct, hasDiscovery, isActive) {
  if (isActive && hasDiscovery) return '#ff9020';  // orange-gold: active + discovery
  if (isActive && pct > 0) return '#ff66ff';       // pink: active + in progress
  if (isActive) return '#ff44cc';                   // bright pink: active, fresh
  if (hasDiscovery) return '#ffd06a';               // gold: discovery, done
  if (pct >= 100) return '#a0f0ff';                 // cyan: complete
  if (pct > 0) {
    const t = pct / 100;
    const r = Math.floor(196 * (1-t) + 160 * t);
    const g = Math.floor(160 * (1-t) + 240 * t);
    return '#' + r.toString(16).padStart(2,'0') + g.toString(16).padStart(2,'0') + 'ff';
  }
  return '#1a1a2e';
}

let heatmapData = [];

async function updateSponge() {
  try {
    heatmapData = await fetch(API + '/progress/heatmap?blocks=' + mengerCubes.length).then(r => r.json());
  } catch(e) {}
}

function drawSponge() {
  if (renderMode === 'off') return;
  const ctx = sCtx;
  ctx.clearRect(0, 0, spongeCanvas.width, spongeCanvas.height);

  // FPS
  frameCount++;
  const now = performance.now();
  if (now - lastFrameTime >= 1000) {
    fpsDisplay = frameCount;
    frameCount = 0;
    lastFrameTime = now;
    document.getElementById('fps-counter').textContent = fpsDisplay + ' fps';
  }

  if (autoRotate) rotY += 0.003;

  // Sort cubes by depth (back to front)
  const sorted = mengerCubes.map((c, i) => {
    const r = rotate3D(c.x, c.y, c.z);
    return {...c, idx: i, depth: r.z};
  });
  sorted.sort((a, b) => a.depth - b.depth);

  if (renderMode === 'fast') {
    // Point cloud mode — one dot per cube
    const ps = currentDepth <= 2 ? 6 : currentDepth === 3 ? 3 : 1.5;
    sorted.forEach(cube => {
      const block = heatmapData[cube.idx] || {pct: 0, discoveries: 0, active: 0};
      const isActive = block.active > 0;
      let color = pctToColor(block.pct, block.discoveries > 0, isActive);
      if (color === '#1a1a2e') color = '#444466'; // pending: visible dim purple
      const p = project(cube.x, cube.y, cube.z);
      ctx.fillStyle = color;
      ctx.fillRect(p.x - ps/2, p.y - ps/2, ps, ps);
    });
  } else {
    sorted.forEach(cube => {
      const block = heatmapData[cube.idx] || {pct: 0, discoveries: 0, active: 0};
      const isActive = block.active > 0;
      const color = pctToColor(block.pct, block.discoveries > 0, isActive);
      const glow = isActive || block.discoveries > 0 || block.pct >= 100;
      const pulse = isActive;
      drawIsoCube(ctx, cube.x, cube.y, cube.z, cubeSize, color, glow, pulse);
    });
  }

  // Drag hint
  if (autoRotate) {
    ctx.fillStyle = 'rgba(85,85,104,0.4)';
    ctx.font = '11px monospace';
    ctx.fillText('drag to rotate \u00b7 double-click to auto-spin', spongeCanvas.width/2 - 130, spongeCanvas.height - 8);
  }

  animFrame = requestAnimationFrame(drawSponge);
}

updateSponge();
drawSponge();

update();
setInterval(update, 5000);
setInterval(updateSponge, 10000);
</script>
</body>
</html>
"""

# ═══════════════════════════════════════════════════════════
# Results Page
# ═══════════════════════════════════════════════════════════

@app.get("/results", response_class=HTMLResponse)
def results_page():
    return RESULTS_HTML

RESULTS_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>W@Home — Live Results</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: #0a0a12; color: #e0e0e8;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    line-height: 1.7;
  }
  a { color: #a78bfa; text-decoration: none; }
  a:hover { color: #c4b5fd; text-decoration: underline; }

  .header {
    background: rgba(255,255,255,0.03); border-bottom: 1px solid rgba(255,255,255,0.06);
    padding: 0.8rem 1.5rem; display: flex; align-items: center; justify-content: space-between;
  }
  .header h1 { font-size: 1.2rem; font-weight: 400; color: #a78bfa; }
  .header nav { display: flex; gap: 1rem; font-size: 0.9rem; }

  .container { max-width: 1100px; margin: 0 auto; padding: 2rem 1.5rem; }
  h2 { font-size: 1.8rem; font-weight: 300; margin-bottom: 1rem; color: #c4b5fd; letter-spacing: 0.02em; }
  h3 { color: #a78bfa; margin-bottom: 0.8rem; font-weight: 400; }
  .section { margin-bottom: 3rem; }

  /* Stats row */
  .stats-row {
    display: flex; justify-content: space-around; flex-wrap: wrap; gap: 1rem;
    background: rgba(124,58,237,0.08); border: 1px solid rgba(124,58,237,0.2);
    border-radius: 12px; padding: 1.5rem; margin-bottom: 2rem;
  }
  .stat { text-align: center; }
  .stat .num { font-size: 2rem; font-weight: 700; color: #a78bfa; }
  .stat .label { font-size: 0.75rem; color: #7070a0; text-transform: uppercase; letter-spacing: 0.1em; }

  /* Progress bar */
  .progress-wrap {
    background: rgba(255,255,255,0.03); border-radius: 8px; height: 24px;
    margin: 1rem 0; overflow: hidden; position: relative;
  }
  .progress-fill { height: 100%; border-radius: 8px; transition: width 1s; }
  .progress-text { position: absolute; right: 8px; top: 2px; font-size: 0.8rem; color: #e0e0f0; }

  /* Tabs */
  .tab-bar {
    display: flex; gap: 0; margin-bottom: 2rem; border-bottom: 2px solid rgba(255,255,255,0.06);
  }
  .tab-btn {
    padding: 0.8rem 1.5rem; background: none; border: none; color: #7070a0;
    font-size: 0.95rem; cursor: pointer; font-family: inherit; position: relative;
    transition: color 0.2s;
  }
  .tab-btn:hover { color: #c0c0d8; }
  .tab-btn.active { color: #e0e0f0; }
  .tab-btn.active::after {
    content: ''; position: absolute; bottom: -2px; left: 0; right: 0;
    height: 2px; border-radius: 1px;
  }
  .tab-btn[data-tab="eigenvalue"].active::after { background: #a78bfa; }
  .tab-btn[data-tab="falsification"].active::after { background: #22c55e; }
  .tab-btn[data-tab="clock"].active::after { background: #6dd5ed; }

  .tab-indicator {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 0.5em;
  }

  .tab-panel { display: none; }
  .tab-panel.active { display: block; }

  /* Per-type progress bars */
  .type-progress-row {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 2rem;
  }
  .type-card {
    background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px; padding: 1rem; text-align: center;
  }
  .type-card .type-name { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.5em; }
  .type-card .type-num { font-size: 1.6rem; font-weight: 700; }
  .type-card .type-sub { font-size: 0.75rem; color: #7070a0; margin-top: 0.3em; }
  .type-bar { height: 6px; background: rgba(255,255,255,0.05); border-radius: 3px; margin-top: 0.5em; overflow: hidden; }
  .type-bar-fill { height: 100%; border-radius: 3px; transition: width 1s; }

  /* Tables */
  table {
    width: 100%; border-collapse: collapse; font-size: 0.9rem;
    background: rgba(255,255,255,0.02); border-radius: 8px; overflow: hidden;
  }
  th {
    background: rgba(124,58,237,0.15); color: #c4b5fd; padding: 0.7rem 1rem;
    text-align: left; font-weight: 500; font-size: 0.8rem;
    text-transform: uppercase; letter-spacing: 0.05em;
  }
  td { padding: 0.6rem 1rem; border-bottom: 1px solid rgba(255,255,255,0.04); color: #b0b0c8; }
  tr:hover td { background: rgba(124,58,237,0.05); }
  tr.menger td { color: #a0f0ff !important; font-weight: 600; background: rgba(160,240,255,0.05); }
  tr.menger td .badge {
    display: inline-block; background: #22c55e; color: #000; padding: 0.1rem 0.5rem;
    border-radius: 4px; font-size: 0.7rem; font-weight: 700; margin-left: 0.5rem;
  }

  /* Discovery cards */
  .disc-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 1rem;
  }
  .disc-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px; padding: 1.2rem;
  }
  .disc-card .const-name { color: #a78bfa; font-weight: 600; font-size: 1.1rem; }
  .disc-card .lambda { color: #6dd5ed; font-family: monospace; }
  .disc-card .worker { color: #7070a0; font-size: 0.85rem; }
  .disc-card .verified { color: #22c55e; }
  .disc-card .unverified { color: #ffd06b; }

  .lb-table .online { color: #22c55e; }
  .lb-table .offline { color: #505070; }

  .conclusion {
    background: rgba(34,197,94,0.08); border: 1px solid rgba(34,197,94,0.2);
    border-radius: 12px; padding: 1.5rem; margin-top: 1.5rem;
  }
  .conclusion h3 { color: #22c55e; }
  .conclusion p { color: #b0b0c8; font-size: 0.95rem; margin-top: 0.5rem; }
  .conclusion .key { color: #e0e0f0; font-weight: 500; }

  /* Canvas containers */
  .viz-wrap {
    background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px; padding: 1rem; margin-bottom: 1.5rem;
  }
  canvas { max-width: 100%; display: block; margin: 0 auto; }

  footer {
    text-align: center; padding: 2rem; color: #505070; font-size: 0.85rem;
    border-top: 1px solid rgba(255,255,255,0.05);
  }

  @media (max-width: 600px) {
    .container { padding: 1rem; }
    h2 { font-size: 1.4rem; }
    .stat .num { font-size: 1.4rem; }
    .disc-grid { grid-template-columns: 1fr; }
    .type-progress-row { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>

<div class="header">
  <h1><a href="/" style="color:#a78bfa">W@Home</a> &mdash; Results</h1>
  <nav>
    <a href="/">Home</a>
    <a href="/dashboard">Dashboard</a>
    <a href="/chat">Chat</a>
    <a href="/economy">W Economy</a>
    <a href="https://akataleptos.com" target="_blank">Theory</a>
  </nav>
</div>

<div class="container">

  <!-- Global Stats -->
  <div class="section">
    <h2>Live Progress</h2>
    <div class="stats-row">
      <div class="stat"><div class="num" id="sWorkers">-</div><div class="label">Active Workers</div></div>
      <div class="stat"><div class="num" id="sJobs">-</div><div class="label">Jobs Completed</div></div>
      <div class="stat"><div class="num" id="sHits">-</div><div class="label">Hits Found</div></div>
      <div class="stat"><div class="num" id="sConfirmed">-</div><div class="label">Confirmed</div></div>
      <div class="stat"><div class="num" id="sHours">-</div><div class="label">Compute Hours</div></div>
    </div>

    <!-- Per-type progress cards -->
    <div class="type-progress-row">
      <div class="type-card">
        <div class="type-name" style="color:#a78bfa;">Eigenvalue Search</div>
        <div class="type-num" style="color:#a78bfa;" id="tEigPct">0%</div>
        <div class="type-sub"><span id="tEigDone">0</span> jobs &middot; <span id="tEigHrs">0</span>h compute</div>
        <div class="type-bar"><div class="type-bar-fill" id="tEigBar" style="width:0%; background: linear-gradient(90deg, #7c3aed, #6dd5ed);"></div></div>
      </div>
      <div class="type-card">
        <div class="type-name" style="color:#22c55e;">Fractal Falsification</div>
        <div class="type-num" style="color:#22c55e;" id="tFalsPct">0%</div>
        <div class="type-sub"><span id="tFalsDone">0</span> jobs &middot; <span id="tFalsHrs">0</span>h compute</div>
        <div class="type-bar"><div class="type-bar-fill" id="tFalsBar" style="width:0%; background: linear-gradient(90deg, #22c55e, #ffd06a);"></div></div>
      </div>
      <div class="type-card">
        <div class="type-name" style="color:#6dd5ed;">Countdown Clock</div>
        <div class="type-num" style="color:#6dd5ed;" id="tClockPct">0%</div>
        <div class="type-sub"><span id="tClockDone">0</span> jobs &middot; <span id="tClockHrs">0</span>h compute</div>
        <div class="type-bar"><div class="type-bar-fill" id="tClockBar" style="width:0%; background: linear-gradient(90deg, #6dd5ed, #e0e0f0);"></div></div>
      </div>
    </div>

    <div class="progress-wrap">
      <div class="progress-fill" id="progBar" style="width:0%; background: linear-gradient(90deg, #7c3aed, #6dd5ed);"></div>
      <div class="progress-text" id="progText">0%</div>
    </div>
    <p style="color:#7070a0;font-size:0.85rem;">Total progress across all work types.</p>
  </div>

  <!-- Tab Navigation -->
  <div class="tab-bar">
    <button class="tab-btn active" data-tab="eigenvalue" onclick="switchTab('eigenvalue')">
      <span class="tab-indicator" style="background:#a78bfa;"></span>Eigenvalue Search
    </button>
    <button class="tab-btn" data-tab="falsification" onclick="switchTab('falsification')">
      <span class="tab-indicator" style="background:#22c55e;"></span>Fractal Falsification
    </button>
    <button class="tab-btn" data-tab="clock" onclick="switchTab('clock')">
      <span class="tab-indicator" style="background:#6dd5ed;"></span>Countdown Clock
    </button>
  </div>

  <!-- ═══ Tab 1: Eigenvalue Search ═══ -->
  <div class="tab-panel active" id="panel-eigenvalue">
    <div class="section">
      <h2>Eigenvalue Search</h2>
      <p style="color:#9090b0;margin-bottom:1rem;">Sweeping lambda from 0.4 to 0.6 in 200,000 steps. Each step: build the W-operator on a Menger sponge contact graph, solve eigenvalues, compare ratios against known physical constants.</p>
    </div>

    <div class="section">
      <h3>Recent Discoveries</h3>
      <div class="disc-grid" id="discGrid">
        <div style="color:#505070;">Loading...</div>
      </div>
    </div>

    <div class="section">
      <h3>Leaderboard</h3>
      <table class="lb-table">
        <thead><tr><th>#</th><th>Worker</th><th>Jobs</th><th>Hours</th><th>Hits</th><th>W</th><th>Trust</th><th>Status</th></tr></thead>
        <tbody id="lbBody"><tr><td colspan="8" style="color:#505070;">Loading...</td></tr></tbody>
      </table>
    </div>
  </div>

  <!-- ═══ Tab 2: Fractal Falsification ═══ -->
  <div class="tab-panel" id="panel-falsification">
    <div class="section">
      <h2>Fractal Falsification</h2>
      <p style="color:#9090b0;margin-bottom:1rem;">
        If the Menger sponge's physical constant predictions are mere numerology, other fractals should
        produce similar matches. We test <strong>every</strong> threshold-based 3D fractal from base 3 to base 21
        (odd bases only). Each is defined by removing subcubes with &ge;P center coordinates from a b&times;b&times;b grid.
      </p>

      <div class="viz-wrap">
        <canvas id="falsCanvas" width="700" height="300"></canvas>
      </div>

      <table>
        <thead><tr><th>Base (b)</th><th>Threshold (P)</th><th>S</th><th>&Delta;</th><th>Kept (k)</th><th>Removed (r)</th><th>Constants Matched</th></tr></thead>
        <tbody id="falsBody"><tr><td colspan="7" style="color:#505070;">Computing...</td></tr></tbody>
      </table>

      <div class="conclusion" id="falsConclusion" style="display:none;">
        <h3>Conclusion</h3>
        <p>
          Out of <span class="key" id="cTotal">?</span> threshold-based fractal configurations tested,
          <span class="key">only the Menger sponge</span> (b=3, P=2) matches <span class="key" id="cMenger">?</span>
          of 13 physical constants. The next best scores <span class="key" id="cNext">?</span>/13.
          The parameters are <em>fixed by the removal rule</em>, not fitted to data.
        </p>
        <p style="margin-top:0.8rem;">
          Probability by chance: ~1% per formula &times; 12 simultaneous matches gives
          p &lt; 10<sup>-20</sup>. The Menger sponge is not a coincidence.
        </p>
      </div>
    </div>
  </div>

  <!-- ═══ Tab 3: Countdown Clock ═══ -->
  <div class="tab-panel" id="panel-clock">
    <div class="section">
      <h2>Menger Countdown Clock</h2>
      <p style="color:#9090b0;margin-bottom:1rem;">
        If physical constants come from Menger spectral invariants, the residuals between predicted and
        measured values decrease geometrically with iteration depth. The convergence ratio P/k = 2/20 = 0.1
        means each Menger level refines predictions by 10&times;.
      </p>

      <div class="viz-wrap" style="display:flex;gap:1rem;flex-wrap:wrap;">
        <div style="flex:1;min-width:300px;">
          <div style="color:#6dd5ed;font-size:0.85rem;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5em;">Spectral Dimension Convergence</div>
          <canvas id="clockCanvas1" width="500" height="280"></canvas>
        </div>
        <div style="flex:1;min-width:300px;">
          <div style="color:#ffd06a;font-size:0.85rem;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5em;">Fine Structure Residual (log scale)</div>
          <canvas id="clockCanvas2" width="500" height="280"></canvas>
        </div>
      </div>

      <div style="margin-top:1rem;">
        <h3>Multiplicity Tower</h3>
        <p style="color:#9090b0;margin-bottom:0.8rem;">
          Eigenvalue-2 multiplicity: m(n) = (18<sup>n</sup> + 153&middot;4<sup>n</sup> + 1155) / 357
        </p>
        <table>
          <thead><tr><th>Level</th><th>Multiplicity</th><th>d<sub>S</sub></th><th>Gap Ratio</th><th>Residual vs d<sub>H</sub></th></tr></thead>
          <tbody id="clockBody"><tr><td colspan="5" style="color:#505070;">Loading...</td></tr></tbody>
        </table>
      </div>

      <div class="conclusion" id="clockConclusion" style="display:none;">
        <h3>What This Means</h3>
        <p id="clockConcText"></p>
      </div>
    </div>
  </div>

</div>

<footer>
  <p>W@Home is part of the <a href="https://akataleptos.com">Akataleptos Project</a></p>
  <p style="margin-top:0.3rem;">Results update every 10 seconds. All data is public.</p>
  <p style="margin-top:0.8rem; font-size:0.7rem; color:#404060; font-style:italic;">&dagger; Disclaimer: The observer may be affecting results. This is not a bug&mdash;it&rsquo;s the thesis. See &lambda;.</p>
</footer>

<script>
function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }
let currentTab = 'eigenvalue';

function switchTab(name) {
  currentTab = name;
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === name));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.toggle('active', p.id === 'panel-' + name));
}

// ── Live progress ──
async function updateProgress() {
  try {
    const [pr, ar] = await Promise.all([
      fetch('/progress').then(r => r.json()),
      fetch('/active').then(r => r.json()),
    ]);
    document.getElementById('sWorkers').textContent = ar.length || '0';
    document.getElementById('sJobs').textContent = (pr.completed || 0).toLocaleString();
    document.getElementById('sHits').textContent = pr.total_discoveries || '0';
    document.getElementById('sConfirmed').textContent = pr.total_confirmed || '0';
    document.getElementById('sHours').textContent = pr.total_compute_hours || '0';
    const pct = (pr.percent_complete || 0).toFixed(2);
    document.getElementById('progBar').style.width = pct + '%';
    document.getElementById('progText').textContent = pct + '%';

    // Per-type cards
    if (pr.per_type) {
      const pt = pr.per_type;
      if (pt.eigenvalue) {
        document.getElementById('tEigPct').textContent = pt.eigenvalue.pct + '%';
        document.getElementById('tEigDone').textContent = pt.eigenvalue.completed.toLocaleString();
        document.getElementById('tEigHrs').textContent = pt.eigenvalue.compute_hours;
        document.getElementById('tEigBar').style.width = pt.eigenvalue.pct + '%';
      }
      if (pt.falsification) {
        document.getElementById('tFalsPct').textContent = pt.falsification.pct + '%';
        document.getElementById('tFalsDone').textContent = pt.falsification.completed.toLocaleString();
        document.getElementById('tFalsHrs').textContent = pt.falsification.compute_hours;
        document.getElementById('tFalsBar').style.width = pt.falsification.pct + '%';
      }
      if (pt.clock) {
        document.getElementById('tClockPct').textContent = pt.clock.pct + '%';
        document.getElementById('tClockDone').textContent = pt.clock.completed.toLocaleString();
        document.getElementById('tClockHrs').textContent = pt.clock.compute_hours;
        document.getElementById('tClockBar').style.width = pt.clock.pct + '%';
      }
    }
  } catch(e) {}
}

// ── Discoveries ──
async function updateDiscoveries() {
  try {
    const disc = await fetch('/discoveries').then(r => r.json());
    const grid = document.getElementById('discGrid');
    grid.innerHTML = '';
    if (!disc.length) {
      grid.innerHTML = '<div style="color:#505070;">No discoveries yet. Keep computing!</div>';
      return;
    }
    disc.slice(0, 20).forEach(d => {
      const card = document.createElement('div');
      card.className = 'disc-card';
      const vClass = d.job_verified ? 'verified' : 'unverified';
      const vText = d.job_verified ? 'Confirmed' : 'Pending';
      const tier = d.param_tier === 'mobile' ? ' (scout)' : '';
      card.innerHTML = '<div class="const-name">' + esc(d.constant_name || '?') + tier + '</div>' +
        '<div class="lambda">lambda = ' + (d.lambda_val || 0).toFixed(6) + '</div>' +
        '<div>Ratio: ' + (d.ratio_value || 0).toFixed(8) + '</div>' +
        '<div class="worker">Found by ' + esc(d.worker_name || 'anonymous') + '</div>' +
        '<div class="' + vClass + '">' + vText + '</div>';
      grid.appendChild(card);
    });
  } catch(e) {}
}

// ── Leaderboard ──
async function updateLeaderboard() {
  try {
    const lb = await fetch('/leaderboard').then(r => r.json());
    const body = document.getElementById('lbBody');
    body.innerHTML = '';
    lb.forEach(w => {
      const tr = document.createElement('tr');
      const status = w.online ? '<span class="online">online</span>' : '<span class="offline">offline</span>';
      tr.innerHTML = '<td>' + w.rank + '</td><td>' + esc(w.name) + '</td>' +
        '<td>' + w.jobs + '</td><td>' + w.compute_hours + '</td>' +
        '<td>' + (w.discoveries || 0) + '</td><td style="color:#ffd06a;">' + (w.w_balance || 0).toFixed(2) + '</td><td>' + w.trust + '</td><td>' + status + '</td>';
      body.appendChild(tr);
    });
    if (!lb.length) body.innerHTML = '<tr><td colspan="8" style="color:#505070;">No workers yet</td></tr>';
  } catch(e) {}
}

// ── Falsification (table + animated bar chart) ──
let falsData = null;
async function loadFalsification() {
  try {
    falsData = await fetch('/api/falsification').then(r => r.json());
    const body = document.getElementById('falsBody');
    body.innerHTML = '';
    let nextBest = 0;
    falsData.results.forEach(r => {
      const tr = document.createElement('tr');
      if (r.is_menger) tr.className = 'menger';
      if (!r.is_menger && r.n_matched > nextBest) nextBest = r.n_matched;
      const matchCell = r.is_menger
        ? r.n_matched + '/13 <span class="badge">MENGER</span>'
        : r.n_matched + '/13';
      const matchNames = r.matches.map(m => m.name).join(', ');
      tr.innerHTML = '<td>' + r.b + '</td><td>' + r.P + '</td><td>' + r.S + '</td>' +
        '<td>' + r.Delta + '</td><td>' + r.k + '</td><td>' + r.r + '</td>' +
        '<td>' + matchCell + (matchNames ? '<br><span style="font-size:0.75rem;color:#7070a0;">' + matchNames + '</span>' : '') + '</td>';
      body.appendChild(tr);
    });

    const conc = document.getElementById('falsConclusion');
    conc.style.display = 'block';
    document.getElementById('cTotal').textContent = falsData.total_configs;
    document.getElementById('cMenger').textContent = falsData.menger_match;
    document.getElementById('cNext').textContent = nextBest;

    drawFalsChart();
  } catch(e) { console.error('Falsification error:', e); }
}

function drawFalsChart() {
  if (!falsData) return;
  const c = document.getElementById('falsCanvas');
  const ctx = c.getContext('2d');
  ctx.clearRect(0, 0, c.width, c.height);

  const data = falsData.results;
  const n = data.length;
  const pad = {l: 50, r: 20, t: 20, b: 40};
  const w = c.width - pad.l - pad.r;
  const h = c.height - pad.t - pad.b;
  const barW = Math.max(4, Math.floor(w / n) - 2);

  // Axes
  ctx.strokeStyle = 'rgba(255,255,255,0.1)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, pad.t + h); ctx.lineTo(pad.l + w, pad.t + h);
  ctx.stroke();

  // Y labels
  ctx.fillStyle = '#7070a0'; ctx.font = '11px system-ui';
  for (let y = 0; y <= 13; y += 3) {
    const py = pad.t + h - (y / 13) * h;
    ctx.fillText(y, pad.l - 20, py + 4);
    ctx.strokeStyle = 'rgba(255,255,255,0.04)';
    ctx.beginPath(); ctx.moveTo(pad.l, py); ctx.lineTo(pad.l + w, py); ctx.stroke();
  }

  // Bars
  data.forEach((r, i) => {
    const x = pad.l + i * (w / n) + (w / n - barW) / 2;
    const bh = (r.n_matched / 13) * h;
    const y = pad.t + h - bh;

    if (r.is_menger) {
      ctx.fillStyle = '#a0f0ff';
      ctx.shadowColor = '#a0f0ff'; ctx.shadowBlur = 12;
    } else if (r.n_matched > 0) {
      ctx.fillStyle = '#22c55e';
      ctx.shadowBlur = 0;
    } else {
      ctx.fillStyle = '#2a2a3e';
      ctx.shadowBlur = 0;
    }

    ctx.fillRect(x, y, barW, bh);
    ctx.shadowBlur = 0;
  });

  // Label Menger bar
  const mengerIdx = data.findIndex(r => r.is_menger);
  if (mengerIdx >= 0) {
    const mx = pad.l + mengerIdx * (w / n) + (w / n) / 2;
    ctx.fillStyle = '#a0f0ff'; ctx.font = 'bold 11px system-ui';
    ctx.textAlign = 'center';
    ctx.fillText('MENGER', mx, pad.t + 12);
    ctx.textAlign = 'start';
  }

  // X label
  ctx.fillStyle = '#7070a0'; ctx.font = '11px system-ui';
  ctx.textAlign = 'center';
  ctx.fillText('Fractal configurations (base 3-21, threshold 1-3)', pad.l + w/2, c.height - 5);
  ctx.textAlign = 'start';
}

// ── Clock (convergence curves) ──
let clockData = null;
async function loadClock() {
  try {
    clockData = await fetch('/api/clock').then(r => r.json());
    drawClockCharts();

    // Fill table
    const body = document.getElementById('clockBody');
    body.innerHTML = '';
    const dS = clockData.spectral_dim_convergence;
    const gaps = clockData.gap_ratio_convergence;
    const mults = clockData.multiplicity_tower;

    for (let i = 0; i < dS.length; i++) {
      const tr = document.createElement('tr');
      const m = mults[i] || {};
      const g = gaps[i] || {};
      const pred = dS[i].predicted ? ' <span style="color:#ffd06a;font-size:0.7rem;">(predicted)</span>' : '';
      tr.innerHTML = '<td>L' + dS[i].level + pred + '</td>' +
        '<td>' + (m.multiplicity || '?') + '</td>' +
        '<td>' + dS[i].d_S + '</td>' +
        '<td>' + (g.ratio || '?') + '</td>' +
        '<td>' + dS[i].residual + '</td>';
      if (dS[i].predicted) tr.style.opacity = '0.7';
      body.appendChild(tr);
    }

    const conc = document.getElementById('clockConclusion');
    conc.style.display = 'block';
    document.getElementById('clockConcText').innerHTML = clockData.conclusion +
      ' <span style="color:#6dd5ed;">Convergence ratio: ' + clockData.target_ratio + '</span>';
  } catch(e) { console.error('Clock error:', e); }
}

function drawClockCharts() {
  if (!clockData) return;
  drawSpectralDimChart();
  drawResidualChart();
}

function drawSpectralDimChart() {
  const c = document.getElementById('clockCanvas1');
  const ctx = c.getContext('2d');
  ctx.clearRect(0, 0, c.width, c.height);

  const pad = {l: 50, r: 30, t: 20, b: 40};
  const w = c.width - pad.l - pad.r;
  const h = c.height - pad.t - pad.b;

  const dS = clockData.spectral_dim_convergence;
  const dH = clockData.hausdorff_dim;

  // Y range: 1.0 to 3.0
  const yMin = 1.0, yMax = 3.0;
  function yPos(v) { return pad.t + h - ((v - yMin) / (yMax - yMin)) * h; }
  function xPos(level) { return pad.l + ((level - 1) / 4) * w; }

  // Grid
  ctx.strokeStyle = 'rgba(255,255,255,0.06)'; ctx.lineWidth = 1;
  for (let y = 1.0; y <= 3.0; y += 0.5) {
    const py = yPos(y);
    ctx.beginPath(); ctx.moveTo(pad.l, py); ctx.lineTo(pad.l + w, py); ctx.stroke();
    ctx.fillStyle = '#7070a0'; ctx.font = '10px system-ui';
    ctx.fillText(y.toFixed(1), pad.l - 30, py + 4);
  }

  // d_H limit line
  ctx.strokeStyle = 'rgba(109,213,237,0.4)'; ctx.lineWidth = 1;
  ctx.setLineDash([6, 4]);
  ctx.beginPath();
  ctx.moveTo(pad.l, yPos(dH)); ctx.lineTo(pad.l + w, yPos(dH));
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#6dd5ed'; ctx.font = '10px system-ui';
  ctx.fillText('d_H = ' + dH.toFixed(3), pad.l + w - 70, yPos(dH) - 5);

  // Data points and curve
  ctx.strokeStyle = '#6dd5ed'; ctx.lineWidth = 2;
  ctx.beginPath();
  dS.forEach((pt, i) => {
    const x = xPos(pt.level);
    const y = yPos(pt.d_S);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Points
  dS.forEach(pt => {
    const x = xPos(pt.level); const y = yPos(pt.d_S);
    ctx.beginPath(); ctx.arc(x, y, pt.predicted ? 4 : 6, 0, Math.PI * 2);
    if (pt.predicted) {
      ctx.strokeStyle = '#ffd06a'; ctx.lineWidth = 2; ctx.stroke();
    } else {
      ctx.fillStyle = '#6dd5ed'; ctx.fill();
    }
    ctx.fillStyle = '#e0e0f0'; ctx.font = '11px system-ui';
    ctx.fillText(pt.d_S.toFixed(2), x + 8, y - 5);
  });

  // X labels
  ctx.fillStyle = '#7070a0'; ctx.font = '11px system-ui'; ctx.textAlign = 'center';
  for (let l = 1; l <= 4; l++) {
    ctx.fillText('L' + l, xPos(l), c.height - 8);
  }
  ctx.textAlign = 'start';
}

function drawResidualChart() {
  const c = document.getElementById('clockCanvas2');
  const ctx = c.getContext('2d');
  ctx.clearRect(0, 0, c.width, c.height);

  const pad = {l: 50, r: 30, t: 20, b: 40};
  const w = c.width - pad.l - pad.r;
  const h = c.height - pad.t - pad.b;

  const residuals = clockData.alpha_residuals;

  // Log scale Y: 0.01 to 15
  const yMin = -2, yMax = 1.2; // log10 scale
  function yPos(logV) { return pad.t + h - ((logV - yMin) / (yMax - yMin)) * h; }
  function xPos(level) { return pad.l + ((level - 1) / 4) * w; }

  // Grid
  ctx.strokeStyle = 'rgba(255,255,255,0.06)'; ctx.lineWidth = 1;
  [-2, -1, 0, 1].forEach(exp => {
    const py = yPos(exp);
    ctx.beginPath(); ctx.moveTo(pad.l, py); ctx.lineTo(pad.l + w, py); ctx.stroke();
    ctx.fillStyle = '#7070a0'; ctx.font = '10px system-ui';
    ctx.fillText(Math.pow(10, exp).toFixed(exp < 0 ? -exp : 0) + '%', pad.l - 42, py + 4);
  });

  // P/k = 0.1 decay guide line (dashed)
  ctx.strokeStyle = 'rgba(255,208,106,0.3)'; ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(xPos(1), yPos(Math.log10(residuals[0].residual_pct)));
  for (let l = 2; l <= 4; l++) {
    const pred = residuals[0].residual_pct * Math.pow(0.1, l - 1);
    ctx.lineTo(xPos(l), yPos(Math.log10(pred)));
  }
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#ffd06a'; ctx.font = '9px system-ui';
  ctx.fillText('P/k = 0.1 decay', pad.l + w - 80, yPos(yMax) + 15);

  // Data curve
  ctx.strokeStyle = '#ffd06a'; ctx.lineWidth = 2;
  ctx.beginPath();
  residuals.forEach((r, i) => {
    const x = xPos(r.level);
    const y = yPos(Math.log10(r.residual_pct));
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Points
  residuals.forEach(r => {
    const x = xPos(r.level);
    const y = yPos(Math.log10(r.residual_pct));
    const isPred = r.level === 4;
    ctx.beginPath(); ctx.arc(x, y, isPred ? 4 : 6, 0, Math.PI * 2);
    if (isPred) {
      ctx.strokeStyle = '#ffd06a'; ctx.lineWidth = 2; ctx.stroke();
    } else {
      ctx.fillStyle = '#ffd06a'; ctx.fill();
    }
    ctx.fillStyle = '#e0e0f0'; ctx.font = '11px system-ui';
    ctx.fillText(r.residual_pct + '%', x + 8, y - 5);
  });

  // X labels
  ctx.fillStyle = '#7070a0'; ctx.font = '11px system-ui'; ctx.textAlign = 'center';
  for (let l = 1; l <= 4; l++) ctx.fillText('L' + l, xPos(l), c.height - 8);
  ctx.textAlign = 'start';
}

// Init
updateProgress();
updateDiscoveries();
updateLeaderboard();
loadFalsification();
loadClock();
setInterval(updateProgress, 10000);
setInterval(updateDiscoveries, 30000);
setInterval(updateLeaderboard, 60000);
</script>
</body>
</html>
"""

# ═══════════════════════════════════════════════════════════
# Economy Page
# ═══════════════════════════════════════════════════════════

ECONOMY_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>W@Home — W Economy</title>
<style>
  :root {
    --bg: #0a0a10; --bg2: #12121e; --bg3: #1a1a2e;
    --cyan: #a0f0ff; --gold: #ffd06a; --violet: #c4a0ff;
    --green: #80ffaa; --red: #ff6b6b; --text: #c8c8d8;
    --dim: #555568; --border: #2a2a3a; --mono: 'JetBrains Mono', monospace;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: var(--bg); color: var(--text); font-family: var(--mono); min-height: 100vh; padding: 1.5em; }
  .header { text-align: center; padding: 2em 0 1em; border-bottom: 1px solid var(--border); margin-bottom: 2em; }
  .header h1 { color: var(--gold); font-size: 1.8em; letter-spacing: 0.08em; }
  .header nav { margin-top: 0.5em; }
  .header nav a { color: var(--violet); text-decoration: none; margin: 0 0.5em; font-size: 0.85em; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.2em; max-width: 1200px; margin: 0 auto; }
  .card { background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; padding: 1.3em; }
  .card h2 { color: var(--gold); font-size: 0.85em; letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 1em; }
  .big-num { font-size: 2.2em; color: var(--gold); font-weight: bold; }
  .big-num .unit { font-size: 0.4em; color: var(--dim); margin-left: 0.3em; }
  .stat-row { display: flex; justify-content: space-between; padding: 0.4em 0; border-bottom: 1px solid var(--border); font-size: 0.85em; }
  .stat-label { color: var(--dim); }
  .stat-val { color: var(--cyan); }
  table { width: 100%; border-collapse: collapse; font-size: 0.8em; }
  th { text-align: left; color: var(--dim); padding: 0.5em; border-bottom: 1px solid var(--border); }
  td { padding: 0.5em; border-bottom: 1px solid rgba(42,42,58,0.5); }
  .mint { color: var(--green); }
  .transfer { color: var(--cyan); }
  .stake-tx { color: var(--violet); }
  .stake-bar { height: 24px; border-radius: 4px; margin: 0.3em 0; display: flex; overflow: hidden; }
  .stake-bar div { height: 100%; display: flex; align-items: center; justify-content: center; font-size: 0.7em; color: #000; font-weight: bold; }
  .chain-block { background: var(--bg3); border: 1px solid var(--border); border-radius: 6px; padding: 0.8em; margin-bottom: 0.5em; }
  .chain-block .hash { color: var(--dim); font-size: 0.7em; word-break: break-all; }
  .rewards-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.8em; }
  .reward-card { background: var(--bg3); border-radius: 6px; padding: 1em; text-align: center; }
  .reward-card .amount { font-size: 1.8em; color: var(--gold); }
  .reward-card .type { color: var(--dim); font-size: 0.75em; text-transform: uppercase; margin-top: 0.3em; }
</style>
</head>
<body>

<div class="header">
  <h1>W Economy</h1>
  <div style="color: var(--dim); font-size: 0.85em; margin-top: 0.3em;">Proof-of-Useful-Computation Currency</div>
  <nav>
    <a href="/">Home</a>
    <a href="/dashboard">Dashboard</a>
    <a href="/results">Results</a>
    <a href="/chat">Chat</a>
  </nav>
</div>

<div class="grid">

  <!-- Supply Overview -->
  <div class="card">
    <h2>Supply</h2>
    <div class="big-num" id="totalSupply">0<span class="unit">W</span></div>
    <div class="stat-row"><span class="stat-label">Circulating</span><span class="stat-val" id="circulating">0</span></div>
    <div class="stat-row"><span class="stat-label">Total Staked</span><span class="stat-val" id="totalStaked">0</span></div>
    <div class="stat-row"><span class="stat-label">Total Transferred</span><span class="stat-val" id="totalTransferred">0</span></div>
    <div class="stat-row"><span class="stat-label">Active Wallets</span><span class="stat-val" id="activeWallets">0</span></div>
    <div class="stat-row"><span class="stat-label">Transactions</span><span class="stat-val" id="totalTxns">0</span></div>
  </div>

  <!-- Reward Rates -->
  <div class="card">
    <h2>Mining Rewards</h2>
    <div class="rewards-grid">
      <div class="reward-card" style="border-left: 3px solid var(--violet);">
        <div class="amount" id="rEig">1.0</div>
        <div class="type">Eigenvalue</div>
      </div>
      <div class="reward-card" style="border-left: 3px solid var(--green);">
        <div class="amount" id="rFals">2.0</div>
        <div class="type">Falsification</div>
      </div>
      <div class="reward-card" style="border-left: 3px solid var(--cyan);">
        <div class="amount" id="rClock">5.0</div>
        <div class="type">Clock</div>
      </div>
    </div>
    <div style="margin-top: 1em; font-size: 0.75em; color: var(--dim);">
      Base rates &times; trust score. First verifier gets +0.5W bonus. Mobile &times;0.5.
    </div>
  </div>

  <!-- Blockchain -->
  <div class="card">
    <h2>Blockchain</h2>
    <div class="stat-row"><span class="stat-label">Chain Height</span><span class="stat-val" id="chainHeight">0</span></div>
    <div class="stat-row"><span class="stat-label">Latest Hash</span><span class="stat-val" id="latestHash" style="font-size:0.7em;">genesis</span></div>
    <div style="margin-top: 1em;" id="blocksContainer"></div>
  </div>

  <!-- Staking Breakdown -->
  <div class="card">
    <h2>Staking</h2>
    <div class="stake-bar" id="stakeBar">
      <div style="background: var(--violet); width: 33%;">Eigen</div>
      <div style="background: var(--green); width: 33%;">Falsif</div>
      <div style="background: var(--cyan); width: 34%;">Clock</div>
    </div>
    <div id="stakeDetails"></div>
  </div>

  <!-- Energy Footprint -->
  <div class="card" style="grid-column: 1 / -1;">
    <h2>Energy Footprint</h2>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1em; margin-bottom: 1em;">
      <div style="text-align: center;">
        <div class="big-num" style="font-size: 1.6em; color: var(--green);" id="eKwh">0<span class="unit">kWh</span></div>
        <div style="color: var(--dim); font-size: 0.75em;">Total Energy Used</div>
      </div>
      <div style="text-align: center;">
        <div class="big-num" style="font-size: 1.6em; color: var(--cyan);" id="eCo2">0<span class="unit">kg CO₂</span></div>
        <div style="color: var(--dim); font-size: 0.75em;">Carbon Footprint</div>
      </div>
      <div style="text-align: center;">
        <div class="big-num" style="font-size: 1.6em; color: var(--violet);" id="eEff">0<span class="unit">W/kWh</span></div>
        <div style="color: var(--dim); font-size: 0.75em;">Mining Efficiency</div>
      </div>
      <div style="text-align: center;">
        <div class="big-num" style="font-size: 1.6em; color: var(--gold);" id="eBtc">0<span class="unit">BTC tx</span></div>
        <div style="color: var(--dim); font-size: 0.75em;">Bitcoin Equivalent</div>
      </div>
    </div>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1em; font-size: 0.8em;">
      <div>
        <div class="stat-row"><span class="stat-label">Coal equivalent</span><span class="stat-val" id="eCoal">0 kg</span></div>
        <div class="stat-row"><span class="stat-label">Verified computations</span><span class="stat-val" id="eVerified">0</span></div>
        <div class="stat-row"><span class="stat-label">kWh per verification</span><span class="stat-val" id="ePerVer">0</span></div>
      </div>
      <div>
        <div class="stat-row"><span class="stat-label">Discoveries per kWh</span><span class="stat-val" id="eDiscKwh">0</span></div>
        <div class="stat-row"><span class="stat-label">vs Bitcoin per-tx</span><span class="stat-val" id="eBtcRatio" style="color: var(--green);">--</span></div>
        <div class="stat-row"><span class="stat-label">Scientific output</span><span class="stat-val" style="color: var(--green);">100%</span></div>
      </div>
    </div>
    <div style="margin-top: 1em; padding: 0.8em; background: rgba(128,255,170,0.06); border: 1px solid rgba(128,255,170,0.15); border-radius: 6px; font-size: 0.75em; color: var(--dim);">
      Every joule powers real spectral analysis of the Menger sponge topology. Zero energy wasted on meaningless hash puzzles. Estimates based on device TDP under compute load.
    </div>
  </div>

  <!-- Top Holders -->
  <div class="card" style="grid-column: 1 / -1;">
    <h2>Top Holders</h2>
    <table>
      <thead><tr><th>#</th><th>Worker</th><th>Balance</th><th>Total Earned</th></tr></thead>
      <tbody id="holdersBody"><tr><td colspan="4" style="color:var(--dim);">Loading...</td></tr></tbody>
    </table>
  </div>

  <!-- Recent Transactions -->
  <div class="card" style="grid-column: 1 / -1;">
    <h2>Transaction Feed</h2>
    <table>
      <thead><tr><th>Type</th><th>From</th><th>To</th><th>Amount</th><th>Memo</th><th>Block</th></tr></thead>
      <tbody id="txBody"><tr><td colspan="6" style="color:var(--dim);">Loading...</td></tr></tbody>
    </table>
  </div>

</div>

<script>
const API = '';

function fmtTime(ts) {
  if (!ts) return '--';
  return new Date(ts * 1000).toLocaleString();
}

async function update() {
  try {
    const [econ, chain] = await Promise.all([
      fetch(API + '/api/economy').then(r => r.json()),
      fetch(API + '/api/chain').then(r => r.json()),
    ]);

    document.getElementById('totalSupply').innerHTML = econ.total_supply.toFixed(2) + '<span class="unit">W</span>';
    document.getElementById('circulating').textContent = econ.circulating.toFixed(2) + ' W';
    document.getElementById('totalStaked').textContent = econ.total_staked.toFixed(2) + ' W';
    document.getElementById('totalTransferred').textContent = econ.total_transferred.toFixed(2) + ' W';
    document.getElementById('activeWallets').textContent = econ.active_wallets;
    document.getElementById('totalTxns').textContent = econ.total_transactions;

    // Energy footprint
    if (econ.energy) {
      const e = econ.energy;
      document.getElementById('eKwh').innerHTML = e.total_kwh.toFixed(3) + '<span class="unit">kWh</span>';
      document.getElementById('eCo2').innerHTML = e.co2_kg.toFixed(3) + '<span class="unit">kg CO\u2082</span>';
      document.getElementById('eEff').innerHTML = e.w_per_kwh.toFixed(0) + '<span class="unit">W/kWh</span>';
      document.getElementById('eBtc').innerHTML = e.btc_equivalent_tx.toFixed(4) + '<span class="unit">BTC tx</span>';
      document.getElementById('eCoal').textContent = e.coal_kg.toFixed(3) + ' kg';
      document.getElementById('eVerified').textContent = e.verified_jobs;
      document.getElementById('ePerVer').textContent = e.kwh_per_verification.toFixed(4) + ' kWh';
      document.getElementById('eDiscKwh').textContent = e.discoveries_per_kwh.toFixed(0);
      const btcRatio = e.total_kwh > 0 ? Math.round(1449 / (e.total_kwh / Math.max(e.verified_jobs, 1))) : 0;
      document.getElementById('eBtcRatio').textContent = btcRatio > 0 ? btcRatio.toLocaleString() + 'x more efficient' : '--';
    }

    // Rewards
    if (econ.rewards) {
      document.getElementById('rEig').textContent = econ.rewards.eigenvalue;
      document.getElementById('rFals').textContent = econ.rewards.falsification;
      document.getElementById('rClock').textContent = econ.rewards.clock;
    }

    // Chain
    document.getElementById('chainHeight').textContent = econ.chain ? econ.chain.height : 0;
    document.getElementById('latestHash').textContent = econ.chain ? econ.chain.latest_hash : 'genesis';

    // Staking breakdown
    const sb = econ.stake_breakdown || {};
    const stakeTotal = (sb.eigenvalue || 0) + (sb.falsification || 0) + (sb.clock || 0);
    const bar = document.getElementById('stakeBar');
    if (stakeTotal > 0) {
      const ep = (sb.eigenvalue / stakeTotal * 100).toFixed(1);
      const fp = (sb.falsification / stakeTotal * 100).toFixed(1);
      const cp = (sb.clock / stakeTotal * 100).toFixed(1);
      bar.innerHTML = '<div style="background:var(--violet);width:' + ep + '%">' + ep + '%</div>' +
        '<div style="background:var(--green);width:' + fp + '%">' + fp + '%</div>' +
        '<div style="background:var(--cyan);width:' + cp + '%">' + cp + '%</div>';
    }
    document.getElementById('stakeDetails').innerHTML =
      '<div class="stat-row"><span class="stat-label">Eigenvalue</span><span class="stat-val">' + (sb.eigenvalue || 0) + ' W</span></div>' +
      '<div class="stat-row"><span class="stat-label">Falsification</span><span class="stat-val">' + (sb.falsification || 0) + ' W</span></div>' +
      '<div class="stat-row"><span class="stat-label">Clock</span><span class="stat-val">' + (sb.clock || 0) + ' W</span></div>';

    // Top holders
    const hBody = document.getElementById('holdersBody');
    hBody.innerHTML = (econ.top_holders || []).map((h, i) =>
      '<tr><td>' + (i+1) + '</td><td>' + h.name + '</td><td style="color:var(--gold)">' + h.balance.toFixed(2) + ' W</td><td>' + h.earned.toFixed(2) + ' W</td></tr>'
    ).join('') || '<tr><td colspan="4" style="color:var(--dim);">No holders yet</td></tr>';

    // Transactions
    const txBody = document.getElementById('txBody');
    txBody.innerHTML = (econ.recent_transactions || []).map(t => {
      const cls = t.tx_type === 'mint' ? 'mint' : t.tx_type === 'transfer' ? 'transfer' : 'stake-tx';
      return '<tr><td class="' + cls + '">' + t.tx_type + '</td><td>' + (t.from_name || t.from_id || '--') +
        '</td><td>' + (t.to_name || t.to_id || '--') + '</td><td style="color:var(--gold)">' + t.amount.toFixed(4) +
        ' W</td><td style="color:var(--dim);font-size:0.75em">' + (t.memo || '') +
        '</td><td>' + (t.block_height > 0 ? '#' + t.block_height : 'pending') + '</td></tr>';
    }).join('') || '<tr><td colspan="6" style="color:var(--dim);">No transactions yet</td></tr>';

    // Blocks
    const bc = document.getElementById('blocksContainer');
    bc.innerHTML = (chain.blocks || []).slice(0, 10).map(b =>
      '<div class="chain-block"><div style="display:flex;justify-content:space-between;">' +
      '<span style="color:var(--gold);">Block #' + b.height + '</span>' +
      '<span style="color:var(--dim);font-size:0.75em;">' + b.n_transactions + ' txns / ' + b.total_minted.toFixed(2) + ' W</span></div>' +
      '<div class="hash">' + b.block_hash + '</div></div>'
    ).join('') || '<div style="color:var(--dim);">No blocks yet — first block seals after transactions.</div>';

  } catch(e) { console.error('Economy update error:', e); }
}

update();
setInterval(update, 15000);
</script>
</body>
</html>
"""

# ═══════════════════════════════════════════════════════════
# Landing Page
# ═══════════════════════════════════════════════════════════

LANDING_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>W@Home — Distributed Search for Universal Constants</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: #0a0a12; color: #e0e0e8;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    line-height: 1.7; overflow-x: hidden;
  }
  a { color: #a78bfa; text-decoration: none; }
  a:hover { color: #c4b5fd; text-decoration: underline; }

  /* Hero */
  .hero {
    min-height: 100vh; display: flex; flex-direction: column;
    align-items: center; justify-content: center; text-align: center;
    padding: 2rem; position: relative;
    background: radial-gradient(ellipse at 50% 30%, #1a1040 0%, #0a0a12 70%);
  }
  .hero canvas {
    position: absolute; top: 0; left: 0; width: 100%; height: 100%;
    opacity: 0.3; pointer-events: none;
  }
  .hero-content { position: relative; z-index: 1; max-width: 800px; }
  .hero h1 {
    font-size: 3.5rem; font-weight: 200; letter-spacing: 0.05em;
    background: linear-gradient(135deg, #a78bfa, #6dd5ed, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
  }
  .hero .tagline {
    font-size: 1.3rem; color: #9090b0; margin-bottom: 2rem; font-weight: 300;
  }
  .hero .hook {
    font-size: 1.1rem; color: #c0c0d8; max-width: 650px; margin: 0 auto 2.5rem;
  }
  .cta-row { display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap; }
  .btn {
    padding: 0.8rem 2rem; border-radius: 8px; font-size: 1rem;
    font-weight: 600; cursor: pointer; border: none; transition: all 0.2s;
  }
  .btn-primary {
    background: linear-gradient(135deg, #7c3aed, #6d28d9);
    color: white; box-shadow: 0 4px 20px rgba(124,58,237,0.3);
  }
  .btn-primary:hover { transform: translateY(-2px); box-shadow: 0 6px 30px rgba(124,58,237,0.5); }
  .btn-secondary {
    background: rgba(255,255,255,0.05); color: #c0c0d8;
    border: 1px solid rgba(255,255,255,0.1);
  }
  .btn-secondary:hover { background: rgba(255,255,255,0.1); }
  .scroll-hint {
    position: absolute; bottom: 2rem; color: #505070;
    animation: bob 2s ease-in-out infinite;
  }
  @keyframes bob { 0%,100% { transform: translateY(0); } 50% { transform: translateY(8px); } }

  /* Sections */
  section { padding: 5rem 2rem; max-width: 900px; margin: 0 auto; }
  section h2 {
    font-size: 2rem; font-weight: 300; margin-bottom: 1.5rem;
    color: #c4b5fd; letter-spacing: 0.02em;
  }
  section p { margin-bottom: 1.2rem; color: #b0b0c8; font-size: 1.05rem; }
  .highlight { color: #e0e0f0; font-weight: 500; }

  /* What section */
  .what-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1.5rem; margin-top: 2rem;
  }
  .what-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px; padding: 1.5rem;
  }
  .what-card h3 { color: #a78bfa; font-size: 1.1rem; margin-bottom: 0.5rem; }
  .what-card p { font-size: 0.95rem; color: #9090b0; margin: 0; }

  /* Why it matters */
  .timeline { border-left: 2px solid #2a2a4a; padding-left: 2rem; margin-top: 1.5rem; }
  .timeline-item { margin-bottom: 2rem; position: relative; }
  .timeline-item::before {
    content: ''; width: 12px; height: 12px; border-radius: 50%;
    background: #7c3aed; position: absolute; left: -2.4rem; top: 0.4rem;
  }
  .timeline-item h3 { color: #c4b5fd; font-size: 1rem; margin-bottom: 0.3rem; }
  .timeline-item p { color: #9090b0; font-size: 0.95rem; margin: 0; }

  /* Download */
  .dl-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem; margin-top: 2rem;
  }
  .dl-card {
    background: rgba(124,58,237,0.08); border: 1px solid rgba(124,58,237,0.2);
    border-radius: 12px; padding: 1.5rem; text-align: center;
    transition: all 0.2s;
  }
  .dl-card:hover { background: rgba(124,58,237,0.15); transform: translateY(-2px); }
  .dl-card .icon { font-size: 2rem; margin-bottom: 0.5rem; }
  .dl-card .platform { font-weight: 600; color: #e0e0f0; }
  .dl-card .detail { font-size: 0.85rem; color: #7070a0; margin-top: 0.3rem; }

  /* Live stats bar */
  .live-bar {
    background: rgba(124,58,237,0.1); border: 1px solid rgba(124,58,237,0.2);
    border-radius: 12px; padding: 1.5rem; margin: 2rem 0;
    display: flex; justify-content: space-around; flex-wrap: wrap; gap: 1rem;
    text-align: center;
  }
  .live-stat .num { font-size: 1.8rem; font-weight: 700; color: #a78bfa; }
  .live-stat .label { font-size: 0.8rem; color: #7070a0; text-transform: uppercase; letter-spacing: 0.1em; }

  /* Footer */
  footer {
    text-align: center; padding: 3rem 2rem; color: #505070;
    border-top: 1px solid rgba(255,255,255,0.05);
    font-size: 0.9rem;
  }
  footer a { color: #7070a0; }

  /* FAQ */
  details { margin-bottom: 1rem; }
  summary {
    cursor: pointer; color: #c4b5fd; font-size: 1.05rem; font-weight: 500;
    padding: 0.5rem 0;
  }
  details p { padding-left: 1rem; }

  @media (max-width: 600px) {
    .hero h1 { font-size: 2.2rem; }
    .hero .tagline { font-size: 1rem; }
    section { padding: 3rem 1.2rem; }
  }
</style>
</head>
<body>

<div class="hero">
  <canvas id="bgCanvas"></canvas>
  <div class="hero-content">
    <h1>W@Home</h1>
    <div class="tagline">Distributed Search for Universal Constants</div>
    <p class="hook">
      Your computer can help search for hidden mathematical constants
      in the spectral structure of fractal geometry. Like SETI@Home searched
      for aliens in radio signals, we're searching for the fingerprints
      of fundamental physics in the eigenvalues of a Menger sponge.
    </p>
    <div class="cta-row">
      <a href="#download" class="btn btn-primary">Join the Search</a>
      <a href="/dashboard" class="btn btn-secondary">Live Dashboard</a>
      <a href="/results" class="btn btn-secondary">Results</a>
      <a href="/chat" class="btn btn-secondary">Chat</a>
      <a href="/economy" class="btn btn-secondary" style="border-color:#ffd06a;color:#ffd06a;">W Economy</a>
    </div>
  </div>
  <div class="scroll-hint">scroll</div>
</div>

<section>
  <div class="live-bar" id="liveStats">
    <div class="live-stat"><div class="num" id="statWorkers">-</div><div class="label">Active Workers</div></div>
    <div class="live-stat"><div class="num" id="statJobs">-</div><div class="label">Jobs Completed</div></div>
    <div class="live-stat"><div class="num" id="statDiscoveries">-</div><div class="label">Hits</div></div>
    <div class="live-stat"><div class="num" id="statProgress">-</div><div class="label">% Swept</div></div>
  </div>
</section>

<section>
  <h2>What is this?</h2>
  <p>
    In early 2025, researcher <span class="highlight">Sylvan Gaskin</span> discovered that the
    eigenvalue spectrum of the <span class="highlight">W-operator</span> — a boundary
    operator on the Menger sponge fractal — encodes ratios that match fundamental
    physical constants. Not approximately. <span class="highlight">Exactly.</span>
  </p>
  <p>
    The Menger sponge is a fractal built by recursively removing the center of a cube,
    20 subcubes at a time. Its boundary has a rich spectral structure — eigenvalues
    that describe how waves propagate through this fractal geometry. When those
    eigenvalues are compared as ratios, they match values like pi, phi, sqrt(2),
    and dozens of physical constants from particle physics and cosmology.
  </p>
  <p>
    <span class="highlight">W@Home</span> is a distributed computing project that
    systematically sweeps the parameter space of this operator, computing eigenvalue
    spectra at 200,000 different parameter values. Your computer (or phone) computes
    a small piece of the puzzle, and together we map the full spectral landscape.
  </p>

  <div class="what-grid">
    <div class="what-card">
      <h3>The Sponge</h3>
      <p>A Menger sponge at depth 4: 160,000 subcubes, each face tiled into a torus graph. The W-operator acts on this boundary.</p>
    </div>
    <div class="what-card">
      <h3>The Sweep</h3>
      <p>We vary a coupling parameter lambda from 0.4 to 2.4 in 200,000 steps. At each step, we solve the eigenvalue spectrum.</p>
    </div>
    <div class="what-card">
      <h3>The Search</h3>
      <p>Every eigenvalue ratio is compared against known constants. Matches are flagged as hits. When two independent workers agree, it becomes a confirmed discovery.</p>
    </div>
  </div>
</section>

<section>
  <h2>Why does this matter?</h2>
  <p>
    If the spectral structure of a pure mathematical object — a fractal with no physics
    baked in — naturally produces the constants of our universe, that would be one of
    the most significant discoveries in the history of science. It would suggest that
    the laws of physics aren't arbitrary, but emerge from geometry itself.
  </p>

  <div class="timeline">
    <div class="timeline-item">
      <h3>If we find nothing</h3>
      <p>The original matches were coincidences. We've ruled out a hypothesis cleanly, and that has scientific value too. The computation data becomes a public spectral atlas of the Menger sponge.</p>
    </div>
    <div class="timeline-item">
      <h3>If we find some matches</h3>
      <p>Specific lambda values produce clusters of physical constants. This narrows the parameter space for deeper investigation and points toward which constants are structurally related.</p>
    </div>
    <div class="timeline-item">
      <h3>If the whole spectrum is rich</h3>
      <p>The Menger sponge boundary operator is a Rosetta Stone for physics. The geometry of a fractal encodes the fundamental constants of nature. We rewrite the foundations of theoretical physics.</p>
    </div>
  </div>
</section>

<section id="download">
  <h2>Join the Search</h2>
  <p>
    Download W@Home for your platform. First run asks for a name and password — that's it.
    Your computer starts computing immediately. No configuration needed.
  </p>

  <div style="margin-bottom: 2rem; background: linear-gradient(135deg, rgba(124,58,237,0.15), rgba(109,213,237,0.1)); border: 1px solid rgba(167,139,250,0.3); border-radius: 12px; padding: 2rem; text-align: center;">
    <div style="font-size: 1.4rem; color: #c4b5fd; margin-bottom: 0.5rem;">No download? No problem.</div>
    <p style="color: #9090b0; margin-bottom: 1.2rem; font-size: 0.95rem;">Run W@Home directly in your browser — works on any device. Same computation, zero installation.</p>
    <a href="/compute" class="btn btn-primary" style="font-size: 1.1rem; padding: 0.9rem 2.5rem;">Run in Browser</a>
  </div>

  <div style="text-align: center; color: #505070; margin-bottom: 1.5rem; font-size: 0.85rem;">or install a native client for background computing:</div>

  <div class="dl-grid">
    <a href="/static/WHome-Setup.exe" class="dl-card">
      <div class="icon">&#x1F5A5;</div>
      <div class="platform">Windows</div>
      <div class="detail">Download installer, double-click</div>
    </a>
    <a href="#" onclick="return false;" class="dl-card">
      <div class="icon">&#x1F34E;</div>
      <div class="platform">macOS</div>
      <div class="detail">Paste the command below in Terminal</div>
    </a>
    <a href="#" onclick="return false;" class="dl-card">
      <div class="icon">&#x1F427;</div>
      <div class="platform">Linux</div>
      <div class="detail">Paste the command below in Terminal</div>
    </a>
    <a href="/static/wathome-latest.apk" class="dl-card">
      <div class="icon">&#x1F4F1;</div>
      <div class="platform">Android</div>
      <div class="detail">Download APK, allow install</div>
    </a>
  </div>

  <div style="margin-top: 2rem; background: #12121e; border: 1px solid #2a2a3e; border-radius: 8px; padding: 1.2rem; text-align: left;">
    <div style="color: #a0f0ff; font-weight: bold; margin-bottom: 0.5rem;">Mac / Linux — paste in Terminal:</div>
    <code style="display: block; background: #0a0a14; padding: 0.8rem; border-radius: 4px; word-break: break-all; user-select: all; cursor: pointer;">curl -sSL https://wathome.akataleptos.com/static/install.sh | bash</code>
    <div style="color: #505070; font-size: 0.8rem; margin-top: 0.5rem;">Installs GUI + compute worker. Sets up venv, desktop entry, and optional autostart.</div>
  </div>
  <details style="margin-top: 1rem;">
    <summary>Termux (Android terminal)</summary>
    <p><code>curl -sSL https://wathome.akataleptos.com/static/install_termux.sh | bash</code></p>
  </details>
</section>

<section>
  <h2>FAQ</h2>
  <details>
    <summary>Is this safe to run?</summary>
    <p>Yes. The client does pure math — matrix construction and eigenvalue decomposition using numpy/scipy. It doesn't access your files, install anything, or run in the background. Kill it anytime. The source code is open: <a href="https://github.com/Cosmolalia/whome">github.com/Cosmolalia/whome</a>.</p>
  </details>
  <details>
    <summary>How much CPU does it use?</summary>
    <p>One core at ~100% while computing. Each job takes 30-120 seconds depending on your hardware. There's a small pause between jobs. It won't slow your machine to a crawl — it's one thread.</p>
  </details>
  <details>
    <summary>What's the math behind this?</summary>
    <p>The Menger sponge is a fractal with Hausdorff dimension ~2.73. Its boundary can be encoded as a graph, and the magnetic Laplacian on that graph has a discrete eigenvalue spectrum. We parameterize a coupling constant (lambda) that controls how faces of the sponge interact, then solve for eigenvalues at each lambda. Ratios between eigenvalues are compared against known physical constants. See <a href="https://akataleptos.com">akataleptos.com</a> for the full theory.</p>
  </details>
  <details>
    <summary>What are "hits"?</summary>
    <p>When an eigenvalue ratio matches a known physical constant (pi, phi, sqrt(2), fine structure constant, etc.) within tolerance, it's flagged as a <strong>hit</strong>. Each hit is independently verified by a second worker computing the same job — once verified, it becomes a <strong>confirmed discovery</strong>.</p>
  </details>
  <details>
    <summary>Who is behind this?</summary>
    <p>Sylvan Gaskin — independent researcher of antinomics, based in Hawaii. The Akataleptos project has produced 124 verified predictions from 7 Menger integers with zero fitted parameters. This isn't a crank project. The math is real, the code is open, and the results are reproducible.</p>
  </details>
</section>

<footer>
  <p>W@Home is part of the <a href="https://akataleptos.com">Akataleptos Project</a></p>
  <p style="margin-top:0.5rem;">Built with volunteer compute. Every cycle counts.</p>
</footer>

<script>
// Background particle canvas
const c = document.getElementById('bgCanvas');
const ctx = c.getContext('2d');
function resize() { c.width = c.offsetWidth; c.height = c.offsetHeight; }
resize(); window.addEventListener('resize', resize);
const particles = Array.from({length: 60}, () => ({
  x: Math.random() * c.width, y: Math.random() * c.height,
  vx: (Math.random() - 0.5) * 0.3, vy: (Math.random() - 0.5) * 0.3,
  r: Math.random() * 2 + 1
}));
function drawBg() {
  ctx.clearRect(0, 0, c.width, c.height);
  ctx.fillStyle = 'rgba(167,139,250,0.5)';
  particles.forEach(p => {
    p.x += p.vx; p.y += p.vy;
    if (p.x < 0 || p.x > c.width) p.vx *= -1;
    if (p.y < 0 || p.y > c.height) p.vy *= -1;
    ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2); ctx.fill();
  });
  // Draw connections
  ctx.strokeStyle = 'rgba(167,139,250,0.08)';
  for (let i = 0; i < particles.length; i++)
    for (let j = i+1; j < particles.length; j++) {
      const dx = particles[i].x - particles[j].x, dy = particles[i].y - particles[j].y;
      if (dx*dx + dy*dy < 15000) {
        ctx.beginPath(); ctx.moveTo(particles[i].x, particles[i].y);
        ctx.lineTo(particles[j].x, particles[j].y); ctx.stroke();
      }
    }
  requestAnimationFrame(drawBg);
}
drawBg();

// Live stats
async function updateStats() {
  try {
    const r = await fetch('/progress');
    const d = await r.json();
    document.getElementById('statJobs').textContent = (d.completed || 0).toLocaleString();
    document.getElementById('statDiscoveries').textContent = d.total_discoveries || '0';
    document.getElementById('statProgress').textContent = (d.percent_complete || 0).toFixed(2) + '%';
    const a = await fetch('/active');
    const w = await a.json();
    document.getElementById('statWorkers').textContent = w.length || '0';
  } catch(e) {}
}
updateStats();
setInterval(updateStats, 5000);
</script>
</body>
</html>
"""

# ═══════════════════════════════════════════════════════════
# Chat Page
# ═══════════════════════════════════════════════════════════

CHAT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>W@Home — Chat</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: #0a0a12; color: #e0e0e8;
    font-family: 'Segoe UI', system-ui, sans-serif;
    height: 100vh; display: flex; flex-direction: column;
  }
  a { color: #a78bfa; text-decoration: none; }

  /* Header */
  .header {
    background: rgba(255,255,255,0.03); border-bottom: 1px solid rgba(255,255,255,0.06);
    padding: 0.8rem 1.5rem; display: flex; align-items: center; justify-content: space-between;
  }
  .header h1 { font-size: 1.2rem; font-weight: 400; color: #a78bfa; }
  .header nav { display: flex; gap: 1rem; font-size: 0.9rem; }

  /* Layout */
  .main { display: flex; flex: 1; overflow: hidden; }
  .chat-panel { flex: 1; display: flex; flex-direction: column; min-width: 0; }
  .dash-panel {
    width: 380px; border-left: 1px solid rgba(255,255,255,0.06);
    overflow-y: auto; background: rgba(255,255,255,0.01);
  }
  @media (max-width: 800px) { .dash-panel { display: none; } }

  /* Messages */
  .messages {
    flex: 1; overflow-y: auto; padding: 1rem; display: flex; flex-direction: column;
    gap: 0.3rem;
  }
  .msg { padding: 0.3rem 0; font-size: 0.95rem; word-wrap: break-word; }
  .msg .name { color: #a78bfa; font-weight: 600; margin-right: 0.5rem; }
  .msg .time { color: #505070; font-size: 0.75rem; margin-right: 0.5rem; }
  .msg .text { color: #c0c0d8; }
  .msg-system { color: #505070; font-style: italic; font-size: 0.85rem; }

  /* Input */
  .input-bar {
    border-top: 1px solid rgba(255,255,255,0.06);
    padding: 0.8rem 1rem; display: flex; gap: 0.5rem;
  }
  .input-bar input {
    flex: 1; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px; padding: 0.6rem 1rem; color: #e0e0e8; font-size: 0.95rem;
    outline: none;
  }
  .input-bar input:focus { border-color: #7c3aed; }
  .input-bar button {
    padding: 0.6rem 1.5rem; background: #7c3aed; color: white; border: none;
    border-radius: 8px; cursor: pointer; font-weight: 600;
  }
  .input-bar button:hover { background: #6d28d9; }
  .input-bar button:disabled { opacity: 0.5; cursor: not-allowed; }

  /* Auth overlay */
  .auth-overlay {
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(10,10,18,0.95); display: flex; align-items: center;
    justify-content: center; z-index: 100;
  }
  .auth-box {
    background: #12121e; border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px; padding: 2rem; width: 360px; max-width: 90vw;
  }
  .auth-box h2 { font-size: 1.3rem; font-weight: 400; color: #a78bfa; margin-bottom: 1.5rem; text-align: center; }
  .auth-box input {
    width: 100%; padding: 0.7rem 1rem; margin-bottom: 1rem;
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px; color: #e0e0e8; font-size: 1rem; outline: none;
  }
  .auth-box button {
    width: 100%; padding: 0.8rem; background: #7c3aed; color: white;
    border: none; border-radius: 8px; font-size: 1rem; cursor: pointer; font-weight: 600;
  }
  .auth-box .err { color: #ef4444; font-size: 0.85rem; margin-bottom: 1rem; text-align: center; }

  /* Mini dashboard */
  .mini-stat {
    padding: 1rem 1.5rem; border-bottom: 1px solid rgba(255,255,255,0.04);
  }
  .mini-stat .label { font-size: 0.75rem; color: #505070; text-transform: uppercase; letter-spacing: 0.1em; }
  .mini-stat .val { font-size: 1.4rem; font-weight: 600; color: #a78bfa; }
  .mini-discoveries { padding: 1rem 1.5rem; }
  .mini-discoveries h3 { font-size: 0.85rem; color: #505070; text-transform: uppercase; margin-bottom: 0.5rem; }
  .disc-item { font-size: 0.85rem; color: #9090b0; padding: 0.2rem 0; }
  .disc-item .const { color: #ffd700; font-weight: 600; }
  .user-item { font-size: 0.85rem; padding: 0.2rem 0; display: flex; align-items: center; gap: 0.4rem; }
  .dot-chat { width: 6px; height: 6px; border-radius: 50%; background: #22c55e; display: inline-block; }
  .dot-compute { width: 6px; height: 6px; border-radius: 50%; background: #a78bfa; display: inline-block; }
  .user-name { color: #c0c0d8; }
</style>
</head>
<body>

<div class="header">
  <h1>W@Home Chat</h1>
  <nav>
    <a href="/">Home</a>
    <a href="/dashboard">Dashboard</a>
    <a href="/economy">W Economy</a>
    <span id="userDisplay" style="color:#505070;"></span>
  </nav>
</div>

<div class="main">
  <div class="chat-panel">
    <div class="messages" id="messages"></div>
    <div class="input-bar">
      <input type="text" id="msgInput" placeholder="Type a message..." disabled maxlength="500">
      <button id="sendBtn" disabled onclick="sendMsg()">Send</button>
    </div>
  </div>
  <div class="dash-panel" id="dashPanel">
    <div class="mini-stat"><div class="label">Workers Online</div><div class="val" id="dWorkers">-</div></div>
    <div class="mini-stat"><div class="label">Jobs Completed</div><div class="val" id="dJobs">-</div></div>
    <div class="mini-stat"><div class="label">Lambda Swept</div><div class="val" id="dProgress">-</div></div>
    <div class="mini-stat"><div class="label">Hits</div><div class="val" id="dDisc">-</div></div>
    <div class="mini-discoveries" id="onlineList">
      <h3>In Chat</h3>
      <div id="chatUsers" style="margin-bottom:1rem;"></div>
      <h3>Computing</h3>
      <div id="computeUsers" style="margin-bottom:1rem;"></div>
    </div>
    <div class="mini-discoveries" id="discList"><h3>Recent Hits</h3></div>
  </div>
</div>

<!-- Auth overlay -->
<div class="auth-overlay" id="authOverlay">
  <div class="auth-box">
    <h2>Join Chat</h2>
    <div class="err" id="authErr"></div>
    <input type="text" id="authName" placeholder="Name">
    <input type="password" id="authPw" placeholder="Password">
    <button onclick="doAuth()">Enter</button>
    <p style="text-align:center;margin-top:1rem;font-size:0.85rem;color:#505070;">
      New name = register. Existing name = login.
    </p>
  </div>
</div>

<script>
let apiKey = localStorage.getItem('chat_api_key');
let username = localStorage.getItem('chat_username');
let ws = null;

if (apiKey && username) {
  document.getElementById('authOverlay').style.display = 'none';
  initChat();
}

async function doAuth() {
  const name = document.getElementById('authName').value.trim();
  const pw = document.getElementById('authPw').value;
  const err = document.getElementById('authErr');
  if (!name || !pw) { err.textContent = 'Name and password required'; return; }

  // Try register
  let r = await fetch('/register', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({name, password: pw, gpu_info: 'chat-only', device_name: 'browser'})
  });
  if (r.status === 409) {
    // Name taken, try login
    r = await fetch('/login', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({name, password: pw, device_name: 'browser', gpu_info: 'chat-only'})
    });
  }
  if (r.ok) {
    const d = await r.json();
    apiKey = d.api_key;
    username = name;
    localStorage.setItem('chat_api_key', apiKey);
    localStorage.setItem('chat_username', username);
    document.getElementById('authOverlay').style.display = 'none';
    initChat();
  } else {
    const d = await r.json().catch(() => ({}));
    err.textContent = d.detail || 'Authentication failed';
  }
}

// Enter key on password field
document.getElementById('authPw').addEventListener('keydown', e => { if (e.key === 'Enter') doAuth(); });

function initChat() {
  document.getElementById('userDisplay').textContent = username;
  document.getElementById('msgInput').disabled = false;
  document.getElementById('sendBtn').disabled = false;

  // Load history
  fetch('/chat/history').then(r => r.json()).then(msgs => {
    msgs.forEach(m => appendMsg(m));
    scrollBottom();
  });

  // WebSocket
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(proto + '//' + location.host + '/chat/ws');
  ws.onopen = () => {
    // Identify ourselves so server tracks our username
    ws.send(JSON.stringify({api_key: apiKey, content: ''}));
  };
  ws.onmessage = e => {
    const m = JSON.parse(e.data);
    if (m.type === 'presence') {
      updateUserList(m.chat_users);
      return;
    }
    appendMsg(m);
    scrollBottom();
  };
  ws.onclose = () => setTimeout(initChat, 3000);

  document.getElementById('msgInput').addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) sendMsg();
  });
}

function appendMsg(m) {
  const div = document.createElement('div');
  div.className = 'msg';
  const t = new Date(m.time * 1000);
  const ts = t.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
  if (m.type === 'system') {
    div.className = 'msg msg-system';
    div.textContent = m.content;
  } else {
    div.innerHTML = '<span class="time">' + ts + '</span><span class="name">' +
      escHtml(m.username) + '</span><span class="text">' + escHtml(m.content) + '</span>';
  }
  document.getElementById('messages').appendChild(div);
}

function escHtml(s) {
  const d = document.createElement('div'); d.textContent = s; return d.innerHTML;
}

function scrollBottom() {
  const el = document.getElementById('messages');
  el.scrollTop = el.scrollHeight;
}

function sendMsg() {
  const input = document.getElementById('msgInput');
  const text = input.value.trim();
  if (!text || !ws) return;
  ws.send(JSON.stringify({api_key: apiKey, content: text}));
  input.value = '';
}

// Mini dashboard updates
function updateUserList(chatUsers) {
  const el = document.getElementById('chatUsers');
  el.innerHTML = '';
  chatUsers.forEach(u => {
    const div = document.createElement('div');
    div.className = 'user-item';
    div.innerHTML = '<span class="dot-chat"></span><span class="user-name">' + escHtml(u) + '</span>';
    el.appendChild(div);
  });
  if (!chatUsers.length) el.innerHTML = '<div class="user-item" style="color:#505070;">nobody yet</div>';
}

async function updateDash() {
  try {
    const [pr, ar, dr, on] = await Promise.all([
      fetch('/progress').then(r=>r.json()),
      fetch('/active').then(r=>r.json()),
      fetch('/discoveries').then(r=>r.json()),
      fetch('/chat/online').then(r=>r.json()),
    ]);
    document.getElementById('dWorkers').textContent = ar.length;
    document.getElementById('dJobs').textContent = (pr.completed||0).toLocaleString();
    document.getElementById('dProgress').textContent = (pr.percent_complete||0).toFixed(2) + '%';
    document.getElementById('dDisc').textContent = pr.total_discoveries || '0';

    // Computing users
    const cel = document.getElementById('computeUsers');
    cel.innerHTML = '';
    (on.computing||[]).forEach(u => {
      const div = document.createElement('div');
      div.className = 'user-item';
      div.innerHTML = '<span class="dot-compute"></span><span class="user-name">' + escHtml(u) + '</span>';
      cel.appendChild(div);
    });
    if (!(on.computing||[]).length) cel.innerHTML = '<div class="user-item" style="color:#505070;">nobody computing</div>';

    const list = document.getElementById('discList');
    list.innerHTML = '<h3>Recent Hits</h3>';
    (dr.slice && dr.slice(-10) || []).reverse().forEach(d => {
      const div = document.createElement('div');
      div.className = 'disc-item';
      const tier = d.param_tier === 'mobile' ? ' <span style="color:#7070a0;font-size:0.75rem;">[scout]</span>' : ' <span style="color:#22c55e;font-size:0.75rem;">[HD]</span>';
      div.innerHTML = '<span class="const">' + escHtml(d.constant_name || d.name || '?') +
        '</span> at lambda=' + (d.lambda_val||0).toFixed(6) + tier;
      list.appendChild(div);
    });
  } catch(e) {}
}
updateDash();
setInterval(updateDash, 5000);
</script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
