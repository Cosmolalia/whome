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

# ═══════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════

LAMBDA_START = 0.4
LAMBDA_END = 0.6
LAMBDA_STEP = 0.000001
TOTAL_JOBS = int((LAMBDA_END - LAMBDA_START) / LAMBDA_STEP)  # 200,000

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

# Job parameters
JOB_PARAMS = {
    "k": 4,
    "G1": 32,
    "G2": 32,
    "S": 16,
    "w_glue": 1000.0,
}

# Smaller params for mobile/low-memory devices (dense eigensolver can handle ~16K vertices)
JOB_PARAMS_MOBILE = {
    "k": 2,
    "G1": 8,
    "G2": 8,
    "S": 4,
    "w_glue": 1000.0,
}

# Quorum: how many workers must agree on a result
QUORUM_SIZE = 2

# How long before an assigned job is considered abandoned (seconds)
JOB_DEADLINE = 3600  # 1 hour

# Eigenvalue comparison tolerance for quorum validation
EIGEN_TOLERANCE = 1e-6

# ═══════════════════════════════════════════════════════════
# Database
# ═══════════════════════════════════════════════════════════

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn

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

    conn.close()

def seed_jobs(batch_size=1000):
    """Generate job records in batches. Idempotent — skips existing."""
    conn = get_db()
    existing = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    if existing >= TOTAL_JOBS:
        conn.close()
        return

    now = time.time()
    start_id = existing
    while start_id < TOTAL_JOBS:
        end_id = min(start_id + batch_size, TOTAL_JOBS)
        rows = []
        for i in range(start_id, end_id):
            lam = LAMBDA_START + i * LAMBDA_STEP
            rows.append((i, lam, 'pending', QUORUM_SIZE, 0, 0, now))
        conn.executemany(
            "INSERT OR IGNORE INTO jobs (id, lambda_val, status, quorum_target, quorum_received, verified, created_at) VALUES (?,?,?,?,?,?,?)",
            rows
        )
        conn.commit()
        start_id = end_id
        print(f"  Seeded jobs {start_id}/{TOTAL_JOBS}")

    conn.close()
    print(f"  Job seeding complete: {TOTAL_JOBS:,} jobs")

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
        conn.close()
        return

    now = time.time()
    # Insert canaries with IDs starting after real jobs
    base_id = TOTAL_JOBS + 1000
    for i, c in enumerate(canaries):
        conn.execute(
            "INSERT OR IGNORE INTO jobs (id, lambda_val, status, quorum_target, quorum_received, verified, created_at, is_canary, canary_hash) VALUES (?,?,?,?,?,?,?,?,?)",
            (base_id + i, c['lambda'], 'pending', 1, 0, 0, now, 1, c['hash'])
        )
    conn.commit()
    conn.close()
    print(f"  Seeded {len(canaries)} canary jobs")

def check_canary_result(job_id: int, eig_hash: str, worker_id: str, conn, param_tier: str = "desktop"):
    """
    Check if a submitted result for a verified job matches the known-good answer.
    Uses real verified results as canaries — no synthetic canary jobs needed.
    """
    job = conn.execute(
        "SELECT verified FROM jobs WHERE id = ?", (job_id,)
    ).fetchone()

    if not job or not job['verified']:
        return None  # Not a verified job — can't check

    # Was this worker already counted in the verification? Skip check if so
    existing = conn.execute(
        "SELECT id FROM results WHERE job_id = ? AND worker_id = ?", (job_id, worker_id)
    ).fetchone()
    # If they already have a result for this job, this is a re-submission (quorum), not a spot-check
    # We only spot-check NEW workers against verified jobs
    result_count = conn.execute(
        "SELECT COUNT(*) FROM results WHERE job_id = ? AND worker_id = ?", (job_id, worker_id)
    ).fetchone()[0]
    if result_count > 1:
        return None  # Already submitted before — not a spot-check

    # Get the verified hash (most common hash among existing results for this tier)
    verified_hash = conn.execute("""
        SELECT eigenvalues_hash, COUNT(*) as cnt FROM results
        WHERE job_id = ? AND param_tier = ?
        GROUP BY eigenvalues_hash ORDER BY cnt DESC LIMIT 1
    """, (job_id, param_tier)).fetchone()

    if not verified_hash:
        return None

    expected = verified_hash['eigenvalues_hash']
    passed = (eig_hash == expected)

    if passed:
        conn.execute("""
            UPDATE workers SET
                canaries_passed = canaries_passed + 1,
                trust_score = MIN(1.0, trust_score + 0.05)
            WHERE id = ?
        """, (worker_id,))
        print(f"  [SpotCheck] Worker {worker_id} PASSED on verified job {job_id}")
    else:
        conn.execute("""
            UPDATE workers SET
                canaries_failed = canaries_failed + 1,
                trust_score = MAX(0.0, trust_score - 0.25)
            WHERE id = ?
        """, (worker_id,))
        print(f"  [SpotCheck] Worker {worker_id} FAILED on verified job {job_id} "
              f"(got {eig_hash[:12]}... expected {expected[:12]}...)")

        # If trust drops below threshold, flag worker
        worker = conn.execute("SELECT trust_score FROM workers WHERE id = ?", (worker_id,)).fetchone()
        if worker and worker['trust_score'] < 0.3:
            conn.execute("UPDATE workers SET status = 'flagged' WHERE id = ?", (worker_id,))
            conn.execute("""
                UPDATE jobs SET status = 'pending', quorum_received = MAX(0, quorum_received - 1)
                WHERE id IN (SELECT job_id FROM results WHERE worker_id = ?)
                AND verified = 0
            """, (worker_id,))
            print(f"  [SpotCheck] Worker {worker_id} FLAGGED — re-queuing their results")

    conn.commit()
    return passed

# Spot-check rate: 1 in N jobs assigned to a new worker is a re-test of a verified job
SPOT_CHECK_RATE = 15  # ~7% of assignments

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

def verify_worker(api_key: str, conn) -> Optional[dict]:
    """Verify API key, return worker row or None."""
    key_hash = hash_key(api_key)
    # Check new api_keys table first
    ak = conn.execute("SELECT worker_id FROM api_keys WHERE key_hash = ?", (key_hash,)).fetchone()
    if ak:
        conn.execute("UPDATE api_keys SET last_used = ? WHERE key_hash = ?", (time.time(), key_hash))
        row = conn.execute("SELECT * FROM workers WHERE id = ?", (ak['worker_id'],)).fetchone()
        if row:
            conn.execute("UPDATE workers SET last_heartbeat = ? WHERE id = ?", (time.time(), row['id']))
            conn.commit()
            return dict(row)
    # Fallback: check legacy api_key_hash on workers table (pre-password accounts)
    row = conn.execute("SELECT * FROM workers WHERE api_key_hash = ?", (key_hash,)).fetchone()
    if row:
        conn.execute("UPDATE workers SET last_heartbeat = ? WHERE id = ?", (time.time(), row['id']))
        conn.commit()
    return dict(row) if row else None

# ═══════════════════════════════════════════════════════════
# Quorum validation
# ═══════════════════════════════════════════════════════════

def validate_quorum(job_id: int, conn, param_tier: str = "desktop"):
    """Check if enough results agree within the same param tier. If so, mark job verified."""
    results = conn.execute(
        "SELECT eigenvalues_hash, worker_id FROM results WHERE job_id = ? AND param_tier = ?",
        (job_id, param_tier)
    ).fetchall()

    if len(results) < QUORUM_SIZE:
        return False

    # Group by hash — if QUORUM_SIZE results have same hash, verified
    hash_counts = {}
    for r in results:
        h = r['eigenvalues_hash']
        hash_counts[h] = hash_counts.get(h, 0) + 1
        if hash_counts[h] >= QUORUM_SIZE:
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
    """Return current client version hash + downloadable files."""
    return {
        "client_version": _client_version(),
        "files": ["client.py", "w_operator.py"],
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

@app.get("/compute", response_class=HTMLResponse)
def compute_page():
    compute_path = os.path.join(HIVE_DIR, "compute.html")
    with open(compute_path) as f:
        return f.read()

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

@app.get("/pyodide/{filename}")
def serve_pyodide(filename: str):
    safe = filename.replace("..", "").replace("/", "")
    path = os.path.join(HIVE_DIR, "pyodide", safe)
    if not os.path.exists(path):
        raise HTTPException(404, "File not found")
    ext = os.path.splitext(safe)[1].lower()
    return FileResponse(path, media_type=MIME_TYPES.get(ext, 'application/octet-stream'),
                        headers={"Cache-Control": "public, max-age=31536000"})

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

@app.on_event("startup")
async def startup():
    print("[Hive] Initializing database...")
    init_db()
    print("[Hive] Seeding jobs...")
    seed_jobs()
    verified = get_db().execute("SELECT COUNT(*) FROM jobs WHERE verified = 1").fetchone()[0]
    print(f"[Hive] Spot-check pool: {verified} verified jobs")
    print(f"[Hive] Online — {TOTAL_JOBS:,} jobs")
    print(f"[Hive] Secret: {SERVER_SECRET[:8]}...")

# ── Registration ──

class RegisterRequest(BaseModel):
    name: str = ""
    gpu_info: str = ""
    password: str = ""
    device_name: str = ""

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
        conn.close()
        raise HTTPException(403, "Access denied")

    # Check if name is already taken
    existing = conn.execute("SELECT id FROM workers WHERE name = ?", (req.name,)).fetchone()
    if existing:
        conn.close()
        raise HTTPException(409, "Name already taken. Use /login to add a new device to your account.")

    worker_id = secrets.token_hex(8)
    api_key = secrets.token_urlsafe(32)
    now = time.time()
    pw_hash = hash_password(req.password)

    conn.execute(
        "INSERT INTO workers (id, api_key_hash, password_hash, name, registered_at, last_heartbeat, gpu_info) VALUES (?,?,?,?,?,?,?)",
        (worker_id, hash_key(api_key), pw_hash, req.name, now, now, req.gpu_info)
    )
    # Also insert into api_keys table for the new multi-device flow
    conn.execute(
        "INSERT INTO api_keys (key_hash, worker_id, device_name, created_at, last_used) VALUES (?,?,?,?,?)",
        (hash_key(api_key), worker_id, req.device_name or "primary", now, now)
    )
    _log_ip(conn, worker_id, ip, "register")
    conn.commit()
    conn.close()

    return {
        "worker_id": worker_id,
        "api_key": api_key,
        "message": "Welcome to the Hive. Your account is secured with your password."
    }

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
        conn.close()
        raise HTTPException(403, "Access denied")
    row = conn.execute("SELECT * FROM workers WHERE name = ?", (req.name,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(401, "Unknown account name")

    worker = dict(row)
    if not worker['password_hash']:
        conn.close()
        raise HTTPException(401, "This account was created before passwords. Re-register with a password.")

    if not verify_password(req.password, worker['password_hash']):
        conn.close()
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
    conn.close()

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
        conn.close()
        raise HTTPException(401, "Invalid API key")
    if req.name:
        conn.execute("UPDATE workers SET name = ? WHERE id = ?", (req.name, worker['id']))
        conn.commit()
    conn.close()
    return {"status": "updated", "worker_id": worker['id']}

# ── Job Assignment ──

@app.post("/job")
def get_job(request: Request, x_api_key: str = Header(), x_device_type: str = Header(default="desktop")):
    """Pull the next available job. Requires API key. Send x-device-type: mobile for smaller jobs."""
    ip = _get_client_ip(request)
    _check_rate(ip, "job")
    conn = get_db()
    if _is_banned(conn, ip):
        conn.close()
        raise HTTPException(403, "Access denied")
    worker = verify_worker(x_api_key, conn)
    if not worker:
        conn.close()
        raise HTTPException(401, "Invalid API key")
    if worker.get('status') == 'banned':
        conn.close()
        raise HTTPException(403, "Account suspended")
    _log_ip(conn, worker['id'], ip, "job")

    # Auto-detect mobile: header, or no scipy (numpy dense = can't handle large sparse matrices)
    is_mobile = x_device_type == "mobile" or "dense" in (worker.get('gpu_info') or '').lower()
    base_params = JOB_PARAMS_MOBILE if is_mobile else JOB_PARAMS
    print(f"[Hive] Job request from {worker['name']}, device_type={x_device_type}, mobile={is_mobile}")

    # Reclaim any expired assignments
    reclaim_expired(conn)

    # Anti-collusion: find all worker IDs that have used this IP
    # Don't assign same job to workers sharing an IP (prevents sock puppets)
    same_ip_workers = [r[0] for r in conn.execute(
        "SELECT DISTINCT worker_id FROM ip_log WHERE ip = ?", (ip,)).fetchall()]
    if worker['id'] not in same_ip_workers:
        same_ip_workers.append(worker['id'])

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
        conn.close()
        params = dict(base_params)
        params['lambda'] = existing['lambda_val']
        return {
            "status": "assigned",
            "job_id": existing['job_id'],
            "params": params,
            "deadline": time.time() + JOB_DEADLINE,
            "progress": stats,
            "resumed": True,
        }

    # Build exclusion list: jobs already assigned to this worker OR any same-IP worker
    placeholders = ','.join('?' * len(same_ip_workers))
    exclude_sql = f"SELECT job_id FROM assignments WHERE worker_id IN ({placeholders})"

    # Spot-check: randomly re-assign a verified job to test new workers
    import random
    job = None
    if random.randint(1, SPOT_CHECK_RATE) == 1:
        spot = conn.execute(f"""
            SELECT j.id, j.lambda_val FROM jobs j
            WHERE j.verified = 1
            AND j.id NOT IN ({exclude_sql})
            ORDER BY RANDOM() LIMIT 1
        """, same_ip_workers).fetchone()
        if spot:
            job = spot

    # If no canary selected, find a real pending job
    if not job:
        job = conn.execute(f"""
            SELECT j.id, j.lambda_val FROM jobs j
            WHERE j.status = 'assigned' AND j.is_canary = 0
            AND j.quorum_received < j.quorum_target
            AND j.id NOT IN ({exclude_sql})
            ORDER BY j.quorum_received DESC, j.id ASC
            LIMIT 1
        """, same_ip_workers).fetchone()

    if not job:
        job = conn.execute(f"""
            SELECT j.id, j.lambda_val FROM jobs j
            WHERE j.status = 'pending' AND j.is_canary = 0
            AND j.id NOT IN ({exclude_sql})
            ORDER BY j.id ASC
            LIMIT 1
        """, same_ip_workers).fetchone()

    if not job:
        conn.close()
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

    # Get progress stats for client display
    stats = _get_progress_stats(conn)
    conn.close()

    params = dict(base_params)
    params['lambda'] = job['lambda_val']

    return {
        "status": "assigned",
        "job_id": job['id'],
        "params": params,
        "deadline": deadline,
        "progress": stats
    }

# ── Result Submission ──

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
        conn.close()
        raise HTTPException(403, "Access denied")
    worker = verify_worker(x_api_key, conn)
    if not worker:
        conn.close()
        raise HTTPException(401, "Invalid API key")
    _log_ip(conn, worker['id'], ip, "result")

    # Verify this worker was assigned this job
    assignment = conn.execute(
        "SELECT id FROM assignments WHERE job_id = ? AND worker_id = ? AND status = 'assigned'",
        (result.job_id, worker['id'])
    ).fetchone()

    if not assignment:
        conn.close()
        raise HTTPException(400, "No active assignment for this job")

    # Server computes its own hash from raw eigenvalues (authoritative)
    # Don't reject on client hash mismatch — client may be older version with different precision
    computed_hash = hash_eigenvalues(result.eigenvalues)

    now = time.time()

    # Determine param tier based on worker capabilities
    is_mobile = "dense" in (worker.get('gpu_info') or '').lower()
    param_tier = "mobile" if is_mobile else "desktop"

    # Check spot-check BEFORE storing (so we know trust status)
    canary_result = check_canary_result(result.job_id, computed_hash, worker['id'], conn, param_tier)

    # Store result
    conn.execute(
        "INSERT INTO results (job_id, worker_id, eigenvalues_hash, eigenvalues_json, found_constants, compute_seconds, submitted_at, param_tier) VALUES (?,?,?,?,?,?,?,?)",
        (result.job_id, worker['id'], computed_hash,
         json.dumps([round(float(e), 10) for e in sorted(result.eigenvalues)]),
         json.dumps(result.found_constants),
         result.compute_seconds, now, param_tier)
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
            (result.job_id, job['lambda_val'], name, ratio, now, worker['id'], param_tier)
        )
        conn.execute(
            "UPDATE workers SET discoveries = discoveries + 1 WHERE id = ?",
            (worker['id'],)
        )

    conn.commit()

    # Attempt quorum validation (only compare same-tier results)
    verified = validate_quorum(result.job_id, conn, param_tier)

    # Mark job completed if enough results
    job_row = conn.execute("SELECT quorum_received, quorum_target FROM jobs WHERE id = ?", (result.job_id,)).fetchone()
    if job_row['quorum_received'] >= job_row['quorum_target'] and not verified:
        conn.execute("UPDATE jobs SET status = 'completed' WHERE id = ?", (result.job_id,))
        conn.commit()

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

    conn.close()

    return {
        "status": "accepted",
        "verified": verified,
        "discoveries": len(result.found_constants),
        "canary_check": canary_result,  # None if not canary, True/False if was
        "receipt": receipt,
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
    conn.close()
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

    active_workers = conn.execute(
        "SELECT COUNT(*) FROM workers WHERE last_heartbeat > ? AND status = 'active'",
        (time.time() - 300,)  # active in last 5 min
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

    return {
        "total_jobs": total,
        "completed": completed,
        "verified": verified,
        "pending": pending,
        "assigned": assigned,
        "disputed": disputed,
        "percent_complete": round(pct, 4),
        "active_workers": active_workers,
        "total_compute_hours": round(total_compute / 3600, 2),
        "total_discoveries": total_discoveries,
        "total_confirmed": total_confirmed,
        "current_lambda": round(current_lam, 6),
        "jobs_per_hour": recent_completions,
        "eta_hours": round(eta_hours, 1) if eta_hours != float('inf') else None,
        "lambda_range": [LAMBDA_START, LAMBDA_END],
        "lambda_step": LAMBDA_STEP,
    }

@app.get("/progress")
def get_progress():
    conn = get_db()
    stats = _get_progress_stats(conn)
    conn.close()
    return stats

@app.get("/progress/heatmap")
def progress_heatmap(blocks: int = 20):
    """Return per-block completion for Menger sponge visualization."""
    blocks = min(max(blocks, 8), 160000)
    block_size = TOTAL_JOBS / blocks
    conn = get_db()

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

    conn.close()

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
    conn.close()
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
    conn.close()
    return [dict(r) for r in rows]

# ── Leaderboard ──

@app.get("/leaderboard")
def leaderboard():
    conn = get_db()
    rows = conn.execute("""
        SELECT id, name, jobs_completed, compute_seconds, discoveries,
               last_heartbeat, gpu_info, trust_score, canaries_passed, canaries_failed
        FROM workers
        WHERE status != 'flagged'
        ORDER BY jobs_completed DESC, registered_at ASC
        LIMIT 50
    """).fetchall()
    conn.close()
    now = time.time()
    return [{
        "rank": i + 1,
        "name": r['name'] or r['id'][:8],
        "jobs": r['jobs_completed'],
        "compute_hours": round(r['compute_seconds'] / 3600, 2),
        "discoveries": r['discoveries'],
        "online": (now - r['last_heartbeat']) < 300,
        "gpu": r['gpu_info'],
        "trust": round(r['trust_score'], 2),
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
    conn.close()
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
    conn.close()
    return [{"username": r['username'], "content": r['content'],
             "time": r['sent_at'], "type": r['msg_type']} for r in reversed(rows)]

@app.post("/chat/send")
def chat_send(x_api_key: str = Header(), message: dict = {}):
    """Send a chat message (authenticated workers)."""
    conn = get_db()
    worker = verify_worker(x_api_key, conn)
    if not worker:
        conn.close()
        raise HTTPException(401, "Invalid API key")
    content = str(message.get('content', '')).strip()[:500]  # 500 char limit
    if not content:
        conn.close()
        raise HTTPException(400, "Empty message")
    now = time.time()
    conn.execute("INSERT INTO messages (username, content, sent_at) VALUES (?,?,?)",
                 (worker['name'], content, now))
    conn.commit()
    conn.close()
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
        "SELECT name FROM workers WHERE last_heartbeat > ?", (now - 300,)
    ).fetchall()
    conn.close()
    computing = sorted(set(r['name'] for r in rows))
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
                    conn.close()
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
    conn.close()
    return {
        "name": "W@Home Hive — Akataleptos Distributed Spectral Search",
        "version": "2.0",
        "status": "online",
        **stats
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
    conn.close()
    return {"status": "banned", "worker_id": worker_id, "name": name[0] if name else "unknown"}

@app.post("/admin/ban-ip/{ip}")
def admin_ban_ip(ip: str, reason: str = "", x_admin_key: str = Header(default="")):
    _verify_admin(x_admin_key)
    conn = get_db()
    conn.execute("INSERT INTO bans (ip, reason, banned_at) VALUES (?,?,?)",
                 (ip, reason, time.time()))
    conn.commit()
    conn.close()
    return {"status": "banned", "ip": ip}

@app.post("/admin/unban-worker/{worker_id}")
def admin_unban_worker(worker_id: str, x_admin_key: str = Header(default="")):
    _verify_admin(x_admin_key)
    conn = get_db()
    conn.execute("UPDATE workers SET status = 'active' WHERE id = ?", (worker_id,))
    conn.execute("DELETE FROM bans WHERE worker_id = ?", (worker_id,))
    conn.commit()
    conn.close()
    return {"status": "unbanned", "worker_id": worker_id}

@app.post("/admin/unban-ip/{ip}")
def admin_unban_ip(ip: str, x_admin_key: str = Header(default="")):
    _verify_admin(x_admin_key)
    conn = get_db()
    conn.execute("DELETE FROM bans WHERE ip = ?", (ip,))
    conn.commit()
    conn.close()
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
    conn.close()
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
    conn.close()
    return {"status": "keys_revoked", "worker_id": worker_id}

# ── Dashboard ──

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD_HTML

# ── Chat ──

@app.get("/chat", response_class=HTMLResponse)
def chat_page():
    return CHAT_HTML

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
    <a href="/chat" style="color:#a78bfa;">Chat</a>
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
        <th></th><th>Volunteer</th><th>Jobs</th><th>Compute</th><th>Hits</th><th>GPU</th>
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
    const [prog, disc, lb, active] = await Promise.all([
      fetch(API + '/progress').then(r => r.json()),
      fetch(API + '/discoveries').then(r => r.json()),
      fetch(API + '/leaderboard').then(r => r.json()),
      fetch(API + '/active').then(r => r.json()),
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
        '<td style="color: var(--dim);">' + (w.gpu || 'CPU') + '</td>' +
      '</tr>'
    ).join('') || '<tr><td colspan="6" style="color: var(--dim);">No volunteers yet</td></tr>';

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
  if (hasDiscovery) return '#ffd06a';
  if (pct >= 100) return '#a0f0ff';
  if (pct > 0) {
    const t = pct / 100;
    const r = Math.floor(196 * (1-t) + 160 * t);
    const g = Math.floor(160 * (1-t) + 240 * t);
    return '#' + r.toString(16).padStart(2,'0') + g.toString(16).padStart(2,'0') + 'ff';
  }
  if (isActive) return '#ff66ff';
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
      const glow = block.discoveries > 0 || block.pct >= 100;
      const pulse = isActive && block.pct === 0;
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
      <a href="/chat" class="btn btn-secondary">Chat</a>
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
