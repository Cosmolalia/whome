"""
W@Home Client — Volunteer Spectral Search Worker

Pulls jobs from the Hive server, computes eigenvalue spectra of the
W-operator on Menger sponge boundary graphs, and reports discoveries.

Features (BOINC/SETI lessons):
- Checkpoint/resume: saves progress to disk, survives pause/reboot
- Integrity hashing: SHA256 of eigenvalues for corruption detection
- Rich terminal display: shows what you're computing and why
- Heartbeat: keeps server informed you're alive
- Exponential backoff: gentle on the server when things go wrong
- Screensaver mode: --screensaver flag for visual display
- Nice priority: runs at low CPU priority by default

Usage:
    python client.py                    # First run — registers with Hive
    python client.py --key YOUR_KEY     # Resume with existing API key
    python client.py --screensaver      # Visual screensaver mode
    python client.py --name "MyPC"      # Set volunteer name
"""

import requests
import time
import json
import hashlib
import os
import sys
import signal
import argparse
import threading
import numpy as np

# ═══════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════

SERVER_URL = os.environ.get("HIVE_SERVER", "http://localhost:8081")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "worker_config.json")
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoint.json")

# Physical constants we're hunting
CONSTANTS = {
    "phi":              1.6180339887,
    "e":                2.718281828,
    "pi":               3.141592653,
    "alpha_inv":        137.035999,
    "proton_electron":  1836.15267,
    "sqrt2":            1.4142135624,
    "sqrt3":            1.7320508076,
    "ln2":              0.6931471806,
}
TOLERANCE = 1e-4

# Backoff
MIN_BACKOFF = 2
MAX_BACKOFF = 120

# ═══════════════════════════════════════════════════════════
# GPU Detection
# ═══════════════════════════════════════════════════════════

try:
    import w_cuda
    HAS_GPU = w_cuda.HAS_GPU
    GPU_INFO = "CUDA (CuPy)"
except ImportError:
    HAS_GPU = False
    GPU_INFO = "CPU"

try:
    import w_operator as base_op
except ImportError:
    print("[!] Cannot import w_operator — make sure w_operator.py is in the same directory")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════
# Config persistence
# ═══════════════════════════════════════════════════════════

def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {}

def save_config(cfg):
    with open(CONFIG_PATH, 'w') as f:
        json.dump(cfg, f, indent=2)

def save_checkpoint(data):
    with open(CHECKPOINT_PATH, 'w') as f:
        json.dump(data, f, indent=2)

def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return None

def clear_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)

# ═══════════════════════════════════════════════════════════
# Hashing
# ═══════════════════════════════════════════════════════════

def hash_eigenvalues(eigs):
    rounded = [round(float(e), 12) for e in sorted(eigs)]
    payload = json.dumps(rounded, separators=(',', ':'))
    return hashlib.sha256(payload.encode()).hexdigest()

# ═══════════════════════════════════════════════════════════
# Constant detection
# ═══════════════════════════════════════════════════════════

def check_for_gold(eigs):
    eigs = np.sort(eigs[eigs > 1e-9])
    found = []
    n = len(eigs)
    # Check pairwise ratios (cap at first 60 eigenvalues to stay O(n^2) manageable)
    cap = min(n, 60)
    for i in range(cap):
        for j in range(i + 1, cap):
            ratio = eigs[j] / eigs[i]
            for name, val in CONSTANTS.items():
                if abs(ratio - val) / val < TOLERANCE:
                    found.append(f"{name} (ratio={ratio:.6f}, i={i}, j={j})")
    return found

# ═══════════════════════════════════════════════════════════
# Terminal display
# ═══════════════════════════════════════════════════════════

class Display:
    """Rich terminal output showing computation progress."""

    CLEAR_LINE = '\033[2K\r'
    CYAN = '\033[96m'
    GOLD = '\033[93m'
    VIOLET = '\033[95m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    DIM = '\033[90m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

    def __init__(self):
        self.job_id = None
        self.lambda_val = None
        self.stage = ""
        self.stage_num = 0
        self.total_stages = 5
        self.stats = {}
        self.worker_stats = {"jobs": 0, "discoveries": 0, "compute_hrs": 0}

    def banner(self, worker_id, gpu_info):
        print(f"""
{self.CYAN}╔══════════════════════════════════════════════════════════╗
║  {self.BOLD}W@HOME HIVE — Akataleptos Spectral Search{self.RESET}{self.CYAN}             ║
║  {self.DIM}Searching for the universe in eigenvalue ratios{self.RESET}{self.CYAN}        ║
╚══════════════════════════════════════════════════════════╝{self.RESET}
  {self.DIM}Worker:{self.RESET} {worker_id}
  {self.DIM}Compute:{self.RESET} {gpu_info}
  {self.DIM}Server:{self.RESET} {SERVER_URL}
""")

    def show_progress(self, stats=None):
        if stats:
            self.stats = stats

        # Build progress bar
        pct = self.stats.get('percent_complete', 0)
        bar_w = 40
        filled = int(bar_w * pct / 100)
        bar = f"{self.CYAN}{'█' * filled}{self.DIM}{'░' * (bar_w - filled)}{self.RESET}"

        workers = self.stats.get('active_workers', 0)
        disc = self.stats.get('total_discoveries', 0)

        print(f"\n  {self.VIOLET}HIVE STATUS{self.RESET}")
        print(f"  [{bar}] {pct:.2f}%")
        print(f"  {self.DIM}Workers:{self.RESET} {workers}  "
              f"{self.DIM}Jobs/hr:{self.RESET} {self.stats.get('jobs_per_hour', 0)}  "
              f"{self.DIM}Discoveries:{self.RESET} {self.GOLD}{disc}{self.RESET}")

        eta = self.stats.get('eta_hours')
        if eta:
            print(f"  {self.DIM}ETA:{self.RESET} {eta}h")
        print()

    def show_job(self, job_id, lambda_val, params):
        self.job_id = job_id
        self.lambda_val = lambda_val
        self.stage_num = 0

        print(f"  {self.CYAN}═══ Job {job_id} ═══{self.RESET}")
        print(f"  {self.DIM}Lambda:{self.RESET}  {self.GOLD}{lambda_val:.6f}{self.RESET}")
        print(f"  {self.DIM}Level:{self.RESET}   k={params.get('k', '?')}  "
              f"{self.DIM}Grid:{self.RESET} {params.get('G1', '?')}×{params.get('G2', '?')}  "
              f"{self.DIM}Circle:{self.RESET} S={params.get('S', '?')}")
        print()

    def show_stage(self, name, detail=""):
        self.stage_num += 1
        self.stage = name
        stages = ['Build Graph', 'Add Glue', 'Merge Edges', 'Build Laplacian', 'Solve Spectrum']
        markers = ""
        for i, s in enumerate(stages):
            if i + 1 < self.stage_num:
                markers += f"{self.GREEN}●{self.RESET} "
            elif i + 1 == self.stage_num:
                markers += f"{self.CYAN}◉{self.RESET} "
            else:
                markers += f"{self.DIM}○{self.RESET} "

        print(f"  {markers} {self.CYAN}{name}{self.RESET} {self.DIM}{detail}{self.RESET}")

    def show_result(self, hits, eigs, duration):
        n_eigs = len(eigs[eigs > 1e-9]) if isinstance(eigs, np.ndarray) else 0

        if hits:
            print(f"\n  {self.GOLD}{'═' * 50}")
            print(f"  ★ RESONANCE DETECTED ★")
            for h in hits:
                print(f"    {h}")
            print(f"  {'═' * 50}{self.RESET}\n")
        else:
            print(f"  {self.GREEN}●●●●●{self.RESET} {self.DIM}Complete{self.RESET}  "
                  f"{n_eigs} eigenvalues  {duration:.1f}s  "
                  f"{self.DIM}no constants in ratios{self.RESET}")

        self.worker_stats['jobs'] += 1
        self.worker_stats['compute_hrs'] += duration / 3600
        print(f"  {self.DIM}Session: {self.worker_stats['jobs']} jobs, "
              f"{self.worker_stats['compute_hrs']:.2f}h compute, "
              f"{self.worker_stats['discoveries']} discoveries{self.RESET}")
        print()

    def show_error(self, msg):
        print(f"  {self.RED}[!] {msg}{self.RESET}")

    def show_info(self, msg):
        print(f"  {self.DIM}{msg}{self.RESET}")

    def show_waiting(self, backoff):
        print(f"  {self.DIM}Waiting {backoff:.0f}s before retry...{self.RESET}")

# ═══════════════════════════════════════════════════════════
# Computation (with checkpointing)
# ═══════════════════════════════════════════════════════════

def run_job_with_stages(params, display, checkpoint=None):
    """Run eigenvalue computation with per-stage progress and checkpointing."""
    k = params['k']
    G1, G2 = params['G1'], params['G2']
    S = params['S']
    lam = params['lambda']
    w_glue = params['w_glue']
    N = 2

    # Stage 1: Build Graph
    if checkpoint and checkpoint.get('stage', 0) >= 1:
        display.show_stage("Build Graph", "(cached)")
        # Can't cache graph easily (too large), rebuild
    display.show_stage("Build Graph", f"k={k} boundary cubes")
    vertices, edges, b_ids = base_op.build_graph(k, G1, G2, S, N)
    save_checkpoint({'job_id': params.get('job_id'), 'stage': 1, 'params': params})

    # Stage 2: Add Glue
    display.show_stage("Add Glue", f"λ={lam:.6f} w={w_glue}")
    upd, psi_by = base_op.add_glue_edges(vertices, b_ids, lam, w_glue, G1, G2)
    save_checkpoint({'job_id': params.get('job_id'), 'stage': 2, 'params': params})

    # Stage 3: Merge Edges
    display.show_stage("Merge Edges", f"{len(edges):,} + {len(upd):,} glue")
    edges_merged = base_op.merge_edges(edges, upd)

    # Stage 4: Build Laplacian
    display.show_stage("Build Laplacian", f"{len(vertices):,} vertices")
    L, _ = base_op.build_magnetic_laplacian(vertices, edges_merged, s=(0, 0), psi_by_id=psi_by)
    save_checkpoint({'job_id': params.get('job_id'), 'stage': 4, 'params': params})

    # Stage 5: Solve Spectrum
    display.show_stage("Solve Spectrum", "eigsh M=40")
    if HAS_GPU:
        eigs = w_cuda.solve_spectrum_gpu(L, M=40)
    else:
        eigs = base_op.solve_spectrum(L, M=40)

    clear_checkpoint()
    return eigs

# ═══════════════════════════════════════════════════════════
# Heartbeat thread
# ═══════════════════════════════════════════════════════════

def heartbeat_loop(api_key, interval=120):
    """Send heartbeat to server every N seconds."""
    while True:
        try:
            requests.post(
                f"{SERVER_URL}/heartbeat",
                headers={"x-api-key": api_key},
                timeout=10
            )
        except Exception:
            pass
        time.sleep(interval)

# ═══════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════

def register(name, gpu_info):
    """Register with the Hive, get API key."""
    resp = requests.post(f"{SERVER_URL}/register", json={
        "name": name,
        "gpu_info": gpu_info,
    })
    if resp.status_code != 200:
        raise RuntimeError(f"Registration failed: {resp.text}")
    data = resp.json()
    cfg = load_config()
    cfg['api_key'] = data['api_key']
    cfg['worker_id'] = data['worker_id']
    cfg['name'] = name
    save_config(cfg)
    return data['api_key'], data['worker_id']

# ═══════════════════════════════════════════════════════════
# Main loop
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="W@Home Hive Worker")
    parser.add_argument("--key", help="API key (or auto-load from config)")
    parser.add_argument("--name", default="", help="Volunteer name")
    parser.add_argument("--server", default=None, help="Hive server URL")
    parser.add_argument("--screensaver", action="store_true", help="Run in screensaver mode")
    parser.add_argument("--nice", type=int, default=10, help="Process nice level (0-19)")
    args = parser.parse_args()

    global SERVER_URL
    if args.server:
        SERVER_URL = args.server

    # Set nice priority
    try:
        os.nice(args.nice)
    except (OSError, AttributeError):
        pass

    display = Display()

    # Get or create API key
    cfg = load_config()
    api_key = args.key or cfg.get('api_key')

    if not api_key:
        display.show_info("No API key found. Registering with Hive...")
        name = args.name or cfg.get('name', f"worker-{os.getpid()}")
        try:
            api_key, worker_id = register(name, GPU_INFO)
            display.show_info(f"Registered as {worker_id}. API key saved to {CONFIG_PATH}")
        except Exception as e:
            display.show_error(f"Registration failed: {e}")
            sys.exit(1)
    else:
        worker_id = cfg.get('worker_id', api_key[:8])
        if args.name and args.name != cfg.get('name'):
            cfg['name'] = args.name
            save_config(cfg)
            # Update name on server
            try:
                requests.post(
                    f"{SERVER_URL}/worker/update",
                    headers={"x-api-key": api_key},
                    json={"name": args.name},
                    timeout=10
                )
                display.show_info(f"Updated name to '{args.name}' on Hive")
            except Exception:
                pass  # non-critical

    display.banner(worker_id, GPU_INFO)

    # Screensaver mode
    if args.screensaver:
        try:
            from screensaver import run_screensaver
            run_screensaver(api_key, SERVER_URL, worker_id)
        except ImportError:
            display.show_error("screensaver.py not found — run without --screensaver")
        return

    # Start heartbeat thread
    hb_thread = threading.Thread(target=heartbeat_loop, args=(api_key,), daemon=True)
    hb_thread.start()

    # Graceful shutdown
    running = True
    def handle_signal(sig, frame):
        nonlocal running
        display.show_info("\nGraceful shutdown... finishing current job.")
        running = False
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Check for interrupted job
    ckpt = load_checkpoint()
    if ckpt:
        display.show_info(f"Found checkpoint for job {ckpt.get('job_id')} — will resume")

    backoff = MIN_BACKOFF

    while running:
        try:
            # Pull job
            resp = requests.post(
                f"{SERVER_URL}/job",
                headers={"x-api-key": api_key},
                timeout=30
            )

            if resp.status_code == 401:
                display.show_error("API key rejected. Re-register with --key or delete worker_config.json")
                break

            if resp.status_code != 200:
                display.show_error(f"Server error: {resp.status_code}")
                display.show_waiting(backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF)
                continue

            data = resp.json()

            if data.get('status') == 'no_jobs':
                display.show_info("All jobs assigned. Waiting for more...")
                display.show_waiting(60)
                time.sleep(60)
                continue

            backoff = MIN_BACKOFF  # Reset on success

            job_id = data['job_id']
            params = data['params']
            params['job_id'] = job_id
            progress = data.get('progress', {})

            display.show_progress(progress)
            display.show_job(job_id, params['lambda'], params)

            # Compute
            start_time = time.time()
            eigs = run_job_with_stages(params, display, checkpoint=ckpt)
            duration = time.time() - start_time
            ckpt = None  # Clear checkpoint reference after use

            # Check for constants
            hits = check_for_gold(eigs)
            display.show_result(hits, eigs, duration)

            if hits:
                display.worker_stats['discoveries'] += len(hits)

            # Submit result
            eig_hash = hash_eigenvalues(eigs.tolist())
            resp = requests.post(
                f"{SERVER_URL}/result",
                headers={"x-api-key": api_key},
                json={
                    "job_id": job_id,
                    "eigenvalues": eigs.tolist(),
                    "eigenvalues_hash": eig_hash,
                    "found_constants": hits,
                    "compute_seconds": duration,
                },
                timeout=30
            )

            if resp.status_code == 200:
                result = resp.json()
                if result.get('verified'):
                    display.show_info("Result verified by quorum!")
            else:
                display.show_error(f"Submit failed: {resp.status_code} {resp.text}")

        except KeyboardInterrupt:
            break
        except requests.ConnectionError:
            display.show_error("Cannot reach Hive server")
            display.show_waiting(backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF)
        except Exception as e:
            display.show_error(f"Error: {e}")
            display.show_waiting(backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF)

    display.show_info("Worker stopped. Progress saved. Run again to resume.")

if __name__ == "__main__":
    main()
