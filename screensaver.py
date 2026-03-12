"""
W@Home Screensaver — Menger sponge visualization with live computation overlay.

Shows the Menger sponge rotating while the worker computes eigenvalue spectra.
Displays:
  - Current lambda value and sweep position
  - Computation stage progress
  - Eigenvalue spectrum visualization
  - Discovery feed
  - Hive-wide stats (workers online, % complete)

The SETI@home lesson: the screensaver is why people install it.
This is the Menger sponge version.

Usage:
    python screensaver.py                    # Standalone (no computation)
    python screensaver.py --fullscreen       # Fullscreen mode
    WHome.scr /s                             # Windows screensaver mode

Shared state: reads/writes %LOCALAPPDATA%\WHome\compute_status.json (Windows)
or ~/.whome/compute_status.json (Linux/macOS). If the main app is already
computing, the screensaver just displays. If not, it computes on its own.
"""

import pygame
import numpy as np
import math
import sys
import threading
import time
import json
import os
import hashlib
import platform

SCR_DIR = os.path.dirname(os.path.abspath(__file__))
# In PyInstaller, data files are at sys._MEIPASS
if getattr(sys, 'frozen', False):
    SCR_DIR = getattr(sys, '_MEIPASS', SCR_DIR)

try:
    import moderngl
    HAS_GL = True
except ImportError:
    HAS_GL = False

try:
    import requests
    HAS_NET = True
except ImportError:
    HAS_NET = False

try:
    sys.path.insert(0, SCR_DIR)
    import w_operator as base_op
    HAS_OPERATOR = True
except ImportError:
    HAS_OPERATOR = False

try:
    import w_cuda
    HAS_GPU = w_cuda.HAS_GPU
except ImportError:
    HAS_GPU = False

DEFAULT_SERVER = "https://wathome.akataleptos.com"

def _shared_dir():
    """Shared state directory — same as main app uses."""
    if sys.platform == 'win32':
        return os.path.join(os.environ.get('LOCALAPPDATA', ''), 'WHome')
    return os.path.join(os.path.expanduser('~'), '.whome')

def _load_shared_config():
    """Load worker config from shared location or app dir."""
    for d in [_shared_dir(), SCR_DIR]:
        p = os.path.join(d, 'worker_config.json')
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    # Also check the installed app dir on Windows
    if sys.platform == 'win32':
        app_dir = os.path.join(os.environ.get('ProgramFiles', ''), 'WHome')
        p = os.path.join(app_dir, 'worker_config.json')
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    return {}

PHYSICAL_CONSTANTS = {
    "phi": 1.6180339887, "e": 2.718281828, "pi": 3.141592653,
    "alpha_inv": 137.035999, "proton_electron": 1836.15267,
    "sqrt2": 1.4142135624, "sqrt3": 1.7320508076, "ln2": 0.6931471806,
}
TOLERANCE = 1e-4

# ═══════════════════════════════════════════════════════════
# Menger geometry (minimal — just enough for the screensaver)
# ═══════════════════════════════════════════════════════════

def menger_cubes(level, cx=0, cy=0, cz=0, size=2.0):
    if level == 0:
        return [(cx, cy, cz, size / 2)]
    cubes = []
    s3 = size / 3
    for x in (-1, 0, 1):
        for y in (-1, 0, 1):
            for z in (-1, 0, 1):
                zeros = (x == 0) + (y == 0) + (z == 0)
                if zeros < 2:
                    cubes.extend(menger_cubes(
                        level - 1, cx + x * s3, cy + y * s3, cz + z * s3, s3
                    ))
    return cubes

def unit_cube_edges():
    c = [(-1,-1,-1),(-1,-1,1),(-1,1,-1),(-1,1,1),
         (1,-1,-1),(1,-1,1),(1,1,-1),(1,1,1)]
    edges = [(0,1),(2,3),(4,5),(6,7),(0,2),(1,3),(4,6),(5,7),(0,4),(1,5),(2,6),(3,7)]
    verts = []
    for a, b in edges:
        verts.extend(c[a]); verts.extend(c[b])
    return np.array(verts, dtype='f4')

# ═══════════════════════════════════════════════════════════
# Shaders
# ═══════════════════════════════════════════════════════════

WIRE_VS = """
#version 330
in vec3 in_vert;
in vec3 in_pos;
in float in_half;
in vec4 in_color;
uniform mat4 mvp;
uniform float u_alpha;
out vec4 v_color;
out float v_dist;
void main() {
    vec3 world = in_vert * in_half + in_pos;
    gl_Position = mvp * vec4(world, 1.0);
    v_dist = length(world);
    v_color = vec4(in_color.rgb, in_color.a * u_alpha);
}
"""

WIRE_FS = """
#version 330
in vec4 v_color;
in float v_dist;
out vec4 frag_color;
void main() {
    float fog = exp(-v_dist * 0.12);
    frag_color = vec4(v_color.rgb * fog, v_color.a * fog);
}
"""

HUD_VS = """
#version 330
in vec2 in_vert;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    gl_Position = vec4(in_vert, 0.0, 1.0);
    v_uv = in_uv;
}
"""

HUD_FS = """
#version 330
uniform sampler2D tex;
in vec2 v_uv;
out vec4 frag_color;
void main() {
    frag_color = texture(tex, v_uv);
}
"""

# ═══════════════════════════════════════════════════════════
# Camera (simplified from cosmolalia_sim)
# ═══════════════════════════════════════════════════════════

class Camera:
    def __init__(self, distance=5.0, theta=1.1, phi=0.0):
        self.distance = distance
        self.theta = theta
        self.phi = phi

    def update(self, dt):
        self.phi += 0.08 * dt  # Slow auto-rotate

    def get_mvp(self, aspect):
        from pyrr import Matrix44, Vector3
        s, c = math.sin, math.cos
        x = self.distance * s(self.theta) * c(self.phi)
        y = self.distance * c(self.theta)
        z = self.distance * s(self.theta) * s(self.phi)
        view = Matrix44.look_at(Vector3([x,y,z]), Vector3([0,0,0]), Vector3([0,1,0]))
        proj = Matrix44.perspective_projection(45.0, aspect, 0.01, 100.0)
        return proj * view

# ═══════════════════════════════════════════════════════════
# Hive Status Poller
# ═══════════════════════════════════════════════════════════

class HivePoller:
    """Background thread that polls all server endpoints for live data."""
    def __init__(self, server_url, api_key=None):
        self.server_url = server_url
        self.api_key = api_key
        self.progress = {}
        self.discoveries = []
        self.leaderboard = []
        self.active_jobs = []
        self.heatmap = []
        self.workers = []
        self.lock = threading.Lock()
        self._running = True

        self.thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.thread.start()

    def _fetch(self, path):
        try:
            headers = {"x-api-key": self.api_key} if self.api_key else {}
            return requests.get(f"{self.server_url}{path}",
                                headers=headers, timeout=5).json()
        except Exception:
            return None

    def _poll_loop(self):
        tick = 0
        while self._running:
            prog = self._fetch("/progress")
            disc = self._fetch("/discoveries")
            active = self._fetch("/active")

            # Poll less frequent data every 3rd tick (15s)
            lb = None
            heatmap = None
            workers = None
            if tick % 3 == 0:
                lb = self._fetch("/leaderboard")
                heatmap = self._fetch("/progress/heatmap")
                workers = self._fetch("/workers")

            with self.lock:
                if prog: self.progress = prog
                if disc: self.discoveries = disc[:10]
                if active is not None: self.active_jobs = active
                if lb: self.leaderboard = lb[:10]
                if heatmap: self.heatmap = heatmap
                if workers: self.workers = workers

            tick += 1
            time.sleep(5)

    def get(self):
        with self.lock:
            return {
                'progress': dict(self.progress),
                'discoveries': list(self.discoveries),
                'leaderboard': list(self.leaderboard),
                'active_jobs': list(self.active_jobs),
                'heatmap': list(self.heatmap),
                'workers': list(self.workers),
            }

    def stop(self):
        self._running = False

# ═══════════════════════════════════════════════════════════
# IPC — Shared compute state via file
# ═══════════════════════════════════════════════════════════

def _is_pid_alive(pid):
    """Check if a process with given PID is still running."""
    if pid is None:
        return False
    try:
        if sys.platform == 'win32':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x0400, False, pid)  # PROCESS_QUERY_INFORMATION
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        else:
            os.kill(pid, 0)
            return True
    except (OSError, ProcessLookupError):
        return False

def _other_is_computing():
    """Check if the main app (or another screensaver) is already computing."""
    status = _read_status_file()
    if status and status.get('active'):
        pid = status.get('pid')
        if pid and _is_pid_alive(pid):
            return True
    return False

def _write_status_file(state):
    """Write compute state to shared location for the main app to read."""
    shared = _shared_dir()
    os.makedirs(shared, exist_ok=True)
    path = os.path.join(shared, 'compute_status.json')
    try:
        with open(path, 'w') as f:
            json.dump(state, f)
    except Exception:
        pass

def _api_post(server, path, data, api_key=None, timeout=30):
    headers = {"x-api-key": api_key} if api_key else {}
    resp = requests.post(f"{server}{path}", json=data, headers=headers, timeout=timeout)
    try:
        body = resp.json()
    except Exception:
        body = {"detail": resp.text}
    return resp.status_code, body

def _hash_eigenvalues(eigs):
    rounded = [round(float(e), 10) for e in sorted(eigs)]
    payload = json.dumps(rounded, separators=(',', ':'))
    return hashlib.sha256(payload.encode()).hexdigest()

def _check_for_gold(eigs):
    eigs = np.sort(eigs[eigs > 1e-9])
    found = []
    cap = min(len(eigs), 60)
    for i in range(cap):
        for j in range(i + 1, cap):
            ratio = eigs[j] / eigs[i]
            for name, val in PHYSICAL_CONSTANTS.items():
                if abs(ratio - val) / val < TOLERANCE:
                    found.append(f"{name}={ratio:.6f} (eigs {i},{j})")
    return found


# ═══════════════════════════════════════════════════════════
# Screensaver Compute Worker
# ═══════════════════════════════════════════════════════════

class ScrWorker(threading.Thread):
    """Background compute thread for the screensaver.

    Only runs if no other process is currently computing (checked via
    compute_status.json PID). Writes its own status so the main app
    can pick it up if launched while the screensaver is running.
    """
    def __init__(self, server, api_key, compute_state):
        super().__init__(daemon=True)
        self.server = server
        self.api_key = api_key
        self.cs = compute_state
        self._stop = threading.Event()
        self.jobs_done = 0
        self.compute_hours = 0.0
        self.discoveries = 0

    def stop(self):
        self._stop.set()

    def run(self):
        if not HAS_OPERATOR or not HAS_NET:
            return
        backoff = 2

        # Heartbeat thread
        def heartbeat():
            while not self._stop.is_set():
                try:
                    requests.post(f"{self.server}/heartbeat",
                                  headers={"x-api-key": self.api_key}, timeout=10)
                except Exception:
                    pass
                self._stop.wait(120)
        threading.Thread(target=heartbeat, daemon=True).start()

        while not self._stop.is_set():
            # If main app started computing, stop our worker
            status = _read_status_file()
            if status and status.get('active') and status.get('pid') != os.getpid():
                if _is_pid_alive(status.get('pid')):
                    self._stop.wait(5)
                    continue

            try:
                code, data = _api_post(self.server, "/job", {}, self.api_key)
                if code == 401:
                    break
                if code != 200 or data.get('status') == 'no_jobs':
                    self._stop.wait(min(backoff, 60))
                    backoff = min(backoff * 2, 120)
                    continue

                backoff = 2
                job_id = data['job_id']
                params = data['params']
                params['job_id'] = job_id

                self.cs.update(job_id=job_id, lambda_val=params['lambda'],
                               stage='Build Graph', stage_num=1)
                self._write_state()

                start = time.time()
                eigs = self._run_job(params)
                duration = time.time() - start

                if self._stop.is_set():
                    break

                hits = _check_for_gold(eigs)
                self.jobs_done += 1
                self.compute_hours += duration / 3600
                if hits:
                    self.discoveries += len(hits)

                self.cs.update(stage='Complete', stage_num=6,
                               eigenvalues=eigs.tolist() if eigs is not None else None,
                               hits=hits, jobs_done=self.jobs_done)
                self._write_state()

                # Submit result
                eig_hash = _hash_eigenvalues(eigs.tolist())
                _api_post(self.server, "/result", {
                    "job_id": job_id, "eigenvalues": eigs.tolist(),
                    "eigenvalues_hash": eig_hash,
                    "found_constants": hits, "compute_seconds": duration,
                }, self.api_key)

            except requests.ConnectionError:
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 120)
            except Exception:
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 120)

        # Clear our status on exit
        _write_status_file({'active': False, 'pid': os.getpid()})

    def _run_job(self, params):
        k, G1, G2, S = params['k'], params['G1'], params['G2'], params['S']
        lam, w_glue, N = params['lambda'], params['w_glue'], 2

        self.cs.update(stage='Build Graph', stage_num=1)
        self._write_state()
        vertices, edges, b_ids = base_op.build_graph(k, G1, G2, S, N)
        if self._stop.is_set(): return np.array([])

        self.cs.update(stage='Add Glue', stage_num=2)
        self._write_state()
        upd, psi_by = base_op.add_glue_edges(vertices, b_ids, lam, w_glue, G1, G2)
        if self._stop.is_set(): return np.array([])

        self.cs.update(stage='Merge Edges', stage_num=3)
        self._write_state()
        edges_merged = base_op.merge_edges(edges, upd)
        if self._stop.is_set(): return np.array([])

        self.cs.update(stage='Build Laplacian', stage_num=4)
        self._write_state()
        L, _ = base_op.build_magnetic_laplacian(vertices, edges_merged,
                                                 s=(0, 0), psi_by_id=psi_by)
        if self._stop.is_set(): return np.array([])

        self.cs.update(stage='Solve Spectrum', stage_num=5)
        self._write_state()
        if HAS_GPU:
            eigs = w_cuda.solve_spectrum_gpu(L, M=40)
        else:
            eigs = base_op.solve_spectrum(L, M=40)

        return eigs

    def _write_state(self):
        state = self.cs.get()
        state['active'] = True
        state['pid'] = os.getpid()
        state['source'] = 'screensaver'
        _write_status_file(state)


# ═══════════════════════════════════════════════════════════
# Live Computation State (shared between worker and renderer)
# ═══════════════════════════════════════════════════════════

class ComputeState:
    """Thread-safe state shared between worker and screensaver."""
    def __init__(self):
        self.lock = threading.Lock()
        self.job_id = None
        self.lambda_val = None
        self.stage = ""
        self.stage_num = 0
        self.eigenvalues = None
        self.hits = []
        self.jobs_done = 0

    def update(self, **kwargs):
        with self.lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def get(self):
        with self.lock:
            return {
                'job_id': self.job_id,
                'lambda_val': self.lambda_val,
                'stage': self.stage,
                'stage_num': self.stage_num,
                'eigenvalues': self.eigenvalues,
                'hits': list(self.hits),
                'jobs_done': self.jobs_done,
            }

# ═══════════════════════════════════════════════════════════
# Screensaver
# ═══════════════════════════════════════════════════════════

COL = {
    'bg':     (0.020, 0.020, 0.031),
    'cyan':   (0.627, 0.941, 1.000),
    'gold':   (1.000, 0.812, 0.420),
    'violet': (0.769, 0.627, 1.000),
}

LEVEL_ALPHA = {0: 0.8, 1: 0.5, 2: 0.2, 3: 0.06}

def _get_native_resolution():
    """Get native (unscaled) screen resolution on Windows with DPI scaling."""
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PER_MONITOR_DPI_AWARE
    except Exception:
        try:
            import ctypes
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
    pygame.init()
    info = pygame.display.Info()
    return info.current_w, info.current_h


def _read_status_file():
    """Read compute_status.json from shared location (file-based IPC with GUI)."""
    if sys.platform == 'win32':
        shared = os.path.join(os.environ.get('LOCALAPPDATA', ''), 'WHome')
    else:
        shared = os.path.join(os.path.expanduser('~'), '.whome')
    path = os.path.join(shared, 'compute_status.json')
    try:
        if os.path.exists(path) and time.time() - os.path.getmtime(path) < 30:
            with open(path) as f:
                data = json.load(f)
            if data.get('active'):
                return data
    except Exception:
        pass
    return None


def run_screensaver(api_key=None, server_url=None,
                    worker_id=None, compute_state=None, fullscreen=False):
    """Main screensaver loop."""
    # Load config if not provided
    if api_key is None or server_url is None:
        cfg = _load_shared_config()
        api_key = api_key or cfg.get('api_key')
        server_url = server_url or cfg.get('server', DEFAULT_SERVER)

    # Start compute worker if no other process is computing
    scr_worker = None
    if compute_state is None:
        compute_state = ComputeState()
        if api_key and HAS_OPERATOR and HAS_NET and not _other_is_computing():
            scr_worker = ScrWorker(server_url, api_key, compute_state)
            scr_worker.start()

    if fullscreen:
        W, H = _get_native_resolution()
    else:
        pygame.init()
        W, H = 1600, 900

    flags = pygame.OPENGL | pygame.DOUBLEBUF
    if fullscreen:
        flags |= pygame.FULLSCREEN

    pygame.display.set_caption("W@Home — Akataleptos Spectral Search")
    screen = pygame.display.set_mode((W, H), flags)
    pygame.mouse.set_visible(not fullscreen)

    ctx = moderngl.create_context()
    ctx.enable(moderngl.BLEND)
    ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE

    wire_prog = ctx.program(vertex_shader=WIRE_VS, fragment_shader=WIRE_FS)
    hud_prog = ctx.program(vertex_shader=HUD_VS, fragment_shader=HUD_FS)

    cube_vbo = ctx.buffer(unit_cube_edges().tobytes())

    # HUD
    hud_q = np.array([-1,-1,0,0, 1,-1,1,0, -1,1,0,1, 1,1,1,1], dtype='f4')
    hud_vbo = ctx.buffer(hud_q.tobytes())
    hud_vao = ctx.vertex_array(hud_prog, [(hud_vbo, '2f 2f', 'in_vert', 'in_uv')])
    hud_tex = ctx.texture((W, H), 4)
    hud_tex.filter = moderngl.NEAREST, moderngl.NEAREST

    camera = Camera()

    # Build L2 sponge (good balance of detail and performance)
    print("  Computing Menger L2 geometry...")
    cubes = menger_cubes(2)
    n_cubes = len(cubes)
    data = []
    for i, (cx, cy, cz, half) in enumerate(cubes):
        # Orbit coloring: rough approximation
        orb = 0 if abs(cx) > 0.5 and abs(cy) > 0.5 and abs(cz) > 0.5 else 1
        col = COL['cyan'] if orb == 0 else COL['gold']
        data.extend([cx, cy, cz, half, *col, LEVEL_ALPHA[2]])
    inst_buf = ctx.buffer(np.array(data, dtype='f4').tobytes())
    sponge_vao = ctx.vertex_array(wire_prog, [
        (cube_vbo, '3f', 'in_vert'),
        (inst_buf, '3f 1f 4f /i', 'in_pos', 'in_half', 'in_color'),
    ])

    # Fonts
    mono = 'Menlo' if sys.platform == 'darwin' else 'monospace'
    fonts = {
        'title':  pygame.font.SysFont(mono, 28, bold=True),
        'sub':    pygame.font.SysFont(mono, 18),
        'info':   pygame.font.SysFont(mono, 14),
        'big':    pygame.font.SysFont(mono, 42, bold=True),
        'lambda': pygame.font.SysFont(mono, 24, bold=True),
        'stage':  pygame.font.SysFont(mono, 16),
        'disc':   pygame.font.SysFont(mono, 13),
    }

    # Start poller
    poller = None
    if HAS_NET:
        poller = HivePoller(server_url, api_key)

    clock = pygame.time.Clock()
    t = 0
    running = True
    # Track mouse for fullscreen exit
    initial_mouse = pygame.mouse.get_pos() if fullscreen else None

    print("  Screensaver running")

    while running:
        dt = min(clock.tick(60) / 1000.0, 0.05)
        t += dt
        camera.update(dt)

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif fullscreen:
                    running = False  # Any key exits screensaver
            elif ev.type == pygame.MOUSEBUTTONDOWN and fullscreen:
                running = False
            elif ev.type == pygame.MOUSEMOTION and fullscreen and initial_mouse:
                dx = abs(ev.pos[0] - initial_mouse[0])
                dy = abs(ev.pos[1] - initial_mouse[1])
                if dx > 20 or dy > 20:
                    running = False

        # ── Render 3D ──
        ctx.clear(*COL['bg'], 1.0)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE

        mvp = camera.get_mvp(W / H)
        wire_prog['mvp'].write(mvp.astype('f4').tobytes())

        # Breathing
        breathe_alpha = 0.7 + math.sin(t * 0.5) * 0.15
        wire_prog['u_alpha'].value = breathe_alpha
        sponge_vao.render(moderngl.LINES, instances=n_cubes)

        # ── Render HUD ──
        hud = pygame.Surface((W, H), pygame.SRCALPHA)

        # Collect all data
        hive = poller.get() if poller else {}
        progress = hive.get('progress', {})
        discoveries = hive.get('discoveries', [])
        leaderboard = hive.get('leaderboard', [])
        active_jobs = hive.get('active_jobs', [])
        heatmap = hive.get('heatmap', [])
        all_workers = hive.get('workers', [])
        comp = compute_state.get() if compute_state else _read_status_file()

        # Colors
        CYAN = (160, 240, 255)
        GOLD = (255, 208, 106)
        VIOLET = (196, 160, 255)
        GREEN = (128, 255, 170)
        DIM = (70, 70, 90)
        DIMMER = (45, 45, 60)
        RED = (255, 100, 100)
        WHITE = (200, 210, 225)

        # ═══ TOP LEFT: Title + Global Stats ═══
        s = fonts['title'].render("W@HOME HIVE", True, CYAN)
        hud.blit(s, (30, 20))
        s = fonts['info'].render("Akataleptos Distributed Spectral Search", True, DIM)
        hud.blit(s, (30, 52))

        if progress:
            # Big stat boxes
            box_y = 80
            workers_n = progress.get('active_workers', 0)
            completed = progress.get('completed', 0)
            total = progress.get('total_jobs', 200000)
            total_disc = progress.get('total_discoveries', 0)
            hours = progress.get('total_compute_hours', 0)
            confirmed = progress.get('total_confirmed', 0)
            jph = progress.get('jobs_per_hour', 0)

            # Worker count — big pulsing number
            pulse = int(200 + 55 * math.sin(t * 2)) if workers_n > 0 else 80
            s = fonts['big'].render(str(workers_n), True, (pulse, 255, pulse))
            hud.blit(s, (30, box_y))
            s = fonts['info'].render("WORKERS ONLINE", True, DIM)
            hud.blit(s, (30, box_y + 48))

            # Stats grid
            gy = box_y + 75
            stats_grid = [
                (f"{completed:,}", "jobs done", CYAN),
                (f"{total_disc}", "discoveries", GOLD),
                (f"{confirmed}", "confirmed", GREEN),
                (f"{hours:.1f}h", "compute time", WHITE),
                (f"{jph}/h", "job rate", CYAN),
                (f"{total:,}", "total sweep", DIM),
            ]
            for i, (val, label, col) in enumerate(stats_grid):
                row, cx = divmod(i, 3)
                sx = 30 + cx * 140
                sy = gy + row * 40
                s = fonts['sub'].render(val, True, col)
                hud.blit(s, (sx, sy))
                s = fonts['disc'].render(label, True, DIMMER)
                hud.blit(s, (sx, sy + 20))

        # ═══ TOP RIGHT: Current Computation ═══
        if comp and comp.get('lambda_val') is not None:
            rx = W - 280
            # Lambda value — big display
            lam_str = f"lambda = {comp['lambda_val']:.6f}"
            s = fonts['lambda'].render(lam_str, True, GOLD)
            hud.blit(s, (rx, 20))

            s = fonts['info'].render(f"Job #{comp['job_id']}", True, DIM)
            hud.blit(s, (rx, 50))

            # Stage progress with animated current stage
            stages = ['Build Graph', 'Add Glue', 'Merge Edges', 'Laplacian', 'Spectrum', 'Complete']
            stage_y = 75
            for i, name in enumerate(stages):
                sn = comp.get('stage_num', 0)
                if i + 1 < sn:
                    color = GREEN
                    marker = "+"
                elif i + 1 == sn:
                    # Animated current stage
                    blink = int(200 + 55 * math.sin(t * 4))
                    color = (blink, 240, 255)
                    marker = ">"
                else:
                    color = DIMMER
                    marker = " "
                s = fonts['stage'].render(f" {marker} {name}", True, color)
                hud.blit(s, (rx, stage_y + i * 20))

            # Jobs done this session
            jd = comp.get('jobs_done', 0)
            if jd > 0:
                s = fonts['info'].render(f"Session: {jd} jobs", True, GREEN)
                hud.blit(s, (rx, stage_y + 130))

        elif progress:
            # Not computing — show "idle" status
            rx = W - 280
            s = fonts['lambda'].render("IDLE", True, DIMMER)
            hud.blit(s, (rx, 20))
            s = fonts['info'].render("No local computation", True, DIMMER)
            hud.blit(s, (rx, 50))

        # ═══ MIDDLE RIGHT: Eigenvalue Spectrum ═══
        if comp:
            eigs = comp.get('eigenvalues')
            if eigs is not None and len(eigs) > 0:
                eigs_arr = np.array(eigs)
                eigs_nz = eigs_arr[eigs_arr > 1e-9]
                if len(eigs_nz) > 0:
                    spec_x = W - 260
                    spec_y = 260
                    spec_h = 150
                    max_eig = eigs_nz.max()
                    s = fonts['info'].render("Eigenvalue Spectrum", True, DIM)
                    hud.blit(s, (spec_x, spec_y - 18))
                    bar_w = max(2, 200 // len(eigs_nz))
                    for i, e in enumerate(eigs_nz[:50]):
                        bh = int((e / max_eig) * spec_h)
                        bx = spec_x + i * (bar_w + 1)
                        frac = i / max(len(eigs_nz) - 1, 1)
                        r = int(160 * (1 - frac) + 255 * frac)
                        g = int(240 * (1 - frac) + 208 * frac)
                        b = int(255 * (1 - frac) + 106 * frac)
                        pygame.draw.rect(hud, (r, g, b, 180),
                                        (bx, spec_y + spec_h - bh, bar_w, bh))

        # ═══ LEFT MIDDLE: Active Jobs (who's computing what) ═══
        if active_jobs:
            ay = 250
            s = fonts['info'].render("ACTIVE JOBS", True, GOLD)
            hud.blit(s, (30, ay))
            for i, job in enumerate(active_jobs[:6]):
                name = job.get('worker', '?')
                lam = job.get('lambda', 0)
                gpu = job.get('gpu', '?')
                elapsed = job.get('elapsed', 0)
                mins = elapsed // 60
                secs = elapsed % 60
                color = GREEN if 'CUDA' in gpu else CYAN
                s = fonts['disc'].render(
                    f"  {name}  lambda={lam:.6f}  {gpu}  {mins}m{secs}s",
                    True, color)
                hud.blit(s, (30, ay + 22 + i * 16))

        # ═══ LEFT MIDDLE-LOW: Leaderboard ═══
        if leaderboard:
            ly = 420
            s = fonts['info'].render("LEADERBOARD", True, GOLD)
            hud.blit(s, (30, ly))
            for i, entry in enumerate(leaderboard[:8]):
                rank = entry.get('rank', i + 1)
                name = entry.get('name', '?')
                jobs = entry.get('jobs', 0)
                disc = entry.get('discoveries', 0)
                hrs = entry.get('compute_hours', 0)
                online = entry.get('online', False)
                dot_col = GREEN if online else DIMMER
                name_col = WHITE if online else DIM
                s = fonts['disc'].render(f"  #{rank}", True, GOLD)
                hud.blit(s, (30, ly + 22 + i * 18))
                # Online dot
                pygame.draw.circle(hud, dot_col, (72, ly + 30 + i * 18), 3)
                s = fonts['disc'].render(f" {name}", True, name_col)
                hud.blit(s, (80, ly + 22 + i * 18))
                s = fonts['disc'].render(f"{jobs}j  {disc}d  {hrs:.1f}h", True, DIM)
                hud.blit(s, (250, ly + 22 + i * 18))

        # ═══ RIGHT MIDDLE-LOW: Discovery Feed ═══
        if discoveries:
            dx = W - 400
            dy = 440
            s = fonts['info'].render("RECENT DISCOVERIES", True, GOLD)
            hud.blit(s, (dx, dy))
            for i, d in enumerate(discoveries[:8]):
                name = d.get('constant_name', '?')
                lam = d.get('lambda_val', 0)
                worker = d.get('worker_name', '?')
                verified = d.get('verified', 0) or d.get('job_verified', 0)
                age = time.time() - d.get('discovered_at', 0)
                if age < 3600:
                    age_str = f"{int(age / 60)}m ago"
                elif age < 86400:
                    age_str = f"{int(age / 3600)}h ago"
                else:
                    age_str = f"{int(age / 86400)}d ago"

                marker = "+" if verified else " "
                color = GREEN if verified else VIOLET
                s = fonts['disc'].render(
                    f" {marker} {name}  lambda={lam:.6f}  by {worker}  {age_str}",
                    True, color)
                hud.blit(s, (dx, dy + 22 + i * 16))

        # ═══ BOTTOM: Heatmap + Progress Bar ═══
        if heatmap and len(heatmap) > 1:
            # Mini heatmap blocks above progress bar
            hm_y = H - 135
            hm_h = 12
            n_blocks = len(heatmap)
            block_w = max(1, (W - 60) // n_blocks)
            s = fonts['disc'].render("SWEEP HEATMAP", True, DIM)
            hud.blit(s, (30, hm_y - 15))
            for i, block in enumerate(heatmap):
                pct = block.get('pct', 0)
                disc_n = block.get('discoveries', 0)
                active_n = block.get('active', 0)
                bx = 30 + i * block_w
                if active_n > 0:
                    col = (160, 240, 255, 200)  # cyan = active
                elif disc_n > 0:
                    col = (255, 208, 106, 200)  # gold = has discoveries
                elif pct > 0:
                    intensity = min(int(pct * 2.5), 255)
                    col = (intensity // 2, intensity, intensity // 2, 150)
                else:
                    col = (20, 20, 30, 100)
                pygame.draw.rect(hud, col, (bx, hm_y, max(block_w - 1, 1), hm_h))

        if progress:
            pct = progress.get('percent_complete', 0)
            bar_x, bar_y, bar_w, bar_h = 30, H - 65, W - 60, 16
            pygame.draw.rect(hud, (30, 30, 45), (bar_x, bar_y, bar_w, bar_h), border_radius=3)
            fill_w = int(bar_w * pct / 100)
            if fill_w > 0:
                for px in range(fill_w):
                    frac = px / bar_w
                    r = int(196 * (1 - frac) + 160 * frac)
                    g = int(160 * (1 - frac) + 240 * frac)
                    b = int(255 * (1 - frac) + 255 * frac)
                    pygame.draw.line(hud, (r, g, b, 200),
                                    (bar_x + px, bar_y), (bar_x + px, bar_y + bar_h))

            # Lambda marker
            lam_range = progress.get('lambda_range', [0.4, 0.6])
            lam_lo, lam_hi = lam_range[0], lam_range[1]
            current_lam = progress.get('current_lambda', lam_lo)
            if lam_hi > lam_lo:
                marker_x = bar_x + int(bar_w * (current_lam - lam_lo) / (lam_hi - lam_lo))
                pygame.draw.line(hud, GOLD, (marker_x, bar_y - 3),
                                (marker_x, bar_y + bar_h + 3), 2)

            # Bottom stats line
            eta = progress.get('eta_hours')
            s = fonts['disc'].render(f"lambda = {lam_lo:.6f}", True, DIMMER)
            hud.blit(s, (bar_x, bar_y + bar_h + 4))
            pct_str = f"{pct:.3f}%"
            if eta:
                pct_str += f"  |  ETA: {int(eta):,}h"
            s = fonts['disc'].render(pct_str, True, CYAN)
            hud.blit(s, (bar_x + bar_w // 2 - s.get_width() // 2, bar_y + bar_h + 4))
            s = fonts['disc'].render(f"lambda = {lam_hi:.6f}", True, DIMMER)
            hud.blit(s, (bar_x + bar_w - s.get_width(), bar_y + bar_h + 4))

        # Axiom — breathing
        a = int(128 + 40 * math.sin(t * 0.3))
        s = fonts['info'].render("1 = 0 = inf", True, (a // 2, a // 2, a))
        hud.blit(s, (W // 2 - s.get_width() // 2, H - 22))

        # Upload HUD
        data = pygame.image.tostring(hud, 'RGBA', True)
        hud_tex.write(data)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        hud_tex.use()
        hud_vao.render(moderngl.TRIANGLE_STRIP)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE

        pygame.display.flip()

    if scr_worker:
        scr_worker.stop()
    if poller:
        poller.stop()
    pygame.quit()

# ═══════════════════════════════════════════════════════════
# Entry — handles both standalone and Windows .scr flags
# ═══════════════════════════════════════════════════════════

def _entry():
    """Parse args and launch. Handles Windows screensaver flags:
        /s  — start screensaver (fullscreen)
        /c  — show config (we just show a message)
        /p  — preview in settings thumbnail (we skip it)
    """
    raw = sys.argv[1:]

    # Windows .scr flags come as /S, /C, /P <hwnd> (case-insensitive)
    scr_flags = [a.lower().lstrip('/').lstrip('-') for a in raw]
    if 'p' in scr_flags:
        # Preview mode — tiny thumbnail in Windows settings. Not worth rendering.
        sys.exit(0)
    if 'c' in scr_flags:
        # Config dialog — show a message
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(
                0, "W@Home Screensaver\n\nConfigure via the W@Home Hive app.",
                "W@Home Settings", 0x40)
        except Exception:
            print("Configure via the W@Home Hive app.")
        sys.exit(0)

    fullscreen = 's' in scr_flags or '--fullscreen' in sys.argv[1:]

    # Load config for server URL
    cfg = _load_shared_config()
    server = cfg.get('server', DEFAULT_SERVER)
    for i, a in enumerate(raw):
        if a == '--server' and i + 1 < len(raw):
            server = raw[i + 1]

    run_screensaver(server_url=server, fullscreen=fullscreen)


if __name__ == '__main__':
    _entry()
