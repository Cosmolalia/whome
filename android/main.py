"""
W@Home Hive — Android Client
Kivy app wrapping the W@Home distributed eigenvalue computation client.
"""

import os
import sys
import json
import hashlib
import time
import threading
import platform

# Ensure our directory is importable
app_dir = os.path.dirname(os.path.abspath(__file__))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

import numpy as np

from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.utils import get_color_from_hex
from kivy.metrics import dp

# Try to import requests; if unavailable, use urllib
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    import urllib.request
    import urllib.error

# Try scipy first, then fallback
try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    sp = None
    spla = None

import w_operator

# ═══════════════════════════════════════════════════════════
# HTTP helpers (works with or without requests library)
# ═══════════════════════════════════════════════════════════

def http_post(url, json_data=None, headers=None, timeout=30):
    """POST request that works with requests or urllib."""
    if HAS_REQUESTS:
        resp = requests.post(url, json=json_data, headers=headers, timeout=timeout)
        return resp.status_code, resp.text, resp
    else:
        data = json.dumps(json_data).encode('utf-8') if json_data else None
        req = urllib.request.Request(url, data=data, method='POST')
        req.add_header('Content-Type', 'application/json')
        if headers:
            for k, v in headers.items():
                req.add_header(k, v)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode('utf-8')
                return resp.status, body, None
        except urllib.error.HTTPError as e:
            body = e.read().decode('utf-8')
            return e.code, body, None

def http_post_json(url, json_data=None, headers=None, timeout=30):
    """POST and parse JSON response."""
    code, body, raw = http_post(url, json_data, headers, timeout)
    try:
        parsed = json.loads(body)
    except Exception:
        parsed = {"raw": body}
    return code, parsed

# ═══════════════════════════════════════════════════════════
# Pure-Python sparse eigensolver fallback
# ═══════════════════════════════════════════════════════════

def solve_spectrum_fallback(L_sparse, M=40, tol=1e-8):
    """
    Fallback eigensolver when scipy.sparse.linalg.eigsh is unavailable.
    Uses numpy dense eigvalsh on small matrices, or Lanczos on larger ones.
    """
    n = L_sparse.shape[0]
    k = min(M + 1, n)

    if n <= 2000:
        # Small enough for dense solve
        if hasattr(L_sparse, 'toarray'):
            dense = L_sparse.toarray()
        else:
            dense = np.array(L_sparse)
        vals = np.linalg.eigvalsh(dense)
        return np.sort(np.real(vals))[:k]
    else:
        # Lanczos iteration (simplified, pure numpy)
        return _lanczos_smallest(L_sparse, k, tol, max_iter=300)


def _lanczos_smallest(A, k, tol=1e-8, max_iter=300):
    """
    Simple Lanczos for smallest eigenvalues of a Hermitian sparse matrix.
    Not as robust as ARPACK, but works without scipy.
    """
    n = A.shape[0]
    m = min(max(2 * k + 20, 60), n)  # Krylov subspace dimension

    # Random starting vector
    v = np.random.randn(n).astype(np.float64)
    v = v / np.linalg.norm(v)

    V = np.zeros((n, m), dtype=np.float64)
    alpha = np.zeros(m, dtype=np.float64)
    beta = np.zeros(m, dtype=np.float64)

    V[:, 0] = v

    for j in range(m):
        w = A.dot(V[:, j]).real
        alpha[j] = np.dot(V[:, j], w)
        if j == 0:
            w = w - alpha[j] * V[:, j]
        else:
            w = w - alpha[j] * V[:, j] - beta[j] * V[:, j - 1]

        # Re-orthogonalize
        for i in range(j + 1):
            w -= np.dot(V[:, i], w) * V[:, i]

        beta_next = np.linalg.norm(w)
        if beta_next < 1e-14:
            m = j + 1
            break
        if j + 1 < m:
            beta[j + 1] = beta_next
            V[:, j + 1] = w / beta_next

    # Build tridiagonal matrix
    T = np.diag(alpha[:m])
    for i in range(m - 1):
        T[i, i + 1] = beta[i + 1]
        T[i + 1, i] = beta[i + 1]

    # Solve tridiagonal eigenvalue problem (small, dense)
    eigs = np.linalg.eigvalsh(T)
    return np.sort(eigs)[:k]


# Monkey-patch w_operator to use our fallback if scipy is missing
if not HAS_SCIPY:
    # Build a minimal sparse matrix class using numpy
    class SimpleSparse:
        """Minimal COO/CSR-like sparse matrix backed by dense numpy."""
        def __init__(self, dense):
            self.data = np.array(dense, dtype=np.complex128)
            self.shape = self.data.shape

        def dot(self, v):
            return self.data.dot(v)

        def toarray(self):
            return self.data

    # Override w_operator's build_magnetic_laplacian to return dense
    _orig_build_mag_lap = w_operator.build_magnetic_laplacian

    def _patched_build_magnetic_laplacian(vertices, edges, s, psi_by_id):
        """Build Laplacian as dense matrix when scipy.sparse is unavailable."""
        v_ids = sorted(vertices.keys())
        id_to_idx = {vid: i for i, vid in enumerate(v_ids)}
        n = len(v_ids)
        TAU = 2 * 3.141592653589793

        deg = np.zeros(n, dtype=np.float64)
        L = np.zeros((n, n), dtype=np.complex128)

        for (u, v), rec in edges.items():
            w = float(rec["w"])
            iu = id_to_idx[u]
            iv = id_to_idx[v]
            deg[iu] += w
            deg[iv] += w

            a_u_to_v = 0.0
            if rec.get("is_glue", False):
                owner = rec.get("owner")
                if owner is not None:
                    psi = psi_by_id.get(owner)
                    if psi is not None:
                        a = TAU * (s[0] * psi[0] + s[1] * psi[1])
                        if owner == u:
                            a_u_to_v = a
                        elif owner == v:
                            a_u_to_v = -a
                        else:
                            a_u_to_v = a

            import cmath
            val_uv = w * cmath.exp(1j * a_u_to_v)
            val_vu = val_uv.conjugate()

            L[iu, iv] -= val_uv
            L[iv, iu] -= val_vu

        for i in range(n):
            L[i, i] = deg[i]

        return SimpleSparse(L), id_to_idx

    w_operator.build_magnetic_laplacian = _patched_build_magnetic_laplacian
    w_operator.solve_spectrum = solve_spectrum_fallback


# ═══════════════════════════════════════════════════════════
# Constants & config
# ═══════════════════════════════════════════════════════════

DEFAULT_SERVER = "https://wathome.akataleptos.com"

CONSTANTS = {
    "phi": 1.6180339887,
    "e": 2.718281828,
    "pi": 3.141592653,
    "alpha_inv": 137.035999,
    "proton_electron": 1836.15267,
    "sqrt2": 1.4142135624,
    "sqrt3": 1.7320508076,
    "ln2": 0.6931471806,
}
TOLERANCE = 1e-4
MIN_BACKOFF = 2
MAX_BACKOFF = 120


def get_data_dir():
    """Get a writable directory for config/checkpoints."""
    try:
        from android.storage import app_storage_path
        return app_storage_path()
    except ImportError:
        return app_dir


def config_path():
    return os.path.join(get_data_dir(), "worker_config.json")


def checkpoint_path():
    return os.path.join(get_data_dir(), "checkpoint.json")


def load_config():
    p = config_path()
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


def save_config(cfg):
    with open(config_path(), 'w') as f:
        json.dump(cfg, f, indent=2)


def save_checkpoint(data):
    with open(checkpoint_path(), 'w') as f:
        json.dump(data, f, indent=2)


def load_checkpoint():
    p = checkpoint_path()
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def clear_checkpoint():
    p = checkpoint_path()
    if os.path.exists(p):
        os.remove(p)


def hash_eigenvalues(eigs):
    rounded = [round(float(e), 12) for e in sorted(eigs)]
    payload = json.dumps(rounded, separators=(',', ':'))
    return hashlib.sha256(payload.encode()).hexdigest()


def check_for_gold(eigs):
    eigs = np.sort(eigs[eigs > 1e-9])
    found = []
    n = len(eigs)
    cap = min(n, 60)
    for i in range(cap):
        for j in range(i + 1, cap):
            ratio = eigs[j] / eigs[i]
            for name, val in CONSTANTS.items():
                if abs(ratio - val) / val < TOLERANCE:
                    found.append(f"{name} (ratio={ratio:.6f}, i={i}, j={j})")
    return found


# ═══════════════════════════════════════════════════════════
# Screens
# ═══════════════════════════════════════════════════════════

BG_COLOR = get_color_from_hex('#1a1a2e')
ACCENT = get_color_from_hex('#00d4ff')
GOLD = get_color_from_hex('#ffd700')
TEXT_COLOR = get_color_from_hex('#e0e0e0')
DIM_COLOR = get_color_from_hex('#808080')
GREEN = get_color_from_hex('#00ff88')
RED = get_color_from_hex('#ff4444')


class LoginScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        layout = BoxLayout(orientation='vertical', padding=dp(20), spacing=dp(12))

        # Title
        layout.add_widget(Label(
            text='W@HOME HIVE',
            font_size=dp(28),
            bold=True,
            color=ACCENT,
            size_hint_y=None,
            height=dp(50),
        ))
        layout.add_widget(Label(
            text='Akataleptos Spectral Search',
            font_size=dp(14),
            color=DIM_COLOR,
            size_hint_y=None,
            height=dp(30),
        ))

        # Spacer
        layout.add_widget(Label(size_hint_y=0.1))

        # Name
        layout.add_widget(Label(
            text='Name',
            font_size=dp(14),
            color=TEXT_COLOR,
            size_hint_y=None,
            height=dp(24),
            halign='left',
        ))
        self.name_input = TextInput(
            hint_text='your-node-name',
            multiline=False,
            font_size=dp(16),
            size_hint_y=None,
            height=dp(44),
        )
        layout.add_widget(self.name_input)

        # Password
        layout.add_widget(Label(
            text='Password',
            font_size=dp(14),
            color=TEXT_COLOR,
            size_hint_y=None,
            height=dp(24),
            halign='left',
        ))
        self.password_input = TextInput(
            hint_text='password (min 4 chars)',
            password=True,
            multiline=False,
            font_size=dp(16),
            size_hint_y=None,
            height=dp(44),
        )
        layout.add_widget(self.password_input)

        # Server URL
        layout.add_widget(Label(
            text='Server',
            font_size=dp(14),
            color=TEXT_COLOR,
            size_hint_y=None,
            height=dp(24),
            halign='left',
        ))
        self.server_input = TextInput(
            text=DEFAULT_SERVER,
            multiline=False,
            font_size=dp(14),
            size_hint_y=None,
            height=dp(44),
        )
        layout.add_widget(self.server_input)

        # Spacer
        layout.add_widget(Label(size_hint_y=0.1))

        # Status
        self.status_label = Label(
            text='',
            font_size=dp(13),
            color=DIM_COLOR,
            size_hint_y=None,
            height=dp(30),
        )
        layout.add_widget(self.status_label)

        # Start button
        self.start_btn = Button(
            text='START',
            font_size=dp(20),
            bold=True,
            size_hint_y=None,
            height=dp(56),
            background_color=get_color_from_hex('#006688'),
        )
        self.start_btn.bind(on_press=self.on_start)
        layout.add_widget(self.start_btn)

        # Compute info
        compute_info = "scipy eigsh" if HAS_SCIPY else "numpy fallback (dense)"
        layout.add_widget(Label(
            text=f'Compute: {compute_info}',
            font_size=dp(11),
            color=DIM_COLOR,
            size_hint_y=None,
            height=dp(20),
        ))

        # Spacer at bottom
        layout.add_widget(Label(size_hint_y=0.2))

        self.add_widget(layout)

        # Load saved config
        cfg = load_config()
        if cfg.get('name'):
            self.name_input.text = cfg['name']
        if cfg.get('server'):
            self.server_input.text = cfg['server']

    def on_start(self, instance):
        name = self.name_input.text.strip()
        password = self.password_input.text.strip()
        server = self.server_input.text.strip()

        if not name:
            self.status_label.text = 'Please enter a name'
            self.status_label.color = RED
            return
        if len(password) < 4:
            self.status_label.text = 'Password must be at least 4 characters'
            self.status_label.color = RED
            return
        if not server:
            server = DEFAULT_SERVER

        self.start_btn.disabled = True
        self.status_label.text = 'Connecting...'
        self.status_label.color = ACCENT

        # Do registration/login in background thread
        threading.Thread(
            target=self._do_connect,
            args=(name, password, server),
            daemon=True,
        ).start()

    def _do_connect(self, name, password, server):
        """Register or login in background thread."""
        try:
            device_name = platform.node() or "android"
            gpu_info = "scipy eigsh" if HAS_SCIPY else "numpy dense"

            # Try register first
            code, data = http_post_json(
                f"{server}/register",
                json_data={
                    "name": name,
                    "gpu_info": gpu_info,
                    "password": password,
                    "device_name": device_name,
                },
                timeout=15,
            )

            if code == 409:
                # Name taken, try login
                Clock.schedule_once(lambda dt: self._set_status('Name taken, logging in...', ACCENT))
                code, data = http_post_json(
                    f"{server}/login",
                    json_data={
                        "name": name,
                        "password": password,
                        "device_name": device_name,
                        "gpu_info": gpu_info,
                    },
                    timeout=15,
                )
                if code != 200:
                    detail = data.get('detail', str(data))
                    Clock.schedule_once(lambda dt: self._set_status(f'Login failed: {detail}', RED))
                    Clock.schedule_once(lambda dt: self._enable_btn())
                    return

            elif code != 200:
                detail = data.get('detail', str(data)) if isinstance(data, dict) else str(data)
                Clock.schedule_once(lambda dt: self._set_status(f'Error: {detail}', RED))
                Clock.schedule_once(lambda dt: self._enable_btn())
                return

            # Save config
            api_key = data.get('api_key', '')
            worker_id = data.get('worker_id', '')
            cfg = {
                'api_key': api_key,
                'worker_id': worker_id,
                'name': name,
                'server': server,
            }
            save_config(cfg)

            # Switch to compute screen
            Clock.schedule_once(lambda dt: self._go_compute(api_key, worker_id, name, server))

        except Exception as e:
            Clock.schedule_once(lambda dt: self._set_status(f'Connection error: {e}', RED))
            Clock.schedule_once(lambda dt: self._enable_btn())

    def _set_status(self, text, color):
        self.status_label.text = text
        self.status_label.color = color

    def _enable_btn(self):
        self.start_btn.disabled = False

    def _go_compute(self, api_key, worker_id, name, server):
        app = App.get_running_app()
        compute_screen = app.root.get_screen('compute')
        compute_screen.start_work(api_key, worker_id, name, server)
        app.root.current = 'compute'


class ComputeScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        layout = BoxLayout(orientation='vertical', padding=dp(12), spacing=dp(8))

        # Header
        header = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(8))
        header.add_widget(Label(
            text='W@HOME',
            font_size=dp(18),
            bold=True,
            color=ACCENT,
            size_hint_x=0.6,
        ))
        self.stop_btn = Button(
            text='STOP',
            font_size=dp(14),
            size_hint_x=0.4,
            background_color=get_color_from_hex('#882222'),
        )
        self.stop_btn.bind(on_press=self.on_stop)
        header.add_widget(self.stop_btn)
        layout.add_widget(header)

        # Worker info line
        self.worker_label = Label(
            text='',
            font_size=dp(12),
            color=DIM_COLOR,
            size_hint_y=None,
            height=dp(20),
        )
        layout.add_widget(self.worker_label)

        # Stats line
        self.stats_label = Label(
            text='',
            font_size=dp(12),
            color=DIM_COLOR,
            size_hint_y=None,
            height=dp(20),
        )
        layout.add_widget(self.stats_label)

        # Current job
        self.job_label = Label(
            text='Waiting for job...',
            font_size=dp(16),
            bold=True,
            color=TEXT_COLOR,
            size_hint_y=None,
            height=dp(30),
        )
        layout.add_widget(self.job_label)

        # Stage indicator
        self.stage_label = Label(
            text='',
            font_size=dp(14),
            color=ACCENT,
            size_hint_y=None,
            height=dp(24),
        )
        layout.add_widget(self.stage_label)

        # Scrollable log
        scroll = ScrollView(size_hint_y=1)
        self.log_label = Label(
            text='',
            font_size=dp(12),
            color=TEXT_COLOR,
            markup=True,
            size_hint_y=None,
            text_size=(Window.width - dp(30), None),
            halign='left',
            valign='top',
        )
        self.log_label.bind(texture_size=self.log_label.setter('size'))
        scroll.add_widget(self.log_label)
        layout.add_widget(scroll)

        # Result line
        self.result_label = Label(
            text='',
            font_size=dp(13),
            color=GREEN,
            size_hint_y=None,
            height=dp(30),
            markup=True,
        )
        layout.add_widget(self.result_label)

        self.add_widget(layout)

        # State
        self.running = False
        self.api_key = None
        self.server = None
        self.jobs_done = 0
        self.discoveries = 0
        self.compute_hours = 0.0
        self._log_lines = []

    def start_work(self, api_key, worker_id, name, server):
        self.api_key = api_key
        self.server = server
        self.running = True
        compute_info = "scipy eigsh" if HAS_SCIPY else "numpy dense"
        self.worker_label.text = f'{name} ({worker_id[:8]}) | {compute_info}'
        self._log('Worker started')

        # Start heartbeat
        threading.Thread(target=self._heartbeat_loop, daemon=True).start()
        # Start compute loop
        threading.Thread(target=self._compute_loop, daemon=True).start()

    def on_stop(self, instance):
        self.running = False
        self._log('Stopping after current job...')
        self.stop_btn.disabled = True

    def _log(self, msg):
        self._log_lines.append(msg)
        if len(self._log_lines) > 100:
            self._log_lines = self._log_lines[-80:]
        text = '\n'.join(self._log_lines)
        Clock.schedule_once(lambda dt: self._set_log(text))

    def _set_log(self, text):
        self.log_label.text = text

    def _set_job(self, text):
        self.job_label.text = text

    def _set_stage(self, text):
        self.stage_label.text = text

    def _set_result(self, text):
        self.result_label.text = text

    def _set_stats(self):
        text = f'Jobs: {self.jobs_done}  Discoveries: {self.discoveries}  Compute: {self.compute_hours:.2f}h'
        Clock.schedule_once(lambda dt: setattr(self.stats_label, 'text', text))

    def _heartbeat_loop(self):
        while self.running:
            try:
                http_post(
                    f"{self.server}/heartbeat",
                    headers={"x-api-key": self.api_key},
                    timeout=10,
                )
            except Exception:
                pass
            time.sleep(120)

    def _compute_loop(self):
        backoff = MIN_BACKOFF

        while self.running:
            try:
                # Pull job
                Clock.schedule_once(lambda dt: self._set_job('Requesting job...'))
                code, data = http_post_json(
                    f"{self.server}/job",
                    headers={"x-api-key": self.api_key, "x-device-type": "mobile"},
                    timeout=30,
                )

                if code == 401:
                    self._log('API key rejected. Please re-register.')
                    break

                if code != 200:
                    self._log(f'Server error: {code}')
                    time.sleep(backoff)
                    backoff = min(backoff * 2, MAX_BACKOFF)
                    continue

                if data.get('status') == 'no_jobs':
                    Clock.schedule_once(lambda dt: self._set_job('Waiting for jobs...'))
                    self._log('All jobs assigned. Waiting 60s...')
                    time.sleep(60)
                    continue

                backoff = MIN_BACKOFF

                job_id = data['job_id']
                params = data['params']
                params['job_id'] = job_id
                progress = data.get('progress', {})

                lam = params.get('lambda', 0)
                k = params.get('k', '?')
                Clock.schedule_once(
                    lambda dt, jid=job_id, l=lam, kk=k:
                    self._set_job(f'Job {jid} | lambda={l:.6f} k={kk}')
                )

                pct = progress.get('percent_complete', 0)
                self._log(f'Job {job_id}: lambda={lam:.6f}, k={k}, hive {pct:.1f}% complete')

                # Run computation
                start_time = time.time()
                eigs = self._run_job(params)
                duration = time.time() - start_time

                # Check for constants
                hits = check_for_gold(eigs)
                n_eigs = len(eigs[eigs > 1e-9]) if isinstance(eigs, np.ndarray) else 0

                self.jobs_done += 1
                self.compute_hours += duration / 3600

                if hits:
                    self.discoveries += len(hits)
                    hit_str = ', '.join(hits[:3])
                    Clock.schedule_once(
                        lambda dt, hs=hit_str:
                        self._set_result(f'[color=ffd700]RESONANCE: {hs}[/color]')
                    )
                    self._log(f'RESONANCE DETECTED: {hits}')
                else:
                    Clock.schedule_once(
                        lambda dt, ne=n_eigs, d=duration:
                        self._set_result(f'{ne} eigenvalues in {d:.1f}s — no constants')
                    )

                self._set_stats()

                # Submit result
                eig_hash = hash_eigenvalues(eigs.tolist())
                code2, resp2 = http_post_json(
                    f"{self.server}/result",
                    headers={"x-api-key": self.api_key},
                    json_data={
                        "job_id": job_id,
                        "eigenvalues": eigs.tolist(),
                        "eigenvalues_hash": eig_hash,
                        "found_constants": hits,
                        "compute_seconds": duration,
                    },
                    timeout=30,
                )

                if code2 == 200:
                    if resp2.get('verified'):
                        self._log(f'Job {job_id}: result verified by quorum!')
                    else:
                        self._log(f'Job {job_id}: submitted ({duration:.1f}s)')
                else:
                    self._log(f'Job {job_id}: submit failed ({code2})')

                clear_checkpoint()

            except Exception as e:
                self._log(f'Error: {e}')
                time.sleep(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF)

        Clock.schedule_once(lambda dt: self._set_job('Stopped'))
        self._log('Worker stopped.')

    def _run_job(self, params):
        """Run eigenvalue computation with stage updates."""
        k = params['k']
        G1, G2 = params['G1'], params['G2']
        S = params['S']
        lam = params['lambda']
        w_glue = params['w_glue']

        Clock.schedule_once(lambda dt: self._set_stage('1/5 Building graph...'))
        vertices, edges, b_ids = w_operator.build_graph(k, G1, G2, S, 2)
        self._log(f'  Graph: {len(vertices)} vertices, {len(edges)} edges')

        Clock.schedule_once(lambda dt: self._set_stage('2/5 Adding glue edges...'))
        upd, psi_by = w_operator.add_glue_edges(vertices, b_ids, lam, w_glue, G1, G2)

        Clock.schedule_once(lambda dt: self._set_stage('3/5 Merging edges...'))
        edges_merged = w_operator.merge_edges(edges, upd)

        Clock.schedule_once(lambda dt: self._set_stage('4/5 Building Laplacian...'))
        L, _ = w_operator.build_magnetic_laplacian(vertices, edges_merged, s=(0, 0), psi_by_id=psi_by)

        Clock.schedule_once(lambda dt: self._set_stage('5/5 Solving spectrum...'))
        if HAS_SCIPY:
            eigs = w_operator.solve_spectrum(L, M=40)
        else:
            eigs = solve_spectrum_fallback(L, M=40)

        Clock.schedule_once(lambda dt: self._set_stage('Complete'))
        return eigs


# ═══════════════════════════════════════════════════════════
# App
# ═══════════════════════════════════════════════════════════

class WAtHomeApp(App):
    title = 'W@Home Hive'

    def build(self):
        Window.clearcolor = BG_COLOR
        sm = ScreenManager()
        sm.add_widget(LoginScreen(name='login'))
        sm.add_widget(ComputeScreen(name='compute'))

        # Auto-login if we have a saved key
        cfg = load_config()
        if cfg.get('api_key'):
            compute = sm.get_screen('compute')
            compute.start_work(
                cfg['api_key'],
                cfg.get('worker_id', ''),
                cfg.get('name', 'android'),
                cfg.get('server', DEFAULT_SERVER),
            )
            sm.current = 'compute'

        return sm

    def on_pause(self):
        # Android: allow background
        return True

    def on_resume(self):
        pass


if __name__ == '__main__':
    WAtHomeApp().run()
