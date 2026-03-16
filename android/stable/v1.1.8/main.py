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
    import ssl
    _ssl_ctx = ssl.create_default_context()
    _ssl_ctx.check_hostname = False
    _ssl_ctx.verify_mode = ssl.CERT_NONE

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
            with urllib.request.urlopen(req, timeout=timeout, context=_ssl_ctx) as resp:
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

def http_get_json(url, timeout=15, headers=None):
    """GET and parse JSON response."""
    if HAS_REQUESTS:
        resp = requests.get(url, timeout=timeout, headers=headers)
        return resp.status_code, resp.json()
    else:
        req = urllib.request.Request(url, method='GET')
        if headers:
            for k, v in headers.items():
                req.add_header(k, v)
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=_ssl_ctx) as resp:
                body = resp.read().decode('utf-8')
                return resp.status, json.loads(body)
        except urllib.error.HTTPError as e:
            return e.code, {}
        except Exception:
            return 0, {}

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

VERSION = "1.0.1"
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
    rounded = [round(float(e), 10) for e in sorted(eigs)]
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


from kivy.uix.widget import Widget
from kivy.uix.switch import Switch
from kivy.graphics import Color, Rectangle, RoundedRectangle, Mesh, Line
from kivy.animation import Animation
import math


# ═══════════════════════════════════════════════════════════
# Settings Screen
# ═══════════════════════════════════════════════════════════

class SettingsScreen(Screen):
    """User preferences — charge-only, max jobs, cooldown, alerts, logout."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        root = BoxLayout(orientation='vertical', padding=(dp(12), dp(8)), spacing=dp(4))

        # Header — pinned at top, never scrolls
        header = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(8))
        back_btn = Button(
            text='< Back',
            font_size=dp(14),
            size_hint_x=0.3,
            background_color=get_color_from_hex('#446688'),
        )
        back_btn.bind(on_press=self.go_back)
        header.add_widget(back_btn)
        header.add_widget(Label(
            text='SETTINGS',
            font_size=dp(18),
            bold=True,
            color=ACCENT,
            size_hint_x=0.7,
        ))
        root.add_widget(header)

        # Scrollable content
        scroll = ScrollView(size_hint_y=1)
        layout = BoxLayout(
            orientation='vertical', spacing=dp(8), padding=(dp(8), dp(4)),
            size_hint_y=None,
        )
        layout.bind(minimum_height=layout.setter('height'))

        # Account info
        self.account_label = Label(
            text='',
            font_size=dp(13),
            color=DIM_COLOR,
            size_hint_y=None,
            height=dp(28),
        )
        layout.add_widget(self.account_label)

        # --- Toggle: Charge only ---
        row = BoxLayout(size_hint_y=None, height=dp(50), spacing=dp(8))
        row.add_widget(Label(
            text='Only compute while charging',
            font_size=dp(14),
            color=TEXT_COLOR,
            size_hint_x=0.7,
            halign='left',
            text_size=(None, None),
        ))
        self.charge_only = Switch(active=False, size_hint_x=0.3)
        row.add_widget(self.charge_only)
        layout.add_widget(row)

        # --- Max jobs ---
        layout.add_widget(Label(
            text='Max jobs per session (0 = unlimited)',
            font_size=dp(13),
            color=DIM_COLOR,
            size_hint_y=None,
            height=dp(22),
            halign='left',
        ))
        row = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(8))
        row.add_widget(Label(
            text='Max jobs',
            font_size=dp(14),
            color=TEXT_COLOR,
            size_hint_x=0.6,
            halign='left',
        ))
        self.max_jobs = TextInput(
            text='0',
            input_filter='int',
            multiline=False,
            font_size=dp(16),
            size_hint_x=0.4,
            size_hint_y=None,
            height=dp(40),
        )
        row.add_widget(self.max_jobs)
        layout.add_widget(row)

        # --- Cooldown ---
        layout.add_widget(Label(
            text='Pause between jobs in seconds (0 = none)',
            font_size=dp(13),
            color=DIM_COLOR,
            size_hint_y=None,
            height=dp(22),
            halign='left',
        ))
        row = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(8))
        row.add_widget(Label(
            text='Cooldown (sec)',
            font_size=dp(14),
            color=TEXT_COLOR,
            size_hint_x=0.6,
            halign='left',
        ))
        self.cooldown = TextInput(
            text='0',
            input_filter='int',
            multiline=False,
            font_size=dp(16),
            size_hint_x=0.4,
            size_hint_y=None,
            height=dp(40),
        )
        row.add_widget(self.cooldown)
        layout.add_widget(row)

        # --- Toggle: Alert on hits ---
        row = BoxLayout(size_hint_y=None, height=dp(50), spacing=dp(8))
        row.add_widget(Label(
            text='Vibrate on resonance hit',
            font_size=dp(14),
            color=TEXT_COLOR,
            size_hint_x=0.7,
            halign='left',
        ))
        self.notify_hits = Switch(active=True, size_hint_x=0.3)
        row.add_widget(self.notify_hits)
        layout.add_widget(row)

        # Server alert banner (hidden until alert arrives)
        self.alert_bar = BoxLayout(
            size_hint_y=None, height=dp(0), spacing=dp(4),
            padding=(dp(8), dp(4)),
        )
        self.alert_label = Label(
            text='', font_size=dp(12), color=GOLD,
            size_hint_x=0.85, halign='left',
            text_size=(Window.width - dp(80), None),
        )
        self.alert_bar.add_widget(self.alert_label)
        dismiss_btn = Button(
            text='X', font_size=dp(12), size_hint_x=0.15,
            background_color=get_color_from_hex('#664422'),
        )
        dismiss_btn.bind(on_press=self.dismiss_alert)
        self.alert_bar.add_widget(dismiss_btn)
        layout.add_widget(self.alert_bar)

        # Save button
        save_btn = Button(
            text='SAVE',
            font_size=dp(16),
            bold=True,
            size_hint_y=None,
            height=dp(48),
            background_color=get_color_from_hex('#006688'),
        )
        save_btn.bind(on_press=self.save_settings)
        layout.add_widget(save_btn)

        # About / Help / Update / Logout row
        btn_row = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(6))

        about_btn = Button(
            text='About',
            font_size=dp(12),
            size_hint_x=0.22,
            background_color=get_color_from_hex('#333355'),
        )
        about_btn.bind(on_press=self.show_about)
        btn_row.add_widget(about_btn)

        help_btn = Button(
            text='Help',
            font_size=dp(12),
            size_hint_x=0.22,
            background_color=get_color_from_hex('#333355'),
        )
        help_btn.bind(on_press=self.show_help)
        btn_row.add_widget(help_btn)

        update_btn = Button(
            text='Update',
            font_size=dp(12),
            size_hint_x=0.28,
            background_color=get_color_from_hex('#226644'),
        )
        update_btn.bind(on_press=self.open_download)
        btn_row.add_widget(update_btn)

        logout_btn = Button(
            text='Logout',
            font_size=dp(12),
            size_hint_x=0.28,
            background_color=get_color_from_hex('#882222'),
        )
        logout_btn.bind(on_press=self.logout)
        btn_row.add_widget(logout_btn)
        layout.add_widget(btn_row)

        # Info label (for about/help messages)
        self.info_label = Label(
            text=f'W@Home v{VERSION} — Akataleptos Spectral Search',
            font_size=dp(11),
            color=DIM_COLOR,
            size_hint_y=None,
            height=dp(80),
            text_size=(Window.width - dp(40), None),
            halign='center',
            valign='top',
        )
        layout.add_widget(self.info_label)

        scroll.add_widget(layout)
        root.add_widget(scroll)
        self.add_widget(root)

    def on_pre_enter(self):
        cfg = load_config()
        self.charge_only.active = cfg.get('charge_only', False)
        self.max_jobs.text = str(cfg.get('max_jobs', 0))
        self.cooldown.text = str(cfg.get('cooldown', 0))
        self.notify_hits.active = cfg.get('notify_hits', True)
        name = cfg.get('name', '?')
        wid = cfg.get('worker_id', '')[:8]
        self.account_label.text = f'Logged in as: {name} ({wid})'
        self.info_label.text = f'W@Home v{VERSION} \u2014 Akataleptos Spectral Search'
        # Check for alerts + updates on settings open
        threading.Thread(target=self._check_server, daemon=True).start()

    def save_settings(self, instance):
        cfg = load_config()
        cfg['charge_only'] = self.charge_only.active
        cfg['max_jobs'] = int(self.max_jobs.text or 0)
        cfg['cooldown'] = int(self.cooldown.text or 0)
        cfg['notify_hits'] = self.notify_hits.active
        save_config(cfg)
        self.info_label.text = 'Settings saved!'
        self.info_label.color = GREEN

    def go_back(self, instance):
        App.get_running_app().root.current = 'compute'

    def show_about(self, instance):
        self.info_label.color = DIM_COLOR
        self.info_label.text = (
            f'W@Home Hive v{VERSION}\n'
            'Distributed Menger spectral search\n\n'
            'Publisher: Akataleptos\n'
            'Contact: obi@akataleptos.com\n'
            'Website: wathome.akataleptos.com'
        )

    def show_help(self, instance):
        self.info_label.color = DIM_COLOR
        self.info_label.text = (
            'Your device searches for universal constants\n'
            'hidden in the Menger sponge spectrum.\n\n'
            'Contact: obi@akataleptos.com\n'
            'Website: wathome.akataleptos.com'
        )

    def _check_server(self):
        """Check for alerts and updates from server."""
        try:
            cfg = load_config()
            server = cfg.get('server', DEFAULT_SERVER)
            code, data = http_get_json(f"{server}/version")
            if code == 200:
                # Server alert
                alert = data.get('alert', '')
                if alert:
                    Clock.schedule_once(lambda dt: self.show_alert(alert))
                # Version check
                sv = data.get('exe_version', '')
                if sv and sv != VERSION:
                    Clock.schedule_once(
                        lambda dt, v=sv: self._show_update(v))
        except Exception:
            pass

    def show_alert(self, message):
        """Show server alert in the alert banner."""
        self.alert_label.text = message
        self.alert_bar.height = dp(44)

    def dismiss_alert(self, instance):
        """Dismiss the alert banner."""
        self.alert_label.text = ''
        self.alert_bar.height = dp(0)

    def _show_update(self, version):
        """Show update available in info label."""
        self.info_label.color = GOLD
        self.info_label.text = (
            f'Update available: v{version}\n'
            f'You have v{VERSION}. Tap Update to download.'
        )

    def open_download(self, instance):
        url = 'https://wathome.akataleptos.com/static/wathome-latest.apk'
        try:
            from jnius import autoclass, cast
            Intent = autoclass('android.content.Intent')
            Uri = autoclass('android.net.Uri')
            from android import mActivity
            intent = Intent(Intent.ACTION_VIEW, Uri.parse(url))
            mActivity.startActivity(intent)
        except Exception:
            import webbrowser
            webbrowser.open(url)

    def logout(self, instance):
        # Stop the service first
        try:
            compute = App.get_running_app().root.get_screen('compute')
            compute.on_stop()
        except Exception:
            pass
        # Clear credentials
        save_config({})
        App.get_running_app().root.current = 'login'


# ═══════════════════════════════════════════════════════════
# Chat Screen — Hive worker chat
# ═══════════════════════════════════════════════════════════

def _http_get(url, timeout=10):
    """Module-level GET helper returning (code, parsed_json)."""
    try:
        if HAS_REQUESTS:
            resp = requests.get(url, timeout=timeout)
            return resp.status_code, resp.json()
        else:
            req = urllib.request.Request(url, method='GET')
            with urllib.request.urlopen(req, timeout=timeout, context=_ssl_ctx) as resp:
                body = resp.read().decode('utf-8')
                return resp.status, json.loads(body)
    except Exception:
        return 0, {}


class ChatScreen(Screen):
    """Live chat between hive workers."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        root = BoxLayout(orientation='vertical', padding=(dp(8), dp(4)), spacing=dp(4))

        # Header
        header = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(8))
        back_btn = Button(
            text='< Back',
            font_size=dp(14),
            size_hint_x=0.25,
            background_color=get_color_from_hex('#446688'),
        )
        back_btn.bind(on_press=lambda x: setattr(App.get_running_app().root, 'current', 'compute'))
        header.add_widget(back_btn)
        header.add_widget(Label(
            text='HIVE CHAT',
            font_size=dp(18),
            bold=True,
            color=ACCENT,
            size_hint_x=0.5,
        ))
        self.online_label = Label(
            text='',
            font_size=dp(11),
            color=DIM_COLOR,
            size_hint_x=0.25,
        )
        header.add_widget(self.online_label)
        root.add_widget(header)

        # Chat messages area (scrollable)
        self.chat_scroll = ScrollView(size_hint_y=1)
        self.chat_box = BoxLayout(
            orientation='vertical',
            size_hint_y=None,
            spacing=dp(2),
            padding=(dp(4), dp(4)),
        )
        self.chat_box.bind(minimum_height=self.chat_box.setter('height'))
        self.chat_scroll.add_widget(self.chat_box)
        root.add_widget(self.chat_scroll)

        # Input row
        input_row = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(6))
        self.msg_input = TextInput(
            hint_text='Type a message...',
            multiline=False,
            font_size=dp(14),
            size_hint_x=0.75,
            size_hint_y=None,
            height=dp(40),
        )
        self.msg_input.bind(on_text_validate=self._send_message)
        input_row.add_widget(self.msg_input)
        send_btn = Button(
            text='Send',
            font_size=dp(14),
            size_hint_x=0.25,
            background_color=get_color_from_hex('#006688'),
        )
        send_btn.bind(on_press=self._send_message)
        input_row.add_widget(send_btn)
        root.add_widget(input_row)

        self.add_widget(root)
        self._poll = None
        self._last_time = 0
        self._loaded = False

    def on_enter(self):
        self._load_history()
        self._poll = Clock.schedule_interval(self._poll_new, 3.0)

    def on_leave(self):
        if self._poll:
            self._poll.cancel()
            self._poll = None

    def _load_history(self):
        """Fetch chat history and populate."""
        cfg = load_config()
        server = cfg.get('server', 'https://wathome.akataleptos.com')
        def fetch():
            code, data = _http_get(f'{server}/chat/history?limit=50')
            if code == 200 and isinstance(data, list):
                Clock.schedule_once(lambda dt: self._populate(data))
        threading.Thread(target=fetch, daemon=True).start()

    def _populate(self, messages):
        self.chat_box.clear_widgets()
        self._last_time = 0
        for msg in messages:
            self._add_message(msg)
        self._loaded = True
        self._scroll_bottom()

    def _add_message(self, msg):
        username = msg.get('username', '?')
        content = msg.get('content', '')
        t = msg.get('time', 0)
        if t > self._last_time:
            self._last_time = t

        # Format time
        try:
            from datetime import datetime
            dt = datetime.fromtimestamp(t)
            ts = dt.strftime('%H:%M')
        except Exception:
            ts = ''

        cfg = load_config()
        is_me = (username == cfg.get('name', ''))

        line = BoxLayout(size_hint_y=None, height=dp(28), spacing=dp(4))

        name_color = ACCENT if is_me else get_color_from_hex('#ffcf6b')
        line.add_widget(Label(
            text=f'[{ts}] {username}:',
            font_size=dp(12),
            color=name_color,
            size_hint_x=0.35,
            halign='right',
            valign='middle',
            text_size=(None, None),
        ))
        line.add_widget(Label(
            text=content,
            font_size=dp(13),
            color=TEXT_COLOR,
            size_hint_x=0.65,
            halign='left',
            valign='middle',
            text_size=(None, None),
        ))
        self.chat_box.add_widget(line)

    def _scroll_bottom(self):
        def do_scroll(dt):
            self.chat_scroll.scroll_y = 0
        Clock.schedule_once(do_scroll, 0.1)

    def _poll_new(self, dt):
        """Poll for new messages since last seen."""
        if not self._loaded:
            return
        cfg = load_config()
        server = cfg.get('server', 'https://wathome.akataleptos.com')
        def fetch():
            code, data = _http_get(f'{server}/chat/history?limit=20')
            if code == 200 and isinstance(data, list):
                new_msgs = [m for m in data if m.get('time', 0) > self._last_time]
                if new_msgs:
                    Clock.schedule_once(lambda dt: self._append_new(new_msgs))
            # Also fetch online status
            code2, online = _http_get(f'{server}/chat/online')
            if code2 == 200:
                chat_users = online.get('chat', [])
                computing = online.get('computing', [])
                total = len(set(chat_users + computing))
                Clock.schedule_once(lambda dt: setattr(
                    self.online_label, 'text', f'{total} online'))
        threading.Thread(target=fetch, daemon=True).start()

    def _append_new(self, messages):
        for msg in messages:
            self._add_message(msg)
        self._scroll_bottom()

    def _send_message(self, instance=None):
        text = self.msg_input.text.strip()
        if not text:
            return
        self.msg_input.text = ''
        cfg = load_config()
        api_key = cfg.get('api_key', '')
        server = cfg.get('server', 'https://wathome.akataleptos.com')
        def send():
            http_post_json(
                f'{server}/chat/send',
                json_data={'content': text},
                headers={'x-api-key': api_key},
                timeout=10,
            )
            # Immediately poll for the new message
            Clock.schedule_once(lambda dt: self._poll_new(0), 0.5)
        threading.Thread(target=send, daemon=True).start()


# ═══════════════════════════════════════════════════════════
# Dashboard Screen — Live Hive Stats + Leaderboard
# ═══════════════════════════════════════════════════════════

class DashboardScreen(Screen):
    """Live hive dashboard — fetched from server API."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        layout = BoxLayout(orientation='vertical', padding=dp(12), spacing=dp(6))

        # Header
        header = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(8))
        back_btn = Button(
            text='< Back',
            font_size=dp(14),
            size_hint_x=0.3,
            background_color=get_color_from_hex('#446688'),
        )
        back_btn.bind(on_press=lambda x: setattr(App.get_running_app().root, 'current', 'compute'))
        header.add_widget(back_btn)
        header.add_widget(Label(
            text='HIVE DASHBOARD',
            font_size=dp(18),
            bold=True,
            color=ACCENT,
            size_hint_x=0.7,
        ))
        layout.add_widget(header)

        # Hive stats section
        self.progress_label = Label(
            text='Loading...',
            font_size=dp(13),
            color=TEXT_COLOR,
            size_hint_y=None,
            height=dp(110),
            text_size=(Window.width - dp(30), None),
            halign='left',
            valign='top',
            markup=True,
        )
        self.progress_label.bind(texture_size=self.progress_label.setter('size'))
        layout.add_widget(self.progress_label)

        # Separator
        layout.add_widget(Label(
            text='LEADERBOARD',
            font_size=dp(14),
            bold=True,
            color=ACCENT,
            size_hint_y=None,
            height=dp(28),
        ))

        # Leaderboard scroll
        scroll = ScrollView(size_hint_y=1)
        self.lb_label = Label(
            text='',
            font_size=dp(12),
            color=TEXT_COLOR,
            size_hint_y=None,
            text_size=(Window.width - dp(30), None),
            halign='left',
            valign='top',
            markup=True,
        )
        self.lb_label.bind(texture_size=self.lb_label.setter('size'))
        scroll.add_widget(self.lb_label)
        layout.add_widget(scroll)

        # Refresh button
        refresh_btn = Button(
            text='REFRESH',
            font_size=dp(14),
            size_hint_y=None,
            height=dp(40),
            background_color=get_color_from_hex('#006688'),
        )
        refresh_btn.bind(on_press=lambda x: self._fetch_data())
        layout.add_widget(refresh_btn)

        self.add_widget(layout)
        self._poll = None

    def on_enter(self):
        self._fetch_data()
        self._poll = Clock.schedule_interval(lambda dt: self._fetch_data(), 10.0)

    def on_leave(self):
        if self._poll:
            self._poll.cancel()
            self._poll = None

    def _fetch_data(self):
        threading.Thread(target=self._load_from_server, daemon=True).start()

    def _http_get_json(self, url, timeout=10):
        """GET request returning parsed JSON."""
        try:
            if HAS_REQUESTS:
                resp = requests.get(url, timeout=timeout)
                return resp.status_code, resp.json()
            else:
                import urllib.request
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=timeout, context=_ssl_ctx) as resp:
                    return resp.status, json.loads(resp.read().decode('utf-8'))
        except Exception:
            return 0, {}

    def _load_from_server(self):
        cfg = load_config()
        server = cfg.get('server', DEFAULT_SERVER)

        code, prog = self._http_get_json(f"{server}/progress")
        if code != 200:
            prog = {}

        code, lb = self._http_get_json(f"{server}/leaderboard")
        if code != 200:
            lb = []

        Clock.schedule_once(lambda dt: self._update_ui(prog, lb))

    def _update_ui(self, prog, lb):
        if prog:
            pct = prog.get('percent_complete', 0)
            total = prog.get('total_jobs', 0)
            done = prog.get('completed', 0)
            verified = prog.get('verified', 0)
            workers = prog.get('active_workers', 0)
            compute_h = prog.get('total_compute_hours', 0)
            discoveries = prog.get('total_discoveries', 0)
            confirmed = prog.get('total_confirmed', 0)
            jph = prog.get('jobs_per_hour', 0)
            eta = prog.get('eta_hours')
            lam = prog.get('current_lambda', 0)

            eta_str = f'{eta:.0f}h' if eta and eta < 100000 else '\u221e'

            self.progress_label.text = (
                f'[color=00d4ff]HIVE STATUS[/color]\n'
                f'Progress: [b]{pct:.2f}%[/b] ({done:,}/{total:,} jobs)\n'
                f'Verified: {verified:,}  |  Workers online: [b]{workers}[/b]\n'
                f'Compute: {compute_h:.1f}h  |  {jph} jobs/hr  |  ETA: {eta_str}\n'
                f'Discoveries: [color=ffd700]{discoveries}[/color] ({confirmed} confirmed)\n'
                f'Current \u03bb: {lam:.6f}'
            )
        else:
            self.progress_label.text = '[color=ff4444]Could not reach server[/color]'

        if lb:
            lines = []
            for entry in lb[:20]:
                rank = entry.get('rank', '?')
                name = entry.get('name', '?')
                jobs = entry.get('jobs', 0)
                hrs = entry.get('compute_hours', 0)
                disc = entry.get('discoveries', 0)
                online = entry.get('online', False)

                dot = '[color=00ff88]\u25cf[/color]' if online else '[color=666666]\u25cb[/color]'
                disc_str = f' [color=ffd700]\u2605{disc}[/color]' if disc > 0 else ''
                lines.append(f'{dot} {rank}. [b]{name}[/b] — {jobs} jobs, {hrs:.1f}h{disc_str}')

            self.lb_label.text = '\n'.join(lines)
        else:
            self.lb_label.text = 'No leaderboard data'


# ═══════════════════════════════════════════════════════════
# 3D Menger Sponge — Charging Screensaver
# ═══════════════════════════════════════════════════════════

class SpongeView(Widget):
    """3D rotating Menger sponge rendered via Kivy Canvas."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.depth = 2
        self.cubes = []
        self.cube_size = 0.32
        self.rot_x = -0.4
        self.rot_y = 0.6
        self.auto_rotate = True
        self._build_sponge()
        self.zoom = 0.32  # default zoom factor
        self._t = 0.0  # animation time
        self._last_touch = 0.0  # for auto-resume after idle
        self._anim = Clock.schedule_interval(self._animate, 1.0 / 30)
        self.bind(size=self._redraw, pos=self._redraw)
        # Touch: single drag = rotate, pinch = zoom
        self._touch_prev = None
        self._touches = {}  # uid -> pos, for pinch tracking

    def _is_menger(self, x, y, z):
        while x > 0 or y > 0 or z > 0:
            cnt = (1 if x % 3 == 1 else 0) + (1 if y % 3 == 1 else 0) + (1 if z % 3 == 1 else 0)
            if cnt >= 2:
                return False
            x //= 3
            y //= 3
            z //= 3
        return True

    def _build_sponge(self):
        cfgs = {1: (3, 1, 0.88), 2: (9, 3, 0.32), 3: (27, 9, 0.105)}
        N, div, self.cube_size = cfgs.get(self.depth, (9, 3, 0.32))
        half = (N - 1) / 2.0
        self.cubes = []
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    if self._is_menger(x, y, z):
                        self.cubes.append(((x - half) / div, (y - half) / div, (z - half) / div))

    def _rotate(self, x, y, z):
        cy, sy = math.cos(self.rot_y), math.sin(self.rot_y)
        x1 = x * cy - z * sy
        z1 = x * sy + z * cy
        cx, sx = math.cos(self.rot_x), math.sin(self.rot_x)
        y1 = y * cx - z1 * sx
        z2 = y * sx + z1 * cx
        return x1, y1, z2

    def _project(self, x, y, z):
        rx, ry, rz = self._rotate(x, y, z)
        scale = min(self.width, self.height) * self.zoom
        persp = 6.0
        f = persp / (persp + rz)
        px = rx * scale * f + self.center_x
        py = -ry * scale * f + self.center_y
        return px, py, rz

    def _pinch_dist(self):
        pts = list(self._touches.values())
        if len(pts) < 2:
            return 0
        dx = pts[0][0] - pts[1][0]
        dy = pts[0][1] - pts[1][1]
        return math.sqrt(dx * dx + dy * dy)

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            self._touches[touch.uid] = touch.pos
            if len(self._touches) == 1:
                self._touch_prev = touch.pos
                self.auto_rotate = False
                self._last_touch = self._t
            elif len(self._touches) == 2:
                self._pinch_start = self._pinch_dist()
                self._zoom_start = self.zoom
            touch.grab(self)
            return True
        return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        if touch.grab_current is not self:
            return super().on_touch_move(touch)
        self._touches[touch.uid] = touch.pos
        if len(self._touches) >= 2:
            # Pinch to zoom
            dist = self._pinch_dist()
            if hasattr(self, '_pinch_start') and self._pinch_start > 10:
                ratio = dist / self._pinch_start
                self.zoom = max(0.1, min(1.0, self._zoom_start * ratio))
            return True
        elif self._touch_prev:
            # Single finger drag = rotate
            dx = touch.pos[0] - self._touch_prev[0]
            dy = touch.pos[1] - self._touch_prev[1]
            self.rot_y += dx * 0.01
            self.rot_x -= dy * 0.01
            self.rot_x = max(-1.2, min(1.2, self.rot_x))
            self._touch_prev = touch.pos
            return True
        return True

    def on_touch_up(self, touch):
        if touch.grab_current is self:
            touch.ungrab(self)
            self._touches.pop(touch.uid, None)
            self._touch_prev = None
            self._last_touch = self._t
            return True
        return super().on_touch_up(touch)

    def _animate(self, dt):
        self._t += dt
        if not self.auto_rotate and (self._t - self._last_touch) > 4.0:
            self.auto_rotate = True  # resume after 4s idle
        if self.auto_rotate:
            self.rot_y += 0.012
            self.rot_x = -0.4 + 0.15 * math.sin(self._t * 0.3)  # gentle wobble
        self._redraw()

    def _redraw(self, *args):
        self.canvas.clear()
        if self.width < 10 or self.height < 10:
            return

        s = self.cube_size
        hs = s / 2.0

        # Project all cube corners + sort by center depth
        draw_list = []
        for cx, cy, cz in self.cubes:
            # 8 corners
            corners = []
            for dx in (-hs, hs):
                for dy in (-hs, hs):
                    for dz in (-hs, hs):
                        corners.append(self._project(cx + dx, cy + dy, cz + dz))
            # Center depth for sorting
            _, _, cdepth = self._rotate(cx, cy, cz)
            draw_list.append((cdepth, corners))

        draw_list.sort(key=lambda item: item[0])

        # Corner indices for 6 faces: top, front, right, left, back, bottom
        face_defs = [
            ([4, 5, 7, 6], 1.0),   # top  (y+)
            ([0, 1, 5, 4], 0.75),  # front (z-)
            ([1, 3, 7, 5], 0.85),  # right (x+)
            ([2, 0, 4, 6], 0.65),  # left  (x-)
            ([3, 2, 6, 7], 0.7),   # back  (z+)
            ([0, 2, 3, 1], 0.5),   # bottom (y-)
        ]

        with self.canvas:
            # Background
            Color(0.05, 0.05, 0.1, 1)
            Rectangle(pos=self.pos, size=self.size)

            for depth, corners in draw_list:
                # Brightness varies with depth
                z_norm = (depth + 1.5) / 3.0
                base_b = 0.35 + 0.65 * max(0, min(1, 1 - z_norm))

                for vidxs, shade in face_defs:
                    v = [corners[i] for i in vidxs]
                    # Backface culling
                    ax = v[1][0] - v[0][0]
                    ay = v[1][1] - v[0][1]
                    bx = v[2][0] - v[0][0]
                    by = v[2][1] - v[0][1]
                    if ax * by - ay * bx <= 0:
                        continue

                    r_c = 0
                    g_c = base_b * shade * 0.83
                    b_c = base_b * shade
                    Color(r_c, g_c, b_c, 1)

                    # Draw quad as triangle fan
                    verts = []
                    for vx, vy, _ in v:
                        verts.extend([vx, vy, 0, 0])
                    Mesh(vertices=verts, indices=[0, 1, 2, 0, 2, 3], mode='triangles')

    def stop_anim(self):
        if self._anim:
            self._anim.cancel()
            self._anim = None

    def start_anim(self):
        if not self._anim:
            self._anim = Clock.schedule_interval(self._animate, 1.0 / 30)
            self.auto_rotate = True


class SpongeScreen(Screen):
    """Fullscreen 3D Menger sponge — charging screensaver with live stats."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        layout = BoxLayout(orientation='vertical')

        # Header with back button + level control
        header = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(6), padding=(dp(8), 0))
        back_btn = Button(
            text='< Back',
            font_size=dp(14),
            size_hint_x=0.22,
            background_color=get_color_from_hex('#446688'),
        )
        back_btn.bind(on_press=self._go_back)
        header.add_widget(back_btn)
        header.add_widget(Label(
            text='3D SPONGE',
            font_size=dp(16),
            bold=True,
            color=ACCENT,
            size_hint_x=0.3,
        ))

        # Level buttons
        for lvl in (1, 2, 3):
            btn = Button(
                text=f'L{lvl}',
                font_size=dp(13),
                size_hint_x=0.12,
                background_color=get_color_from_hex('#006688' if lvl == 2 else '#333355'),
            )
            btn.bind(on_press=lambda inst, l=lvl: self._set_level(l))
            header.add_widget(btn)
            if lvl == 1:
                self._l1_btn = btn
            elif lvl == 2:
                self._l2_btn = btn
            else:
                self._l3_btn = btn

        header.add_widget(Label(
            text='drag to rotate',
            font_size=dp(10),
            color=DIM_COLOR,
            size_hint_x=0.22,
        ))
        layout.add_widget(header)

        # The 3D sponge takes most of the screen
        self.sponge = SpongeView(size_hint_y=1)
        layout.add_widget(self.sponge)

        # Stats overlay bar at bottom
        stats_bar = BoxLayout(
            size_hint_y=None, height=dp(48),
            orientation='vertical', padding=dp(8), spacing=dp(4),
        )

        self.stats_line = Label(
            text='',
            font_size=dp(14),
            color=ACCENT,
            size_hint_y=None,
            height=dp(22),
        )
        stats_bar.add_widget(self.stats_line)

        self.job_line = Label(
            text='',
            font_size=dp(12),
            color=TEXT_COLOR,
            size_hint_y=None,
            height=dp(20),
        )
        stats_bar.add_widget(self.job_line)

        layout.add_widget(stats_bar)
        self.add_widget(layout)

        self._poll = None

    def on_enter(self):
        self.sponge.start_anim()
        self._poll = Clock.schedule_interval(self._update_stats, 2.0)

    def on_leave(self):
        self.sponge.stop_anim()
        if self._poll:
            self._poll.cancel()
            self._poll = None

    def _update_stats(self, dt):
        try:
            compute = App.get_running_app().root.get_screen('compute')
            jobs = compute.jobs_done
            hits = compute.discoveries
            hours = compute.compute_hours
            self.stats_line.text = f'Jobs: {jobs}  |  Hits: {hits}  |  {hours:.2f}h compute'
            self.job_line.text = compute.job_label.text
        except Exception:
            pass

    def _set_level(self, level):
        """Change Menger sponge detail level."""
        self.sponge.depth = level
        self.sponge._build_sponge()
        self.sponge._redraw()
        # Highlight active button
        inactive = get_color_from_hex('#333355')
        active = get_color_from_hex('#006688')
        self._l1_btn.background_color = active if level == 1 else inactive
        self._l2_btn.background_color = active if level == 2 else inactive
        self._l3_btn.background_color = active if level == 3 else inactive

    def _go_back(self, instance):
        App.get_running_app().root.current = 'compute'


class MengerWidget(Widget):
    """2D Menger carpet that fills in cells as jobs complete."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.level = 2  # L2 = 64 cells
        self.jobs_done = 0
        self._cells = []
        self._build_cells()
        self.bind(size=self._rebuild, pos=self._rebuild)

    def _get_carpet_cells(self, level, x, y, size):
        """Recursively generate Menger carpet cell positions."""
        if level == 0:
            return [(x, y, size)]
        cells = []
        sub = size / 3.0
        for row in range(3):
            for col in range(3):
                if row == 1 and col == 1:
                    continue  # The hole
                cells.extend(self._get_carpet_cells(
                    level - 1,
                    x + col * sub,
                    y + row * sub,
                    sub
                ))
        return cells

    def _build_cells(self):
        # 8^level cells at each level
        side = min(self.width, self.height) if self.width > 0 else 200
        ox = self.x + (self.width - side) / 2
        oy = self.y + (self.height - side) / 2
        self._cells = self._get_carpet_cells(self.level, ox, oy, side)

    def _rebuild(self, *args):
        self._build_cells()
        self.draw(self.jobs_done)

    def draw(self, jobs_done):
        self.jobs_done = jobs_done
        self.canvas.clear()

        if not self._cells:
            self._build_cells()

        total = len(self._cells)
        # Cycle through levels: after filling all cells, wrap around
        # Each full cycle = one pass through all cells
        cycle = jobs_done % total if total > 0 else 0
        full_passes = jobs_done // total if total > 0 else 0

        with self.canvas:
            # Background hole (center of each level)
            Color(0.06, 0.06, 0.12, 1)
            Rectangle(pos=self.pos, size=self.size)

            for i in range(total):
                x, y, s = self._cells[i]
                gap = max(1, s * 0.08)

                if i < cycle:
                    # Completed in current cycle — cyan, hue shifts with passes
                    hue_shift = (full_passes * 0.15) % 1.0
                    r = 0.0 + hue_shift * 0.3
                    g = 0.55 + hue_shift * 0.2
                    b = 0.7 - hue_shift * 0.1
                    if i >= cycle - 3:
                        # Recent — brighter
                        Color(r * 1.4, min(1, g * 1.5), min(1, b * 1.3), 1)
                    else:
                        Color(r, g, b, 0.85)
                elif i == cycle:
                    # Current computing cell — gold pulse
                    Color(1, 0.84, 0, 0.9)
                else:
                    # Pending — dim, but tinted if completed in previous passes
                    if full_passes > 0:
                        Color(0.08, 0.15, 0.22, 0.9)
                    else:
                        Color(0.12, 0.12, 0.22, 1)

                Rectangle(pos=(x + gap, y + gap), size=(s - gap * 2, s - gap * 2))

            # Pass counter overlay
            if full_passes > 0:
                Color(1, 0.84, 0, 0.25)
                for i in range(min(full_passes, total)):
                    x, y, s = self._cells[i]
                    Rectangle(pos=(x, y), size=(s, s))


def _get_status_path():
    """Get the service status file path."""
    try:
        from android.storage import app_storage_path
        return os.path.join(app_storage_path(), "service_status.json")
    except ImportError:
        return os.path.join(app_dir, "service_status.json")


def _get_stop_path():
    """Get the service stop signal path."""
    return _get_status_path().replace('service_status.json', 'service_stop')


def _is_android():
    try:
        from android import mActivity
        return True
    except ImportError:
        return False


def _is_charging():
    """Check if device is plugged in (Android only) using sticky broadcast."""
    try:
        from jnius import autoclass
        from android import mActivity
        Intent = autoclass('android.content.Intent')
        IntentFilter = autoclass('android.content.IntentFilter')
        BatteryManager = autoclass('android.os.BatteryManager')
        ifilter = IntentFilter(Intent.ACTION_BATTERY_CHANGED)
        battery = mActivity.registerReceiver(None, ifilter)
        if battery is None:
            return False
        # EXTRA_PLUGGED: 0=unplugged, 1=AC, 2=USB, 4=wireless
        plugged = battery.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0)
        return plugged > 0
    except Exception:
        return False


class ComputeScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        layout = BoxLayout(orientation='vertical', padding=dp(12), spacing=dp(8))

        # Header
        header = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(8))
        header.add_widget(Label(
            text='W@HOME',
            font_size=dp(15),
            bold=True,
            color=ACCENT,
            size_hint_x=0.22,
        ))
        dash_btn = Button(
            text='Hive',
            font_size=dp(11),
            size_hint_x=0.15,
            background_color=get_color_from_hex('#224466'),
        )
        dash_btn.bind(on_press=self._open_dashboard)
        header.add_widget(dash_btn)
        chat_btn = Button(
            text='Chat',
            font_size=dp(11),
            size_hint_x=0.15,
            background_color=get_color_from_hex('#335544'),
        )
        chat_btn.bind(on_press=self._open_chat)
        header.add_widget(chat_btn)
        sponge_btn = Button(
            text='3D',
            font_size=dp(11),
            size_hint_x=0.12,
            background_color=get_color_from_hex('#004466'),
        )
        sponge_btn.bind(on_press=self._open_sponge)
        header.add_widget(sponge_btn)
        settings_btn = Button(
            text='Set',
            font_size=dp(11),
            size_hint_x=0.12,
            background_color=get_color_from_hex('#333355'),
        )
        settings_btn.bind(on_press=self._open_settings)
        header.add_widget(settings_btn)
        self.stop_btn = Button(
            text='STOP',
            font_size=dp(13),
            size_hint_x=0.24,
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

        # Menger carpet visualization
        self.menger = MengerWidget(size_hint_y=1)
        layout.add_widget(self.menger)

        # Scrollable log (smaller)
        scroll = ScrollView(size_hint_y=0.3)
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
        self._status_poll = None
        self._service_mode = _is_android()
        self._last_stats_save = 0
        self._was_charging = False
        self._charge_check_counter = 0

    def _check_charging_screensaver(self):
        """Auto-switch to sponge screensaver when charger is plugged in."""
        self._charge_check_counter += 1
        if self._charge_check_counter < 15:  # Check every ~30s (poll=2s × 15)
            return
        self._charge_check_counter = 0

        charging = _is_charging()
        if charging and not self._was_charging:
            # Just plugged in — switch to sponge
            self._was_charging = True
            app = App.get_running_app()
            if app.root.current == 'compute':
                app.root.current = 'sponge'
        elif not charging:
            self._was_charging = False

    def _save_session_stats(self):
        """Persist session stats to config so they survive app restarts."""
        try:
            cfg = load_config()
            cfg['session_jobs'] = self.jobs_done
            cfg['session_discoveries'] = self.discoveries
            cfg['session_hours'] = round(self.compute_hours, 4)
            save_config(cfg)
        except Exception:
            pass

    def _restore_session_stats(self):
        """Load persisted session stats from config."""
        cfg = load_config()
        self.jobs_done = cfg.get('session_jobs', 0)
        self.discoveries = cfg.get('session_discoveries', 0)
        self.compute_hours = cfg.get('session_hours', 0.0)
        if self.jobs_done > 0:
            self.stats_label.text = f'Jobs: {self.jobs_done}  Hits: {self.discoveries}  Compute: {self.compute_hours:.2f}h'
            self.menger.draw(self.jobs_done)

    def start_work(self, api_key, worker_id, name, server):
        self.api_key = api_key
        self.server = server
        self.running = True
        compute_info = "scipy eigsh" if HAS_SCIPY else "numpy dense"
        self.worker_label.text = f'{name} | {compute_info}'

        # Restore stats from previous session
        self._restore_session_stats()

        if _is_android():
            # Foreground service — runs in separate process, exempt from Doze.
            # buildozer.spec declares foregroundServiceType=dataSync for Android 14+.
            self._service_mode = True
            self._log('Starting foreground service...')
            try:
                self._start_service()
                self._status_poll = Clock.schedule_interval(self._poll_service_status, 2.0)
                self._log('Foreground service started')
            except Exception as e:
                self._log(f'Service failed ({e}), falling back to in-process')
                self._service_mode = False
                self._acquire_wakelock()
                self._request_notification_permission()
                self._show_persistent_notification('Computing...')
                threading.Thread(target=self._heartbeat_loop, daemon=True).start()
                threading.Thread(target=self._compute_loop, daemon=True).start()
        else:
            self._service_mode = False
            self._log('Worker started')
            self._acquire_wakelock()
            self._request_notification_permission()
            self._show_persistent_notification('Computing...')
            threading.Thread(target=self._heartbeat_loop, daemon=True).start()
            threading.Thread(target=self._compute_loop, daemon=True).start()

    def _open_settings(self, instance):
        App.get_running_app().root.current = 'settings'

    def _open_dashboard(self, instance):
        App.get_running_app().root.current = 'dashboard'

    def _open_chat(self, instance):
        App.get_running_app().root.current = 'chat'

    def _open_sponge(self, instance):
        App.get_running_app().root.current = 'sponge'

    def _request_battery_exemption(self):
        """Request exemption from Doze battery optimization.
        Without this, Android blocks network access when screen is off."""
        try:
            from jnius import autoclass
            from android import mActivity
            Intent = autoclass('android.content.Intent')
            Settings = autoclass('android.provider.Settings')
            Uri = autoclass('android.net.Uri')
            PowerManager = autoclass('android.os.PowerManager')
            Context = autoclass('android.content.Context')

            pm = mActivity.getSystemService(Context.POWER_SERVICE)
            pkg = mActivity.getPackageName()
            if not pm.isIgnoringBatteryOptimizations(pkg):
                intent = Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS)
                intent.setData(Uri.parse(f"package:{pkg}"))
                mActivity.startActivity(intent)
                self._log('Battery exemption requested')
            else:
                self._log('Battery exemption already granted')
        except Exception as e:
            self._log(f'Battery exemption unavailable: {e}')

    def _request_notification_permission(self):
        """Request POST_NOTIFICATIONS permission on Android 13+."""
        try:
            from android.permissions import request_permissions, Permission
            request_permissions([Permission.POST_NOTIFICATIONS])
            self._log('Notification permission requested')
        except Exception as e:
            self._log(f'Permission request: {e}')

    def _show_persistent_notification(self, text='Computing...'):
        """Show a rich foreground-style notification from the main process."""
        try:
            from jnius import autoclass
            from android import mActivity

            Context = autoclass('android.content.Context')
            NotificationManager = autoclass('android.app.NotificationManager')
            NotificationChannel = autoclass('android.app.NotificationChannel')
            NotificationBuilder = autoclass('android.app.Notification$Builder')
            Notification = autoclass('android.app.Notification')
            BigTextStyle = autoclass('android.app.Notification$BigTextStyle')
            PendingIntent = autoclass('android.app.PendingIntent')
            Intent = autoclass('android.content.Intent')
            String = autoclass('java.lang.String')

            manager = mActivity.getSystemService(Context.NOTIFICATION_SERVICE)

            # Create channel (Android 8+)
            channel = NotificationChannel(
                String('wathome_compute'),
                String('W@Home Computing'),
                NotificationManager.IMPORTANCE_LOW
            )
            channel.setDescription(String('Eigenvalue computation'))
            manager.createNotificationChannel(channel)

            # Intent to reopen app on tap
            intent = Intent(mActivity, mActivity.getClass())
            intent.setFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP)
            pending = PendingIntent.getActivity(
                mActivity, 0, intent,
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE
            )

            expanded = f"{text}\nSession: 0 jobs \u2022 0 hits \u2022 0.0h compute"

            builder = NotificationBuilder(mActivity, String('wathome_compute'))
            builder.setContentTitle(String('W@Home Hive'))
            builder.setContentText(String(text))
            builder.setSmallIcon(mActivity.getApplicationInfo().icon)
            builder.setContentIntent(pending)
            builder.setOngoing(True)
            builder.setNumber(0)
            builder.setVisibility(Notification.VISIBILITY_PUBLIC)
            # Accent color — cyan (#00d4ff) as signed 32-bit
            builder.setColor(-16063233)
            builder.setColorized(True)

            big_style = BigTextStyle()
            big_style.bigText(String(expanded))
            big_style.setBigContentTitle(String('W@Home \u2022 0 jobs'))
            builder.setStyle(big_style)

            manager.notify(42, builder.build())
            self._notification_manager = manager
            self._notif_classes = {
                'builder': NotificationBuilder,
                'bigtext': BigTextStyle,
                'notification': Notification,
                'string': String,
                'pending': pending,
            }
            self._log('Notification shown')
        except Exception as e:
            self._log(f'Notification failed: {e}')

    def _update_notification(self, text):
        """Update the persistent notification with rich stats."""
        if hasattr(self, '_notification_manager') and self._notification_manager:
            try:
                c = getattr(self, '_notif_classes', None)
                if not c:
                    return
                from android import mActivity
                String = c['string']

                jobs = self.jobs_done
                hits = self.discoveries
                hours = self.compute_hours

                expanded = f"{text}\nSession: {jobs} jobs \u2022 {hits} hits \u2022 {hours:.1f}h compute"

                builder = c['builder'](mActivity, String('wathome_compute'))
                builder.setContentTitle(String(f'W@Home \u2022 {jobs} jobs'))
                builder.setContentText(String(text))
                builder.setSmallIcon(mActivity.getApplicationInfo().icon)
                builder.setContentIntent(c['pending'])
                builder.setOngoing(True)
                builder.setNumber(jobs)
                builder.setVisibility(c['notification'].VISIBILITY_PUBLIC)
                builder.setColor(-16063233)
                builder.setColorized(True)

                big_style = c['bigtext']()
                big_style.bigText(String(expanded))
                big_style.setBigContentTitle(String(f'W@Home \u2022 {jobs} jobs'))
                builder.setStyle(big_style)

                self._notification_manager.notify(42, builder.build())
            except Exception:
                pass

    def _update_rich_notification(self, hits=None):
        """Fetch server stats and update notification with full dashboard."""
        try:
            # Fetch progress + wallet in parallel-ish (sequential but fast)
            prog = {}
            wallet = {}
            try:
                pc, pdata = http_get_json(f"{self.server}/progress")
                if pc == 200:
                    prog = pdata
            except Exception:
                pass
            try:
                wc, wdata = http_get_json(
                    f"{self.server}/wallet",
                    headers={"x-api-key": self.api_key},
                )
                if wc == 200:
                    wallet = wdata
            except Exception:
                pass

            workers = prog.get('active_workers', '?')
            total_disc = prog.get('total_discoveries', 0)
            verified_jobs = prog.get('verified', 0)
            pct = prog.get('percent_complete', 0)
            hive_hours = prog.get('total_compute_hours', 0)

            w_bal = wallet.get('balance', 0)
            w_earned = wallet.get('total_earned', 0)
            username = wallet.get('name', '?')

            # W per hour (from session)
            if self.compute_hours > 0:
                w_per_hr = w_earned / max(self.compute_hours, 0.01)
            else:
                w_per_hr = 0

            # Short text for collapsed view
            if hits:
                short = f'RESONANCE! {self.jobs_done} jobs | {w_bal:.1f} W'
            else:
                short = f'{self.jobs_done} jobs | {w_bal:.1f} W | {workers} online'

            # Expanded text for shade pull-down
            lines = []
            lines.append(f'\u2022 {username} | {self.jobs_done} jobs | {self.discoveries} hits')
            lines.append(f'\u2022 Balance: {w_bal:.2f} W | Earned: {w_earned:.2f} W | {w_per_hr:.1f} W/hr')
            lines.append(f'\u2022 Verified: {verified_jobs} | Discoveries: {total_disc}')
            lines.append(f'\u2022 Hive: {workers} workers | {pct:.1f}% complete | {hive_hours:.1f}h compute')
            expanded = '\n'.join(lines)

            self._update_notification_full(short, expanded)
        except Exception:
            # Fallback to simple notification
            self._update_notification(f'Jobs: {self.jobs_done} | Hits: {self.discoveries}')

    def _update_notification_full(self, short_text, expanded_text):
        """Update notification with separate short and expanded text."""
        if not hasattr(self, '_notification_manager') or not self._notification_manager:
            return
        try:
            c = getattr(self, '_notif_classes', None)
            if not c:
                return
            from android import mActivity
            String = c['string']

            builder = c['builder'](mActivity, String('wathome_compute'))
            builder.setContentTitle(String(f'W@Home \u2022 {self.jobs_done} jobs'))
            builder.setContentText(String(short_text))
            builder.setSmallIcon(mActivity.getApplicationInfo().icon)
            builder.setContentIntent(c['pending'])
            builder.setOngoing(True)
            builder.setNumber(self.jobs_done)
            builder.setVisibility(c['notification'].VISIBILITY_PUBLIC)
            builder.setColor(-16063233)
            builder.setColorized(True)

            big_style = c['bigtext']()
            big_style.bigText(String(expanded_text))
            big_style.setBigContentTitle(String(f'W@Home \u2022 {self.jobs_done} jobs'))
            builder.setStyle(big_style)

            self._notification_manager.notify(42, builder.build())
        except Exception:
            pass

    def _start_service(self):
        """Start the Android foreground service."""
        from android import mActivity
        from jnius import autoclass
        service = autoclass('com.akataleptos.wathome.ServiceWorker')
        service.start(mActivity, '')
        self._log('Background service started')

    def _stop_service(self):
        """Signal the service to stop."""
        try:
            # Write stop signal file
            with open(_get_stop_path(), 'w') as f:
                f.write('stop')
            self._log('Stop signal sent to service')
        except Exception as e:
            self._log(f'Error stopping service: {e}')

        try:
            from android import mActivity
            from jnius import autoclass
            service = autoclass('com.akataleptos.wathome.ServiceWorker')
            service.stop(mActivity)
        except Exception:
            pass

    def _poll_service_status(self, dt):
        """Read status from the service's shared JSON file and update UI."""
        try:
            path = _get_status_path()
            if not os.path.exists(path):
                return
            with open(path) as f:
                status = json.load(f)

            state = status.get('state', '')
            jobs = status.get('jobs_done', 0)
            disc = status.get('discoveries', 0)
            hours = status.get('compute_hours', 0)

            self.jobs_done = jobs
            self.discoveries = disc
            self.compute_hours = hours

            self.stats_label.text = f'Jobs: {jobs}  Hits: {disc}  Compute: {hours:.2f}h'

            # Persist stats every ~30 seconds (poll runs every 2s)
            now = time.time()
            if now - self._last_stats_save > 30:
                self._last_stats_save = now
                self._save_session_stats()

            # Update Menger carpet
            if jobs != self.menger.jobs_done:
                self.menger.draw(jobs)

            if state == 'computing':
                job_id = status.get('job_id', '?')
                lam = status.get('lambda', 0)
                k = status.get('k', '?')
                self.job_label.text = f'Job {job_id} | \u03bb={lam:.6f} k={k}'
                self.stage_label.text = 'Computing eigenvalues...'
                self.stage_label.color = ACCENT
            elif state == 'requesting':
                self.job_label.text = 'Requesting job...'
                self.stage_label.text = 'Connecting to hive'
                self.stage_label.color = DIM_COLOR
            elif state == 'waiting':
                self.job_label.text = 'All jobs assigned'
                self.stage_label.text = 'Waiting for new jobs...'
                self.stage_label.color = DIM_COLOR
            elif state == 'submitted':
                job_id = status.get('job_id', '?')
                dur = status.get('duration', 0)
                hits = status.get('hits', 0)
                verified = status.get('verified', False)
                if hits:
                    self.result_label.text = f'[color=ffd700]HIT on job {job_id}![/color]'
                    self._log(f'RESONANCE on job {job_id}!')
                elif verified:
                    self.result_label.text = f'[color=00ff88]Job {job_id}: VERIFIED[/color]'
                    self._log(f'Job {job_id}: verified by quorum!')
                else:
                    self.result_label.text = f'Job {job_id}: done in {dur:.1f}s'
                self.stage_label.text = 'Submitting result...'
            elif state in ('init', 'starting'):
                msg = status.get('message', 'Initializing...')
                self.stage_label.text = msg
                self.stage_label.color = DIM_COLOR
            elif state == 'stopped':
                self.job_label.text = 'Stopped'
                self.stage_label.text = f'Completed {jobs} jobs'
                self.running = False
                if self._status_poll:
                    self._status_poll.cancel()
            elif state == 'error':
                msg = status.get('message', 'Unknown error')
                self.stage_label.text = f'Error: {msg[:60]}'
                self.stage_label.color = RED
                self._log(f'Service error: {msg}')

        except (json.JSONDecodeError, IOError):
            pass  # File being written, try next poll

        # Check for charger → auto-screensaver
        if self._service_mode:
            self._check_charging_screensaver()

    def on_stop(self, instance=None):
        if self.running:
            # STOP
            self.running = False
            self._save_session_stats()
            if self._service_mode:
                self._stop_service()
            else:
                self._release_wakelock()
            self._log('Stopped.')
            self.stop_btn.text = 'GO'
            self.stop_btn.background_color = get_color_from_hex('#228822')
            if self._status_poll:
                self._status_poll.cancel()
        elif self.api_key:
            # RESUME
            self.running = True
            self._log('Resuming...')
            self.stop_btn.text = 'STOP'
            self.stop_btn.background_color = get_color_from_hex('#882222')
            if _is_android():
                self._service_mode = True
                try:
                    self._start_service()
                    self._status_poll = Clock.schedule_interval(self._poll_service_status, 2.0)
                except Exception:
                    self._service_mode = False
                    self._acquire_wakelock()
                    self._show_persistent_notification('Computing...')
                    threading.Thread(target=self._heartbeat_loop, daemon=True).start()
                    threading.Thread(target=self._compute_loop, daemon=True).start()
            else:
                self._acquire_wakelock()
                self._show_persistent_notification('Computing...')
                threading.Thread(target=self._heartbeat_loop, daemon=True).start()
                threading.Thread(target=self._compute_loop, daemon=True).start()

    def _acquire_wakelock(self):
        """Keep CPU running when screen is off (in-process fallback only)."""
        try:
            from android import mActivity
            from jnius import autoclass
            Context = autoclass('android.content.Context')
            PowerManager = autoclass('android.os.PowerManager')
            pm = mActivity.getSystemService(Context.POWER_SERVICE)
            self._wakelock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, 'wathome:compute')
            self._wakelock.acquire()
            self._log('Wakelock acquired — computing with screen off')
        except Exception as e:
            self._wakelock = None
            self._log(f'Wakelock unavailable: {e}')

    def _release_wakelock(self):
        """Release the wakelock."""
        try:
            if hasattr(self, '_wakelock') and self._wakelock and self._wakelock.isHeld():
                self._wakelock.release()
                self._log('Wakelock released')
        except Exception:
            pass

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
        text = f'Jobs: {self.jobs_done}  Hits: {self.discoveries}  Compute: {self.compute_hours:.2f}h'
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
        """In-process compute loop (desktop/fallback only)."""
        backoff = MIN_BACKOFF

        while self.running:
            try:
                # --- Settings enforcement ---
                cfg = load_config()

                # Charge-only mode: pause until plugged in
                if cfg.get('charge_only', False) and not _is_charging():
                    Clock.schedule_once(lambda dt: self._set_job('Waiting for charger...'))
                    self._update_notification('Paused — waiting for charger')
                    time.sleep(10)
                    continue

                # Max jobs limit — sleep and re-check, don't exit thread
                max_jobs = cfg.get('max_jobs', 0)
                if max_jobs > 0 and self.jobs_done >= max_jobs:
                    Clock.schedule_once(lambda dt, mj=max_jobs: self._set_job(f'Limit reached ({mj}) — change in Settings to resume'))
                    self._update_notification(f'Paused — {self.jobs_done}/{max_jobs} jobs')
                    time.sleep(10)
                    continue

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
                job_type = data.get('job_type', params.get('job_type', 'eigenvalue'))
                params['job_type'] = job_type
                progress = data.get('progress', {})

                lam = params.get('lambda', 0)
                k = params.get('k', '?')
                Clock.schedule_once(
                    lambda dt, jid=job_id, l=lam, kk=k:
                    self._set_job(f'Job {jid} | lambda={l:.6f} k={kk}')
                )

                pct = progress.get('percent_complete', 0)
                self._log(f'Job {job_id}: lambda={lam:.6f}, k={k}, hive {pct:.1f}% complete')

                start_time = time.time()
                eigs = self._run_job(params)
                duration = time.time() - start_time

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
                Clock.schedule_once(lambda dt, j=self.jobs_done: self.menger.draw(j))

                # Update tray notification with rich stats
                self._update_rich_notification(hits)

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

                # Cooldown between jobs
                cooldown = load_config().get('cooldown', 0)
                if cooldown > 0:
                    Clock.schedule_once(lambda dt, cd=cooldown: self._set_stage(f'Cooldown {cd}s...'))
                    time.sleep(cooldown)

                # Check for server alerts + updates after each job
                try:
                    vc, vdata = http_get_json(f"{self.server}/version")
                    if vc == 200:
                        alert = vdata.get('alert', '')
                        if alert and alert != getattr(self, '_seen_alert', ''):
                            self._seen_alert = alert
                            self._log(f'[SERVER] {alert}')
                            self._update_notification(f'Alert: {alert}')
                        sv = vdata.get('exe_version', '')
                        if sv and sv != VERSION and sv != getattr(self, '_seen_update', ''):
                            self._seen_update = sv
                            self._log(f'Update available: v{sv}')
                            self._update_notification(f'Update available: v{sv}')
                except Exception:
                    pass

            except Exception as e:
                self._log(f'Error: {e}')
                time.sleep(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF)

        Clock.schedule_once(lambda dt: self._set_job('Stopped'))
        self._log('Worker stopped.')

    def _run_job(self, params):
        """Route job by type."""
        job_type = params.get('job_type', 'eigenvalue')
        if job_type == 'falsification':
            return self._run_falsification(params)
        elif job_type == 'boundary':
            return self._run_boundary(params)
        else:
            return self._run_eigenvalue(params)

    def _run_eigenvalue(self, params):
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

    def _run_falsification(self, params):
        seed = int(params.get('seed', params.get('lambda', 0)))
        Clock.schedule_once(lambda dt: self._set_stage(f'Falsifying seed={seed}...'))
        try:
            from fractal_falsify import run_work_unit
            result = run_work_unit(seed, b=3, level=1)
            if isinstance(result, dict) and 'eigenvalues' in result:
                Clock.schedule_once(lambda dt: self._set_stage('Complete'))
                return np.array(result['eigenvalues'], dtype=np.float64)
            Clock.schedule_once(lambda dt: self._set_stage('Complete (no eigenvalues)'))
            return np.array([0.0], dtype=np.float64)
        except ImportError:
            Clock.schedule_once(lambda dt: self._set_stage('Complete (no falsify module)'))
            return np.array([0.0], dtype=np.float64)

    def _run_boundary(self, params):
        k = params['k']
        G1, G2 = params['G1'], params['G2']
        S = params['S']
        lam = params['lambda']
        w_glue = params['w_glue']

        Clock.schedule_once(lambda dt: self._set_stage('1/4 Building graph (boundary)...'))
        vertices, edges, b_ids = w_operator.build_graph(k, G1, G2, S, 2)

        Clock.schedule_once(lambda dt: self._set_stage('2/4 Boundary edges...'))
        upd, psi_by = w_operator.add_glue_edges(vertices, b_ids, lam, w_glue, G1, G2)
        edges_merged = w_operator.merge_edges(edges, upd)

        boundary_set = set(b_ids)
        boundary_edges = {(u, v): rec for (u, v), rec in edges_merged.items()
                          if u in boundary_set and v in boundary_set}
        if not boundary_edges:
            return np.array([0.0], dtype=np.float64)

        Clock.schedule_once(lambda dt: self._set_stage('3/4 Boundary Laplacian...'))
        L, _ = w_operator.build_magnetic_laplacian(
            {v: vertices[v] for v in boundary_set}, boundary_edges,
            s=(0, 0), psi_by_id=psi_by)

        Clock.schedule_once(lambda dt: self._set_stage('4/4 Solving boundary spectrum...'))
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
        sm.add_widget(SettingsScreen(name='settings'))
        sm.add_widget(ChatScreen(name='chat'))
        sm.add_widget(DashboardScreen(name='dashboard'))
        sm.add_widget(SpongeScreen(name='sponge'))

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
        # Android: allow going to background — service keeps computing
        return True

    def on_resume(self):
        # UI will pick up service status via polling
        pass

    def on_stop(self):
        # App closing — but DON'T kill the foreground service
        # The service runs independently and survives activity destruction
        try:
            compute = self.root.get_screen('compute')
            if compute.running and not compute._service_mode:
                # Only stop in-process compute, never the foreground service
                compute.on_stop()
        except Exception:
            pass


if __name__ == '__main__':
    WAtHomeApp().run()
