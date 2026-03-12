"""
W@Home Hive — Windows Desktop Application

GUI client for the Akataleptos Distributed Spectral Search.
System tray, dashboard, chat, screensaver launch, auto-update.

Dependencies: tkinter (built-in), requests, numpy, scipy
Optional: pystray + Pillow (system tray), pygame + moderngl (screensaver)
"""

import os
import sys

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
import json
import hashlib
import time
import platform
import subprocess

# Ensure local imports work
if getattr(sys, 'frozen', False):
    # Frozen exe — config goes in %LOCALAPPDATA%\WHome (persistent)
    # Code/imports come from the temp extraction dir
    _CODE_DIR = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
    APP_DIR = os.path.join(
        os.environ.get('LOCALAPPDATA', os.path.dirname(sys.executable)),
        'WHome'
    )
    os.makedirs(APP_DIR, exist_ok=True)
else:
    _CODE_DIR = os.path.dirname(os.path.abspath(__file__))
    APP_DIR = _CODE_DIR
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import requests
import numpy as np

# ── Feature detection ──

try:
    import w_cuda
    HAS_GPU = w_cuda.HAS_GPU
    GPU_INFO = "CUDA (CuPy)"
except ImportError:
    HAS_GPU = False
    GPU_INFO = "CPU only"

try:
    import w_operator as base_op
    HAS_OPERATOR = True
except ImportError:
    HAS_OPERATOR = False

try:
    import pystray
    from PIL import Image, ImageDraw
    HAS_TRAY = True
except ImportError:
    HAS_TRAY = False

# Screensaver: always available when frozen (same exe), check .py when from source
if getattr(sys, 'frozen', False):
    HAS_SCREENSAVER = True
else:
    HAS_SCREENSAVER = os.path.exists(os.path.join(APP_DIR, 'screensaver.py'))

import webbrowser

# ═══════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════

VERSION = "1.0.2"
DEFAULT_SERVER = "https://wathome.akataleptos.com"
CONFIG_PATH = os.path.join(APP_DIR, "worker_config.json")
CHECKPOINT_PATH = os.path.join(APP_DIR, "checkpoint.json")

PHYSICAL_CONSTANTS = {
    "phi": 1.6180339887, "e": 2.718281828, "pi": 3.141592653,
    "alpha_inv": 137.035999, "proton_electron": 1836.15267,
    "sqrt2": 1.4142135624, "sqrt3": 1.7320508076, "ln2": 0.6931471806,
}
TOLERANCE = 1e-4

# Colors
C = {
    'bg': '#0d0d1a', 'surface': '#161628', 'surface2': '#1e1e38',
    'border': '#2a2a4a', 'text': '#c8d0e0', 'dim': '#606878',
    'cyan': '#60e8ff', 'gold': '#ffd06a', 'violet': '#c4a0ff',
    'green': '#80ffaa', 'red': '#ff6666',
    'btn': '#2a3a5a', 'btn_active': '#3a4a7a',
}
MONO = 'Menlo' if sys.platform == 'darwin' else 'Consolas'

# ═══════════════════════════════════════════════════════════
# Config helpers
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

def _set_boot_autostart(enable):
    """Register/unregister app to start on boot (platform-specific)."""
    try:
        if sys.platform == 'win32':
            import winreg
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                                r"Software\Microsoft\Windows\CurrentVersion\Run",
                                0, winreg.KEY_SET_VALUE)
            if enable:
                # Use the exe path if frozen, otherwise python + script
                if getattr(sys, 'frozen', False):
                    winreg.SetValueEx(key, "WHome", 0, winreg.REG_SZ, sys.executable)
                else:
                    winreg.SetValueEx(key, "WHome", 0, winreg.REG_SZ,
                                      f'"{sys.executable}" "{os.path.abspath(__file__)}"')
            else:
                try:
                    winreg.DeleteValue(key, "WHome")
                except FileNotFoundError:
                    pass
            winreg.CloseKey(key)
        elif sys.platform == 'linux':
            desktop_path = os.path.expanduser('~/.config/autostart/whome.desktop')
            if enable:
                os.makedirs(os.path.dirname(desktop_path), exist_ok=True)
                exe = os.path.join(os.path.expanduser('~'), '.whome', 'whome-gui')
                icon = os.path.join(os.path.expanduser('~'), '.whome', 'icon-menger-256.png')
                with open(desktop_path, 'w') as f:
                    f.write(f"[Desktop Entry]\nType=Application\nName=W@Home Hive\n"
                            f"Comment=Akataleptos Distributed Spectral Search\n"
                            f"Exec={exe}\nIcon={icon}\nTerminal=false\n"
                            f"Hidden=false\nX-GNOME-Autostart-enabled=true\n")
            else:
                if os.path.exists(desktop_path):
                    os.remove(desktop_path)
    except Exception:
        pass

# ═══════════════════════════════════════════════════════════
# Thermal protection
# ═══════════════════════════════════════════════════════════

THERMAL_PAUSE = 85   # pause computation above this °C
THERMAL_RESUME = 75  # resume when cooled to this °C

def get_cpu_temp():
    """Get current CPU temperature in °C. Cross-platform."""
    try:
        if sys.platform == 'linux':
            # Read all thermal zones, return the hottest
            best = 0
            for i in range(20):
                try:
                    with open(f'/sys/class/thermal/thermal_zone{i}/temp') as f:
                        t = int(f.read().strip()) / 1000
                        if t > best:
                            best = t
                except (FileNotFoundError, ValueError):
                    break
            return best if best > 0 else None
        elif sys.platform == 'win32':
            # WMI temperature query
            import subprocess
            r = subprocess.run(
                ['powershell', '-Command',
                 'Get-CimInstance MSAcpi_ThermalZoneTemperature -Namespace root/wmi 2>$null | '
                 'Select-Object -ExpandProperty CurrentTemperature | Sort-Object -Descending | '
                 'Select-Object -First 1'],
                capture_output=True, text=True, timeout=5)
            if r.returncode == 0 and r.stdout.strip():
                # WMI returns deciKelvin
                return (int(r.stdout.strip()) / 10) - 273.15
            return None
        elif sys.platform == 'darwin':
            # macOS — try osx-cpu-temp if installed
            import subprocess
            r = subprocess.run(['osx-cpu-temp'], capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                return float(r.stdout.strip().replace('°C', ''))
            return None
    except Exception:
        return None

# ═══════════════════════════════════════════════════════════
# Network helpers
# ═══════════════════════════════════════════════════════════

def api_get(server, path, api_key=None, timeout=10):
    headers = {"x-api-key": api_key} if api_key else {}
    resp = requests.get(f"{server}{path}", headers=headers, timeout=timeout)
    return resp.status_code, resp.json() if resp.status_code == 200 else {}

def api_post(server, path, data, api_key=None, timeout=30):
    headers = {"x-api-key": api_key} if api_key else {}
    resp = requests.post(f"{server}{path}", json=data, headers=headers, timeout=timeout)
    try:
        body = resp.json()
    except Exception:
        body = {"detail": resp.text}
    return resp.status_code, body

# ═══════════════════════════════════════════════════════════
# Computation helpers
# ═══════════════════════════════════════════════════════════

def hash_eigenvalues(eigs):
    rounded = [round(float(e), 10) for e in sorted(eigs)]
    payload = json.dumps(rounded, separators=(',', ':'))
    return hashlib.sha256(payload.encode()).hexdigest()

def check_for_gold(eigs):
    eigs = np.sort(eigs[eigs > 1e-9])
    found = []
    cap = min(len(eigs), 60)
    for i in range(cap):
        for j in range(i + 1, cap):
            ratio = eigs[j] / eigs[i]
            for name, val in PHYSICAL_CONSTANTS.items():
                if abs(ratio - val) / val < TOLERANCE:
                    found.append(f"{name} (ratio={ratio:.6f}, i={i}, j={j})")
    return found

# ═══════════════════════════════════════════════════════════
# Registration / Login
# ═══════════════════════════════════════════════════════════

def do_register(server, name, password, gpu_info):
    code, data = api_post(server, "/register", {
        "name": name, "gpu_info": gpu_info,
        "password": password, "device_name": platform.node(),
    })
    if code == 409:
        raise NameTakenError(data.get('detail', 'Name already taken'))
    if code != 200:
        raise RuntimeError(data.get('detail', f'Registration failed ({code})'))
    cfg = load_config()
    cfg.update(api_key=data['api_key'], worker_id=data['worker_id'],
               name=name, server=server)
    save_config(cfg)
    return data['api_key'], data['worker_id']

class NameTakenError(Exception):
    pass

def do_login(server, name, password):
    code, data = api_post(server, "/login", {
        "name": name, "password": password,
        "device_name": platform.node(), "gpu_info": GPU_INFO,
    })
    if code != 200:
        raise RuntimeError(data.get('detail', f'Login failed ({code})'))
    cfg = load_config()
    cfg.update(api_key=data['api_key'], worker_id=data['worker_id'],
               name=name, server=server)
    save_config(cfg)
    return data['api_key'], data['worker_id']

# ═══════════════════════════════════════════════════════════
# Tray icon
# ═══════════════════════════════════════════════════════════

def create_tray_icon(state='idle', alert=False):
    """Create a 64x64 Menger-pattern tray icon. Gold dot overlay if alert=True."""
    if not HAS_TRAY:
        return None
    img = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    palette = {'idle': (80, 200, 120), 'computing': (128, 255, 170),
               'error': (255, 102, 102), 'paused': (255, 208, 106)}
    color = palette.get(state, palette['idle'])
    s = 20
    for x in range(3):
        for y in range(3):
            if x == 1 and y == 1:
                continue
            draw.rectangle([2 + x*s, 2 + y*s, 2 + (x+1)*s - 2, 2 + (y+1)*s - 2],
                           fill=(*color, 220))
    if alert:
        # Gold notification dot — top right corner
        draw.ellipse([44, 0, 63, 19], fill=(255, 208, 106, 255))
        draw.ellipse([47, 3, 60, 16], fill=(255, 160, 30, 255))
    return img

# ═══════════════════════════════════════════════════════════
# Compute Worker Thread
# ═══════════════════════════════════════════════════════════

class ComputeWorker(threading.Thread):
    """Background thread that fetches jobs, computes, and submits results."""

    def __init__(self, server, api_key, msg_queue):
        super().__init__(daemon=True)
        self.server = server
        self.api_key = api_key
        self.msg_queue = msg_queue
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._pause.set()  # not paused
        self.jobs_done = 0
        self.compute_hours = 0.0
        self.discoveries = 0
        self._seen_alert = ''
        self._seen_update = ''

    def emit(self, event, data=None):
        self.msg_queue.put((event, data or {}))

    def stop(self):
        self._stop.set()
        self._pause.set()  # unblock if paused

    def pause(self):
        self._pause.clear()
        self.emit('status', {'text': 'Paused', 'state': 'paused'})

    def resume(self):
        self._pause.set()
        self.emit('status', {'text': 'Resuming...', 'state': 'computing'})

    @property
    def paused(self):
        return not self._pause.is_set()

    def _thermal_wait(self):
        """Pause if CPU temperature exceeds safe threshold."""
        temp = get_cpu_temp()
        if temp is None or temp < THERMAL_PAUSE:
            return
        self.emit('status', {'text': f'Cooling down ({temp:.0f}°C)...', 'state': 'paused'})
        while not self._stop.is_set():
            self._stop.wait(10)
            temp = get_cpu_temp()
            if temp is None or temp < THERMAL_RESUME:
                self.emit('status', {'text': f'Cooled to {temp:.0f}°C — resuming', 'state': 'computing'})
                return

    def run(self):
        self.emit('status', {'text': 'Worker started', 'state': 'computing'})
        backoff = 2

        # Start heartbeat
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
            self._pause.wait()
            if self._stop.is_set():
                break

            # Thermal protection — wait if CPU is too hot
            self._thermal_wait()

            try:
                self.emit('status', {'text': 'Fetching job...', 'state': 'computing'})
                code, data = api_post(self.server, "/job", {}, self.api_key)

                if code == 401:
                    self.emit('error', {'text': 'API key rejected. Re-register.'})
                    break
                if code != 200:
                    self.emit('error', {'text': f'Server error ({code})'})
                    self._stop.wait(backoff)
                    backoff = min(backoff * 2, 120)
                    continue

                if data.get('status') == 'no_jobs':
                    self.emit('status', {'text': 'No jobs available. Waiting...', 'state': 'idle'})
                    self._stop.wait(60)
                    continue

                backoff = 2
                job_id = data['job_id']
                params = data['params']
                params['job_id'] = job_id
                progress = data.get('progress', {})

                self.emit('job_start', {
                    'job_id': job_id, 'lambda': params['lambda'],
                    'params': params, 'progress': progress,
                })

                start = time.time()
                eigs = self._run_job(params)
                duration = time.time() - start

                if self._stop.is_set():
                    break

                hits = check_for_gold(eigs)
                self.jobs_done += 1
                self.compute_hours += duration / 3600
                if hits:
                    self.discoveries += len(hits)

                self.emit('job_done', {
                    'job_id': job_id, 'hits': hits,
                    'n_eigs': int(np.sum(eigs > 1e-9)),
                    'duration': duration,
                    'session_jobs': self.jobs_done,
                    'session_hours': self.compute_hours,
                    'session_disc': self.discoveries,
                })

                # Submit result
                eig_hash = hash_eigenvalues(eigs.tolist())
                api_post(self.server, "/result", {
                    "job_id": job_id, "eigenvalues": eigs.tolist(),
                    "eigenvalues_hash": eig_hash,
                    "found_constants": hits, "compute_seconds": duration,
                }, self.api_key)

                # Check for updates + server alerts after each job
                try:
                    vc, vdata = api_get(self.server, "/version")
                    if vc == 200:
                        # Alert first (non-blocking), then update prompt (blocking dialog)
                        # Only emit once per unique message/version
                        alert = vdata.get('alert', '')
                        if alert and alert != self._seen_alert:
                            self._seen_alert = alert
                            self.emit('server_alert', {'message': alert})
                        sv = vdata.get('exe_version', '')
                        if sv and sv != VERSION and sv != self._seen_update:
                            self._seen_update = sv
                            self.emit('update_available', {'version': sv})
                except Exception:
                    pass

            except requests.ConnectionError:
                self.emit('error', {'text': 'Cannot reach server'})
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 120)
            except Exception as e:
                self.emit('error', {'text': str(e)})
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 120)

        self.emit('status', {'text': 'Worker stopped', 'state': 'idle'})

    def _run_job(self, params):
        k, G1, G2, S = params['k'], params['G1'], params['G2'], params['S']
        lam, w_glue, N = params['lambda'], params['w_glue'], 2

        self.emit('stage', {'num': 1, 'name': 'Build Graph', 'detail': f'k={k}'})
        vertices, edges, b_ids = base_op.build_graph(k, G1, G2, S, N)
        if self._stop.is_set(): return np.array([])

        self._thermal_wait()

        self.emit('stage', {'num': 2, 'name': 'Add Glue', 'detail': f'\u03bb={lam:.6f}'})
        upd, psi_by = base_op.add_glue_edges(vertices, b_ids, lam, w_glue, G1, G2)
        if self._stop.is_set(): return np.array([])

        self.emit('stage', {'num': 3, 'name': 'Merge Edges',
                            'detail': f'{len(edges):,} + {len(upd):,}'})
        edges_merged = base_op.merge_edges(edges, upd)
        if self._stop.is_set(): return np.array([])

        self._thermal_wait()

        self.emit('stage', {'num': 4, 'name': 'Build Laplacian',
                            'detail': f'{len(vertices):,} vertices'})
        L, _ = base_op.build_magnetic_laplacian(vertices, edges_merged,
                                                 s=(0, 0), psi_by_id=psi_by)
        if self._stop.is_set(): return np.array([])

        self._thermal_wait()

        self.emit('stage', {'num': 5, 'name': 'Solve Spectrum', 'detail': 'eigsh M=40'})
        if HAS_GPU:
            eigs = w_cuda.solve_spectrum_gpu(L, M=40)
        else:
            eigs = base_op.solve_spectrum(L, M=40)

        clear_checkpoint()
        return eigs

# ═══════════════════════════════════════════════════════════
# Theme
# ═══════════════════════════════════════════════════════════

def setup_theme(root):
    style = ttk.Style(root)
    style.theme_use('clam')

    style.configure('.', background=C['bg'], foreground=C['text'],
                    fieldbackground=C['surface'], bordercolor=C['border'],
                    insertcolor=C['text'], selectbackground=C['btn'],
                    selectforeground=C['text'], focuscolor=C['border'])
    style.configure('TNotebook', background=C['bg'], borderwidth=0, tabmargins=[4, 4, 4, 0])
    style.configure('TNotebook.Tab', background=C['surface'], foreground=C['dim'],
                    padding=[14, 6], borderwidth=0)
    style.map('TNotebook.Tab',
              background=[('selected', C['surface2'])],
              foreground=[('selected', C['cyan'])])
    style.configure('TFrame', background=C['bg'])
    style.configure('Card.TFrame', background=C['surface'], relief='flat')
    style.configure('TLabel', background=C['bg'], foreground=C['text'])
    style.configure('Card.TLabel', background=C['surface'])
    style.configure('Dim.TLabel', foreground=C['dim'])
    style.configure('Cyan.TLabel', foreground=C['cyan'])
    style.configure('Gold.TLabel', foreground=C['gold'])
    style.configure('Green.TLabel', foreground=C['green'])
    style.configure('Red.TLabel', foreground=C['red'])
    style.configure('Title.TLabel', foreground=C['cyan'], font=(MONO, 14, 'bold'))
    style.configure('TButton', background=C['btn'], foreground=C['text'],
                    borderwidth=1, padding=[10, 4])
    style.map('TButton', background=[('active', C['btn_active']),
                                      ('pressed', C['surface'])])
    style.configure('Green.TButton', background='#1a4a3a', foreground=C['green'])
    style.map('Green.TButton', background=[('active', '#2a5a4a')])
    style.configure('Red.TButton', background='#4a1a1a', foreground=C['red'])
    style.map('Red.TButton', background=[('active', '#5a2a2a')])
    style.configure('TEntry', fieldbackground=C['surface2'], foreground=C['text'],
                    insertcolor=C['text'], borderwidth=1, padding=[4, 4])
    style.configure('TCheckbutton', background=C['bg'], foreground=C['text'])
    style.map('TCheckbutton', background=[('active', C['bg'])])
    style.configure('Cyan.Horizontal.TProgressbar',
                    troughcolor=C['surface'], background=C['cyan'], borderwidth=0)
    style.configure('Treeview', background=C['surface'], foreground=C['text'],
                    fieldbackground=C['surface'], borderwidth=0, rowheight=26)
    style.configure('Treeview.Heading', background=C['surface2'],
                    foreground=C['cyan'], borderwidth=0)
    style.map('Treeview', background=[('selected', C['btn'])],
              foreground=[('selected', C['cyan'])])
    style.configure('TScrollbar', background=C['surface'], troughcolor=C['bg'],
                    borderwidth=0, arrowcolor=C['dim'])

# ═══════════════════════════════════════════════════════════
# Setup Dialog (first-run registration)
# ═══════════════════════════════════════════════════════════

class SetupDialog:
    def __init__(self, parent):
        self.result = None
        self.dlg = tk.Toplevel(parent)
        self.dlg.title("W@Home — Setup")
        self.dlg.geometry("420x380")
        self.dlg.configure(bg=C['bg'])
        self.dlg.grab_set()
        self.dlg.resizable(False, False)
        self.dlg.lift()
        self.dlg.focus_force()

        ttk.Label(self.dlg, text="W@HOME HIVE", style='Title.TLabel').pack(pady=(20, 2))
        ttk.Label(self.dlg, text="First-time setup", style='Dim.TLabel').pack(pady=(0, 16))

        form = ttk.Frame(self.dlg)
        form.pack(padx=30, fill='x')

        ttk.Label(form, text="Node name:").grid(row=0, column=0, sticky='w', pady=4)
        self.name_var = tk.StringVar(value=f"{platform.node()}-{os.getenv('USER', os.getenv('USERNAME', 'worker'))}")
        ttk.Entry(form, textvariable=self.name_var, width=32).grid(row=0, column=1, pady=4, padx=(8,0))

        ttk.Label(form, text="Password:").grid(row=1, column=0, sticky='w', pady=4)
        self.pw_var = tk.StringVar()
        ttk.Entry(form, textvariable=self.pw_var, show='*', width=32).grid(row=1, column=1, pady=4, padx=(8,0))

        ttk.Label(form, text="Confirm:").grid(row=2, column=0, sticky='w', pady=4)
        self.pw2_var = tk.StringVar()
        ttk.Entry(form, textvariable=self.pw2_var, show='*', width=32).grid(row=2, column=1, pady=4, padx=(8,0))

        ttk.Label(form, text="Server:").grid(row=3, column=0, sticky='w', pady=4)
        self.server_var = tk.StringVar(value=DEFAULT_SERVER)
        ttk.Entry(form, textvariable=self.server_var, width=32).grid(row=3, column=1, pady=4, padx=(8,0))

        self.status = ttk.Label(self.dlg, text="", style='Dim.TLabel')
        self.status.pack(pady=(12, 4))

        btn_frame = ttk.Frame(self.dlg)
        btn_frame.pack(pady=8)
        ttk.Button(btn_frame, text="Register", style='Green.TButton',
                   command=self._register).pack(side='left', padx=4)
        ttk.Button(btn_frame, text="Login", command=self._login).pack(side='left', padx=4)

        ttk.Label(self.dlg, text=f"GPU: {GPU_INFO}", style='Dim.TLabel').pack(pady=(8, 4))

    def _register(self):
        name = self.name_var.get().strip()
        pw = self.pw_var.get()
        pw2 = self.pw2_var.get()
        server = self.server_var.get().strip().rstrip('/')
        if not name:
            self.status.config(text="Name required", style='Red.TLabel')
            return
        if len(pw) < 4:
            self.status.config(text="Password min 4 chars", style='Red.TLabel')
            return
        if pw != pw2:
            self.status.config(text="Passwords don't match", style='Red.TLabel')
            return
        self.status.config(text="Registering...", style='Dim.TLabel')
        self.dlg.update()
        try:
            api_key, worker_id = do_register(server, name, pw, GPU_INFO)
            self.result = {'api_key': api_key, 'worker_id': worker_id,
                           'name': name, 'server': server}
            self.dlg.destroy()
        except NameTakenError:
            self.status.config(text="Name taken. Try Login instead.", style='Gold.TLabel')
        except Exception as e:
            self.status.config(text=str(e)[:50], style='Red.TLabel')

    def _login(self):
        name = self.name_var.get().strip()
        pw = self.pw_var.get()
        server = self.server_var.get().strip().rstrip('/')
        if not name or not pw:
            self.status.config(text="Name and password required", style='Red.TLabel')
            return
        self.status.config(text="Logging in...", style='Dim.TLabel')
        self.dlg.update()
        try:
            api_key, worker_id = do_login(server, name, pw)
            self.result = {'api_key': api_key, 'worker_id': worker_id,
                           'name': name, 'server': server}
            self.dlg.destroy()
        except Exception as e:
            self.status.config(text=str(e)[:50], style='Red.TLabel')

    def wait(self):
        self.dlg.wait_window()
        return self.result

# ═══════════════════════════════════════════════════════════
# Compute Tab
# ═══════════════════════════════════════════════════════════

class ComputeTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self._build()

    def _build(self):
        # Control buttons
        btn_row = ttk.Frame(self)
        btn_row.pack(fill='x', padx=12, pady=(12, 6))

        self.start_btn = ttk.Button(btn_row, text="\u25b6  Start", style='Green.TButton',
                                    command=self.app.start_compute)
        self.start_btn.pack(side='left', padx=(0, 6))

        self.stop_btn = ttk.Button(btn_row, text="\u25a0  Stop", style='Red.TButton',
                                   command=self.app.stop_compute, state='disabled')
        self.stop_btn.pack(side='left', padx=(0, 6))

        self.pause_btn = ttk.Button(btn_row, text="\u23f8  Pause",
                                    command=self.app.toggle_pause, state='disabled')
        self.pause_btn.pack(side='left', padx=(0, 6))

        if HAS_SCREENSAVER:
            ttk.Button(btn_row, text="Screensaver",
                       command=self.app.launch_screensaver).pack(side='right')

        # Status card
        card = ttk.Frame(self, style='Card.TFrame')
        card.pack(fill='x', padx=12, pady=6)
        card_inner = ttk.Frame(card, style='Card.TFrame')
        card_inner.pack(fill='x', padx=12, pady=10)

        self.status_label = ttk.Label(card_inner, text="Idle", style='Cyan.TLabel',
                                      font=(MONO, 12, 'bold'))
        self.status_label.pack(anchor='w')

        self.job_label = ttk.Label(card_inner, text="", style='Card.TLabel',
                                   font=(MONO, 10))
        self.job_label.pack(anchor='w', pady=(4, 8))

        # Stage indicators
        self.stage_frame = ttk.Frame(card_inner, style='Card.TFrame')
        self.stage_frame.pack(anchor='w')
        self.stage_labels = []
        stage_names = ['Build Graph', 'Add Glue', 'Merge Edges',
                       'Build Laplacian', 'Solve Spectrum']
        for name in stage_names:
            lbl = ttk.Label(self.stage_frame, text=f"  \u25cb  {name}",
                            style='Card.TLabel', font=(MONO, 10),
                            foreground=C['dim'])
            lbl.pack(anchor='w', pady=1)
            self.stage_labels.append(lbl)

        # Session stats
        stats_row = ttk.Frame(self)
        stats_row.pack(fill='x', padx=12, pady=6)
        self.stats_label = ttk.Label(stats_row, text="Session: 0 jobs, 0.00h compute",
                                     style='Dim.TLabel', font=(MONO, 9))
        self.stats_label.pack(anchor='w')

        # Log
        ttk.Label(self, text="Log", style='Dim.TLabel',
                  font=(MONO, 9)).pack(anchor='w', padx=14, pady=(8, 2))
        self.log = tk.Text(self, height=10, bg=C['surface'], fg=C['text'],
                           insertbackground=C['text'], selectbackground=C['btn'],
                           font=(MONO, 9), borderwidth=1, relief='flat',
                           state='disabled', wrap='word')
        self.log.pack(fill='both', expand=True, padx=12, pady=(0, 12))
        self.log.tag_config('time', foreground=C['dim'])
        self.log.tag_config('info', foreground=C['text'])
        self.log.tag_config('gold', foreground=C['gold'])
        self.log.tag_config('error', foreground=C['red'])
        self.log.tag_config('green', foreground=C['green'])

    def log_msg(self, text, tag='info'):
        ts = time.strftime('%H:%M:%S')
        self.log.config(state='normal')
        self.log.insert('end', f"[{ts}] ", 'time')
        self.log.insert('end', text + '\n', tag)
        self.log.see('end')
        self.log.config(state='disabled')

    def set_status(self, text):
        self.status_label.config(text=text)

    def set_job(self, job_id, lam, params):
        self.job_label.config(text=f"Job #{job_id}  |  \u03bb = {lam:.6f}  |  "
                                   f"k={params.get('k','?')}  G={params.get('G1','?')}\u00d7{params.get('G2','?')}")

    def set_stage(self, num):
        stage_names = ['Build Graph', 'Add Glue', 'Merge Edges',
                       'Build Laplacian', 'Solve Spectrum']
        for i, lbl in enumerate(self.stage_labels):
            if i + 1 < num:
                lbl.config(text=f"  \u25cf  {stage_names[i]}", foreground=C['green'])
            elif i + 1 == num:
                lbl.config(text=f"  \u25c9  {stage_names[i]}", foreground=C['cyan'])
            else:
                lbl.config(text=f"  \u25cb  {stage_names[i]}", foreground=C['dim'])

    def reset_stages(self):
        stage_names = ['Build Graph', 'Add Glue', 'Merge Edges',
                       'Build Laplacian', 'Solve Spectrum']
        for i, lbl in enumerate(self.stage_labels):
            lbl.config(text=f"  \u25cb  {stage_names[i]}", foreground=C['dim'])

    def set_computing(self, active):
        if active:
            self.start_btn.config(state='disabled')
            self.stop_btn.config(state='normal')
            self.pause_btn.config(state='normal')
        else:
            self.start_btn.config(state='normal')
            self.stop_btn.config(state='disabled')
            self.pause_btn.config(state='disabled')
            self.pause_btn.config(text="\u23f8  Pause")

    def update_stats(self, jobs, hours, disc):
        self.stats_label.config(
            text=f"Session: {jobs} jobs, {hours:.2f}h compute, {disc} discoveries")

# ═══════════════════════════════════════════════════════════
# Dashboard Tab
# ═══════════════════════════════════════════════════════════

class DashboardTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self._build()

    def _build(self):
        ttk.Label(self, text="HIVE STATUS", style='Title.TLabel').pack(
            anchor='w', padx=14, pady=(14, 8))

        # Progress bar
        bar_frame = ttk.Frame(self)
        bar_frame.pack(fill='x', padx=14, pady=(0, 4))
        self.progress = ttk.Progressbar(bar_frame, style='Cyan.Horizontal.TProgressbar',
                                         length=400, mode='determinate')
        self.progress.pack(fill='x')

        self.pct_label = ttk.Label(self, text="0.00%", style='Cyan.TLabel',
                                   font=(MONO, 11))
        self.pct_label.pack(anchor='w', padx=14)

        # Stats row
        self.stats_label = ttk.Label(self, text="Workers: --  |  Jobs/hr: --  |  Discoveries: --",
                                     style='Dim.TLabel', font=(MONO, 9))
        self.stats_label.pack(anchor='w', padx=14, pady=(4, 12))

        # Dashboard button
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill='x', padx=14, pady=(0, 12))
        ttk.Button(btn_frame, text="Open Full Dashboard", style='Green.TButton',
                   command=self._open_dashboard).pack(side='left')
        ttk.Label(btn_frame, text="  wathome.akataleptos.com — live data, always current",
                  style='Dim.TLabel', font=(MONO, 9)).pack(side='left')

        # Two-column: leaderboard + discoveries
        cols = ttk.Frame(self)
        cols.pack(fill='both', expand=True, padx=14, pady=(0, 12))
        cols.columnconfigure(0, weight=1)
        cols.columnconfigure(1, weight=1)

        # Leaderboard
        left = ttk.Frame(cols)
        left.grid(row=0, column=0, sticky='nsew', padx=(0, 6))
        ttk.Label(left, text="Leaderboard", style='Dim.TLabel',
                  font=(MONO, 9)).pack(anchor='w', pady=(0, 4))
        self.leader_tree = ttk.Treeview(left, columns=('rank', 'name', 'jobs'),
                                         show='headings', height=8)
        self.leader_tree.heading('rank', text='#')
        self.leader_tree.heading('name', text='Name')
        self.leader_tree.heading('jobs', text='Jobs')
        self.leader_tree.column('rank', width=30, anchor='center')
        self.leader_tree.column('name', width=120)
        self.leader_tree.column('jobs', width=60, anchor='e')
        self.leader_tree.pack(fill='both', expand=True)

        # Discoveries
        right = ttk.Frame(cols)
        right.grid(row=0, column=1, sticky='nsew', padx=(6, 0))
        ttk.Label(right, text="Recent Discoveries", style='Gold.TLabel',
                  font=(MONO, 9)).pack(anchor='w', pady=(0, 4))
        self.disc_text = tk.Text(right, height=8, bg=C['surface'], fg=C['gold'],
                                  font=(MONO, 9), borderwidth=0, state='disabled',
                                  wrap='word')
        self.disc_text.pack(fill='both', expand=True)

    def _open_dashboard(self):
        webbrowser.open("https://wathome.akataleptos.com/dashboard")

    def update_progress(self, data):
        pct = data.get('percent_complete', 0)
        self.progress['value'] = pct
        self.pct_label.config(text=f"{pct:.2f}%")
        workers = data.get('active_workers', 0)
        jph = data.get('jobs_per_hour', 0)
        disc = data.get('total_discoveries', 0)
        eta = data.get('eta_hours', '')
        text = f"Workers: {workers}  |  Jobs/hr: {jph}  |  Discoveries: {disc}"
        if eta:
            text += f"  |  ETA: {eta}h"
        self.stats_label.config(text=text)

    def update_leaderboard(self, leaders):
        for item in self.leader_tree.get_children():
            self.leader_tree.delete(item)
        for i, entry in enumerate(leaders[:10], 1):
            name = entry.get('name', '?')
            jobs = entry.get('jobs', 0)
            self.leader_tree.insert('', 'end', values=(i, name, f"{jobs:,}"))

    def update_discoveries(self, discoveries):
        self.disc_text.config(state='normal')
        self.disc_text.delete('1.0', 'end')
        for d in discoveries[:8]:
            name = d.get('constant_name', '?')
            lam = d.get('lambda_val', 0)
            ratio = d.get('ratio_value', 0)
            self.disc_text.insert('end', f"{name} at \u03bb={lam:.6f} (r={ratio:.5f})\n")
        self.disc_text.config(state='disabled')

# ═══════════════════════════════════════════════════════════
# Chat Tab
# ═══════════════════════════════════════════════════════════

class ChatTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self._last_msg_id = 0
        self._build()

    def _build(self):
        # Header
        header = ttk.Frame(self)
        header.pack(fill='x', padx=14, pady=(12, 6))
        ttk.Label(header, text="HIVE CHAT", style='Title.TLabel').pack(side='left')
        self.online_label = ttk.Label(header, text="", style='Dim.TLabel',
                                      font=(MONO, 9))
        self.online_label.pack(side='right')

        # Messages
        self.chat_text = tk.Text(self, bg=C['surface'], fg=C['text'],
                                  font=(MONO, 10), borderwidth=0,
                                  state='disabled', wrap='word', spacing3=3)
        self.chat_text.pack(fill='both', expand=True, padx=14, pady=(0, 6))
        self.chat_text.tag_config('user', foreground=C['cyan'], font=(MONO, 10, 'bold'))
        self.chat_text.tag_config('time', foreground=C['dim'])
        self.chat_text.tag_config('msg', foreground=C['text'])
        self.chat_text.tag_config('system', foreground=C['violet'])

        # Input
        input_row = ttk.Frame(self)
        input_row.pack(fill='x', padx=14, pady=(0, 12))
        self.msg_entry = ttk.Entry(input_row, font=(MONO, 10))
        self.msg_entry.pack(side='left', fill='x', expand=True, padx=(0, 6))
        self.msg_entry.bind('<Return>', lambda e: self._send())
        ttk.Button(input_row, text="Send", command=self._send).pack(side='right')

    def _send(self):
        text = self.msg_entry.get().strip()
        if not text:
            return
        self.msg_entry.delete(0, 'end')
        threading.Thread(target=self._do_send, args=(text,), daemon=True).start()

    def _do_send(self, text):
        try:
            api_post(self.app.server, "/chat/send",
                     {"content": text}, self.app.api_key)
        except Exception:
            pass

    def add_message(self, msg):
        self.chat_text.config(state='normal')
        name = msg.get('username', msg.get('name', '?'))
        text = msg.get('content', msg.get('message', ''))
        ts = msg.get('time', '')
        if ts:
            try:
                import datetime
                dt = datetime.datetime.fromtimestamp(float(ts))
                ts = dt.strftime('%H:%M')
            except Exception:
                ts = ''
        self.chat_text.insert('end', f"{name}", 'user')
        if ts:
            self.chat_text.insert('end', f" {ts}", 'time')
        self.chat_text.insert('end', f"  {text}\n", 'msg')
        self.chat_text.see('end')
        self.chat_text.config(state='disabled')

    def poll_messages(self):
        try:
            code, data = api_get(self.app.server,
                                 "/chat/history?limit=20", self.app.api_key)
            if code == 200 and isinstance(data, list):
                # Use timestamp as ID since server doesn't provide one
                for msg in data:
                    msg_time = msg.get('time', 0)
                    if msg_time > self._last_msg_id:
                        self._last_msg_id = msg_time
                        self.app.root.after(0, self.add_message, msg)
        except Exception:
            pass
        try:
            code, data = api_get(self.app.server, "/chat/online", self.app.api_key)
            if code == 200:
                chat_users = data.get('chat', [])
                computing = data.get('computing', [])
                total = len(set(chat_users + computing))
                self.app.root.after(0, self.online_label.config,
                                    {'text': f"Online: {total}"})
        except Exception:
            pass

# ═══════════════════════════════════════════════════════════
# Settings Tab
# ═══════════════════════════════════════════════════════════

class SettingsTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self._build()

    def _build(self):
        ttk.Label(self, text="SETTINGS", style='Title.TLabel').pack(
            anchor='w', padx=14, pady=(14, 12))

        form = ttk.Frame(self)
        form.pack(fill='x', padx=14)

        # Worker name
        ttk.Label(form, text="Worker name:").grid(row=0, column=0, sticky='w', pady=6)
        self.name_var = tk.StringVar(value=self.app.config.get('name', ''))
        ttk.Entry(form, textvariable=self.name_var, width=30).grid(
            row=0, column=1, sticky='w', padx=(8, 0), pady=6)

        # Server
        ttk.Label(form, text="Server:").grid(row=1, column=0, sticky='w', pady=6)
        self.server_var = tk.StringVar(value=self.app.server)
        ttk.Entry(form, textvariable=self.server_var, width=30).grid(
            row=1, column=1, sticky='w', padx=(8, 0), pady=6)

        # API Key (read-only)
        ttk.Label(form, text="API Key:").grid(row=2, column=0, sticky='w', pady=6)
        key_text = self.app.api_key[:16] + '...' if self.app.api_key else 'none'
        ttk.Label(form, text=key_text, style='Dim.TLabel').grid(
            row=2, column=1, sticky='w', padx=(8, 0), pady=6)

        # Worker ID
        ttk.Label(form, text="Worker ID:").grid(row=3, column=0, sticky='w', pady=6)
        ttk.Label(form, text=self.app.config.get('worker_id', '?'),
                  style='Dim.TLabel').grid(row=3, column=1, sticky='w', padx=(8, 0), pady=6)

        # GPU
        ttk.Label(form, text="Compute:").grid(row=4, column=0, sticky='w', pady=6)
        ttk.Label(form, text=GPU_INFO, style='Dim.TLabel').grid(
            row=4, column=1, sticky='w', padx=(8, 0), pady=6)

        # Checkboxes
        self.tray_var = tk.BooleanVar(value=self.app.config.get('minimize_to_tray', True))
        ttk.Checkbutton(self, text="Minimize to system tray on close",
                        variable=self.tray_var).pack(anchor='w', padx=14, pady=(16, 4))

        self.autostart_var = tk.BooleanVar(value=self.app.config.get('auto_compute', False))
        ttk.Checkbutton(self, text="Start computing automatically on launch",
                        variable=self.autostart_var).pack(anchor='w', padx=14, pady=4)

        self.boot_var = tk.BooleanVar(value=self.app.config.get('start_on_boot', False))
        ttk.Checkbutton(self, text="Start W@Home when computer starts",
                        variable=self.boot_var).pack(anchor='w', padx=14, pady=4)

        # Buttons
        btn_row = ttk.Frame(self)
        btn_row.pack(fill='x', padx=14, pady=(20, 6))

        ttk.Button(btn_row, text="Save", style='Green.TButton',
                   command=self._save).pack(side='left', padx=(0, 6))
        ttk.Button(btn_row, text="Check for Update",
                   command=self._check_update).pack(side='left', padx=(0, 6))

        self.save_status = ttk.Label(self, text="", style='Dim.TLabel')
        self.save_status.pack(anchor='w', padx=14, pady=4)

        # ── About ──
        sep = ttk.Separator(self, orient='horizontal')
        sep.pack(fill='x', padx=14, pady=(16, 0))

        about = ttk.Frame(self)
        about.pack(fill='x', padx=14, pady=(10, 8))

        ttk.Label(about, text=f"W@Home Hive  v{VERSION}",
                  style='Card.TLabel', font=(MONO, 11)).pack(anchor='w')
        ttk.Label(about, text="Distributed Menger spectral search",
                  style='Dim.TLabel', font=(MONO, 9)).pack(anchor='w', pady=(2, 6))

        info_lines = [
            ("Publisher", "Akataleptos"),
            ("Contact", "obi@akataleptos.com"),
            ("Website", "https://wathome.akataleptos.com"),
        ]
        for label, value in info_lines:
            row = ttk.Frame(about)
            row.pack(fill='x', pady=1)
            ttk.Label(row, text=f"{label}:", style='Dim.TLabel',
                      font=(MONO, 9), width=10).pack(side='left')
            if value.startswith('http'):
                link = ttk.Label(row, text=value, font=(MONO, 9),
                                 foreground=C['cyan'], cursor='hand2')
                link.pack(side='left')
                link.bind('<Button-1>', lambda e, u=value: webbrowser.open(u))
            else:
                ttk.Label(row, text=value, style='Card.TLabel',
                          font=(MONO, 9)).pack(side='left')

    def _save(self):
        cfg = self.app.config
        cfg['name'] = self.name_var.get().strip()
        cfg['server'] = self.server_var.get().strip().rstrip('/')
        cfg['minimize_to_tray'] = self.tray_var.get()
        cfg['auto_compute'] = self.autostart_var.get()
        cfg['start_on_boot'] = self.boot_var.get()
        save_config(cfg)
        self.app.server = cfg.get('server', DEFAULT_SERVER)
        _set_boot_autostart(self.boot_var.get())
        self.save_status.config(text="Saved", style='Green.TLabel')
        self.app.root.after(2000, lambda: self.save_status.config(text=""))

    def _check_update(self, silent=False):
        """Check for updates and server alerts."""
        if not silent:
            self.save_status.config(text="Checking...", style='Dim.TLabel')
            self.update()
        try:
            code, data = api_get(self.app.server, "/version")
            if code == 200:
                # Server alert — show before update dialog
                alert = data.get('alert', '')
                if alert:
                    self.app.compute_tab.log_msg(f"[SERVER] {alert}", 'gold')
                    self.app._set_alert(alert)

                server_ver = data.get('exe_version', '')
                if server_ver and server_ver != VERSION:
                    self.save_status.config(
                        text=f"Update available: v{server_ver}", style='Gold.TLabel')
                    if messagebox.askyesno("Update Available",
                            f"W@Home v{server_ver} is available (you have v{VERSION}).\n\n"
                            f"Download and install?"):
                        self._download_update()
                else:
                    if not silent:
                        self.save_status.config(text="Up to date", style='Green.TLabel')
            else:
                if not silent:
                    self.save_status.config(text="Could not check", style='Red.TLabel')
        except Exception:
            if not silent:
                self.save_status.config(text="Server unreachable", style='Red.TLabel')

    def _download_update(self):
        """Download update — platform-aware."""
        self.save_status.config(text="Downloading...", style='Dim.TLabel')
        self.update()

        if getattr(sys, 'frozen', False):
            # Windows frozen exe — download installer
            self._download_windows_update()
        else:
            # Linux/macOS — update .py files in place
            self._download_py_update()

    def _download_windows_update(self):
        """Download WHome-Setup.exe and launch it."""
        def _dl():
            try:
                url = f"{self.app.server}/WHome-Setup.exe"
                r = requests.get(url, stream=True, timeout=120)
                if r.status_code != 200:
                    self.app.root.after(0, lambda: self.save_status.config(
                        text="Download failed", style='Red.TLabel'))
                    return

                dl_dir = os.path.join(os.environ.get('TEMP', APP_DIR), 'WHome-Update')
                os.makedirs(dl_dir, exist_ok=True)
                dl_path = os.path.join(dl_dir, 'WHome-Setup.exe')

                total = int(r.headers.get('content-length', 0))
                downloaded = 0
                with open(dl_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=256 * 1024):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            pct = downloaded * 100 // total
                            self.app.root.after(0, lambda p=pct: self.save_status.config(
                                text=f"Downloading... {p}%", style='Dim.TLabel'))

                self.app.root.after(0, lambda: self._launch_installer(dl_path))
            except Exception as e:
                self.app.root.after(0, lambda: self.save_status.config(
                    text=f"Download error: {e}", style='Red.TLabel'))

        threading.Thread(target=_dl, daemon=True).start()

    def _download_py_update(self):
        """Download updated .py files and restart (Linux/macOS)."""
        update_files = ['whome_gui.py', 'screensaver.py', 'w_operator.py', 'w_cuda.py', 'client.py']

        def _dl():
            try:
                updated = 0
                for fname in update_files:
                    url = f"{self.app.server}/static/{fname}"
                    r = requests.get(url, timeout=30)
                    if r.status_code == 200:
                        dest = os.path.join(APP_DIR, fname)
                        with open(dest, 'w') as f:
                            f.write(r.text)
                        updated += 1
                        self.app.root.after(0, lambda n=fname: self.save_status.config(
                            text=f"Updated {n}...", style='Dim.TLabel'))

                if updated > 0:
                    self.app.root.after(0, lambda: self._restart_app(updated))
                else:
                    self.app.root.after(0, lambda: self.save_status.config(
                        text="No files updated", style='Red.TLabel'))
            except Exception as e:
                self.app.root.after(0, lambda: self.save_status.config(
                    text=f"Update error: {e}", style='Red.TLabel'))

        threading.Thread(target=_dl, daemon=True).start()

    def _restart_app(self, count):
        """Restart the app after updating files."""
        self.save_status.config(
            text=f"Updated {count} files — restarting...", style='Green.TLabel')
        self.update()
        import subprocess
        # Re-launch with same Python and same args
        subprocess.Popen([sys.executable, os.path.abspath(__file__)] + sys.argv[1:])
        self.app.root.after(1000, self.app.root.destroy)

    def _launch_installer(self, path):
        """Launch the downloaded Windows installer and exit."""
        self.save_status.config(text="Launching installer...", style='Green.TLabel')
        try:
            import subprocess
            subprocess.Popen([path], shell=True)
            self.app.root.after(1000, self.app.root.destroy)
        except Exception as e:
            self.save_status.config(text=f"Launch failed: {e}", style='Red.TLabel')

# ═══════════════════════════════════════════════════════════
# Main Application
# ═══════════════════════════════════════════════════════════

class WHomeApp:
    def __init__(self):
        self.root = tk.Tk(className='whome')
        self.root.title("W@Home Hive")
        self.root.geometry("820x620")
        self.root.minsize(640, 480)
        self.root.configure(bg=C['bg'])

        setup_theme(self.root)

        # Load config
        self.config = load_config()
        self.api_key = self.config.get('api_key', '')
        self.server = self.config.get('server', DEFAULT_SERVER)
        self.worker = None
        self.msg_queue = queue.Queue()
        self.tray = None
        self._pending_alert = None  # unread server alert message
        self._tray_state = 'idle'   # current tray icon state for redraw
        self._compute_start_time = None  # tracks when current job started
        self._session_jobs = 0
        self._session_hours = 0.0
        self._session_disc = 0
        self._scr_state = {
            'job_id': None, 'lambda_val': None, 'stage': '',
            'stage_num': 0, 'jobs_done': 0, 'active': False,
        }

        # Copy config to shared location so screensaver can find it
        self._copy_config_to_shared()

        # First-run setup if needed
        if not self.api_key:
            result = self._run_setup()
            if not result:
                self.root.destroy()
                sys.exit(0)
            self.config = load_config()
            self.api_key = result['api_key']
            self.server = result['server']
            self._copy_config_to_shared()

        # Window icon — Menger pattern for title bar + taskbar
        try:
            if sys.platform == 'win32':
                # On Windows, use the exe's own embedded icon if running as frozen exe
                # Otherwise generate a temp .ico
                ico_path = None
                if getattr(sys, 'frozen', False):
                    ico_path = sys.executable
                if not ico_path or not os.path.exists(ico_path):
                    from PIL import Image, ImageDraw
                    import tempfile
                    def _mk(sz):
                        img = Image.new('RGBA', (sz, sz), (13, 13, 26, 255))
                        d = ImageDraw.Draw(img)
                        c = (96, 232, 255, 220)
                        s = sz // 3
                        for x in range(3):
                            for y in range(3):
                                if x == 1 and y == 1: continue
                                s2 = s // 3
                                if s2 >= 2:
                                    for x2 in range(3):
                                        for y2 in range(3):
                                            if x2 == 1 and y2 == 1: continue
                                            d.rectangle([x*s+x2*s2, y*s+y2*s2, x*s+(x2+1)*s2-1, y*s+(y2+1)*s2-1], fill=c)
                                else:
                                    d.rectangle([x*s, y*s, (x+1)*s-1, (y+1)*s-1], fill=c)
                        return img
                    icons = [_mk(sz) for sz in [16, 32, 48, 64, 128, 256]]
                    ico_path = os.path.join(tempfile.gettempdir(), 'whome_icon.ico')
                    icons[0].save(ico_path, format='ICO', append_images=icons[1:],
                                  sizes=[(sz, sz) for sz in [16, 32, 48, 64, 128, 256]])
                self.root.iconbitmap(default=ico_path)
            else:
                from PIL import Image, ImageTk, ImageDraw
                def _mk(sz):
                    img = Image.new('RGBA', (sz, sz), (13, 13, 26, 255))
                    d = ImageDraw.Draw(img)
                    c = (96, 232, 255, 220)
                    s = sz // 3
                    for x in range(3):
                        for y in range(3):
                            if x == 1 and y == 1: continue
                            s2 = s // 3
                            if s2 >= 2:
                                for x2 in range(3):
                                    for y2 in range(3):
                                        if x2 == 1 and y2 == 1: continue
                                        d.rectangle([x*s+x2*s2, y*s+y2*s2, x*s+(x2+1)*s2-1, y*s+(y2+1)*s2-1], fill=c)
                            else:
                                d.rectangle([x*s, y*s, (x+1)*s-1, (y+1)*s-1], fill=c)
                    return img
                self._icon_photos = [ImageTk.PhotoImage(_mk(sz)) for sz in [16, 32, 48, 64]]
                self.root.iconphoto(True, *self._icon_photos)
        except Exception:
            pass

        self._build_ui()
        self._setup_tray()
        self._start_pollers()

        self.root.protocol('WM_DELETE_WINDOW', self._on_close)

        # Poll compute worker queue
        self.root.after(100, self._poll_queue)

        # Auto-start if configured
        if self.config.get('auto_compute', False):
            self.root.after(500, self.start_compute)

        # Auto-check for updates (silent — only prompts if update found)
        self.root.after(3000, lambda: self.settings_tab._check_update(silent=True))

    def _run_setup(self):
        """First-run setup — build registration form directly in root window."""
        self.root.geometry("420x400+200+150")
        self._setup_result = None

        frame = ttk.Frame(self.root)
        frame.pack(fill='both', expand=True)

        ttk.Label(frame, text="W@HOME HIVE", style='Title.TLabel').pack(pady=(24, 2))
        ttk.Label(frame, text="First-time setup", style='Dim.TLabel').pack(pady=(0, 16))

        form = ttk.Frame(frame)
        form.pack(padx=30, fill='x')

        ttk.Label(form, text="Node name:").grid(row=0, column=0, sticky='w', pady=4)
        name_var = tk.StringVar(value=f"{platform.node()}-{os.getenv('USER', os.getenv('USERNAME', 'worker'))}")
        ttk.Entry(form, textvariable=name_var, width=32).grid(row=0, column=1, pady=4, padx=(8, 0))

        ttk.Label(form, text="Password:").grid(row=1, column=0, sticky='w', pady=4)
        pw_var = tk.StringVar()
        ttk.Entry(form, textvariable=pw_var, show='*', width=32).grid(row=1, column=1, pady=4, padx=(8, 0))

        ttk.Label(form, text="Confirm:").grid(row=2, column=0, sticky='w', pady=4)
        pw2_var = tk.StringVar()
        ttk.Entry(form, textvariable=pw2_var, show='*', width=32).grid(row=2, column=1, pady=4, padx=(8, 0))

        ttk.Label(form, text="Server:").grid(row=3, column=0, sticky='w', pady=4)
        server_var = tk.StringVar(value=DEFAULT_SERVER)
        ttk.Entry(form, textvariable=server_var, width=32).grid(row=3, column=1, pady=4, padx=(8, 0))

        status = ttk.Label(frame, text="", style='Dim.TLabel')
        status.pack(pady=(12, 4))

        def _on_register():
            name = name_var.get().strip()
            pw, pw2 = pw_var.get(), pw2_var.get()
            server = server_var.get().strip().rstrip('/')
            if not name:
                status.config(text="Name required", style='Red.TLabel'); return
            if len(pw) < 4:
                status.config(text="Password min 4 chars", style='Red.TLabel'); return
            if pw != pw2:
                status.config(text="Passwords don't match", style='Red.TLabel'); return
            status.config(text="Registering...", style='Dim.TLabel')
            self.root.update()
            try:
                api_key, worker_id = do_register(server, name, pw, GPU_INFO)
                self._setup_result = {'api_key': api_key, 'worker_id': worker_id,
                                       'name': name, 'server': server}
                frame.destroy()
            except NameTakenError:
                status.config(text="Name taken. Try Login instead.", style='Gold.TLabel')
            except Exception as e:
                status.config(text=str(e)[:50], style='Red.TLabel')

        def _on_login():
            name = name_var.get().strip()
            pw = pw_var.get()
            server = server_var.get().strip().rstrip('/')
            if not name or not pw:
                status.config(text="Name and password required", style='Red.TLabel'); return
            status.config(text="Logging in...", style='Dim.TLabel')
            self.root.update()
            try:
                api_key, worker_id = do_login(server, name, pw)
                self._setup_result = {'api_key': api_key, 'worker_id': worker_id,
                                       'name': name, 'server': server}
                frame.destroy()
            except Exception as e:
                status.config(text=str(e)[:50], style='Red.TLabel')

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(pady=8)
        ttk.Button(btn_frame, text="Register", style='Green.TButton',
                   command=_on_register).pack(side='left', padx=4)
        ttk.Button(btn_frame, text="Login", command=_on_login).pack(side='left', padx=4)

        ttk.Label(frame, text=f"GPU: {GPU_INFO}", style='Dim.TLabel').pack(pady=(8, 4))

        self.root.wait_window(frame)
        self.root.geometry("820x620")
        return self._setup_result

    def _build_ui(self):
        # Status bar (bottom)
        self.statusbar = ttk.Frame(self.root, style='Card.TFrame')
        self.statusbar.pack(side='bottom', fill='x')
        self.status_text = ttk.Label(self.statusbar, text="\u25cf  Connected",
                                     style='Card.TLabel', font=(MONO, 9),
                                     foreground=C['green'])
        self.status_text.pack(side='left', padx=12, pady=6)
        self.session_text = ttk.Label(self.statusbar, text="",
                                      style='Card.TLabel', font=(MONO, 9),
                                      foreground=C['dim'])
        self.session_text.pack(side='right', padx=12, pady=6)

        # Tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True)

        self.compute_tab = ComputeTab(self.notebook, self)
        self.dashboard_tab = DashboardTab(self.notebook, self)
        self.chat_tab = ChatTab(self.notebook, self)
        self.settings_tab = SettingsTab(self.notebook, self)

        self.notebook.add(self.compute_tab, text='  Compute  ')
        self.notebook.add(self.dashboard_tab, text='  Dashboard  ')
        self.notebook.add(self.chat_tab, text='  Chat  ')
        self.notebook.add(self.settings_tab, text='  Settings  ')


    def _setup_tray(self):
        if not HAS_TRAY:
            return
        icon = create_tray_icon('idle')
        menu = pystray.Menu(
            pystray.MenuItem(
                lambda item: f'Alert: {self._pending_alert[:40]}...'
                             if self._pending_alert and len(self._pending_alert) > 40
                             else f'Alert: {self._pending_alert}'
                             if self._pending_alert else 'No alerts',
                self._tray_read_alert,
                visible=lambda item: self._pending_alert is not None,
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem('Show', self._tray_show, default=True),
            pystray.MenuItem('Pause / Resume', self.toggle_pause),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem('Settings', self._tray_settings),
            pystray.MenuItem('Screensaver', self.launch_screensaver),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem('Quit', self._quit),
        )
        self.tray = pystray.Icon('whome', icon, 'W@Home Hive', menu)
        threading.Thread(target=self.tray.run, daemon=True).start()

    def _update_tray_icon(self, state):
        self._tray_state = state
        if self.tray and HAS_TRAY:
            try:
                self.tray.icon = create_tray_icon(state, alert=self._pending_alert is not None)
            except Exception:
                pass

    def _set_alert(self, message):
        """Set a pending alert — shows dot on tray icon and menu item."""
        self._pending_alert = message
        self._update_tray_icon(self._tray_state)
        # Also fire a system notification
        if self.tray:
            try:
                self.tray.notify(message, "W@Home Alert")
            except Exception:
                pass

    def _tray_read_alert(self, icon=None, item=None):
        """User clicked the alert — show it and clear the indicator."""
        msg = self._pending_alert
        self._pending_alert = None
        self._update_tray_icon(self._tray_state)
        if msg:
            self.root.after(0, lambda: messagebox.showinfo("W@Home Alert", msg))

    def _tray_show(self, icon=None, item=None):
        self.root.after(0, self._show_window)

    def _tray_settings(self, icon=None, item=None):
        self.root.after(0, self._show_settings)

    def _show_settings(self):
        self._show_window()
        self.notebook.select(self.settings_tab)

    def _show_window(self):
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    def _on_close(self):
        if HAS_TRAY and self.config.get('minimize_to_tray', True):
            self.root.withdraw()
        else:
            self._quit()

    def _quit(self, icon=None, item=None):
        if self.worker:
            self.worker.stop()
        if self.tray:
            try:
                self.tray.stop()
            except Exception:
                pass
        self.root.after(0, self.root.destroy)

    # ── Compute controls ──

    def start_compute(self):
        if not HAS_OPERATOR:
            messagebox.showerror("Error", "w_operator.py not found. Cannot compute.")
            return
        if self.worker and self.worker.is_alive():
            return
        self.worker = ComputeWorker(self.server, self.api_key, self.msg_queue)
        self.worker.start()
        self.compute_tab.set_computing(True)
        self._update_tray_icon('computing')

    def stop_compute(self):
        if self.worker:
            self.worker.stop()
            self.compute_tab.set_computing(False)
            self.compute_tab.set_status("Stopping...")
            self._update_tray_icon('idle')

    def toggle_pause(self, icon=None, item=None):
        if not self.worker or not self.worker.is_alive():
            return
        if self.worker.paused:
            self.worker.resume()
            self.root.after(0, lambda: self.compute_tab.pause_btn.config(text="\u23f8  Pause"))
            self._update_tray_icon('computing')
        else:
            self.worker.pause()
            self.root.after(0, lambda: self.compute_tab.pause_btn.config(text="\u25b6  Resume"))
            self._update_tray_icon('paused')

    def launch_screensaver(self, icon=None, item=None):
        if getattr(sys, 'frozen', False):
            # Bundled: launch ourselves with --screensaver flag
            try:
                subprocess.Popen([sys.executable, '--screensaver'])
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror(
                    "Error", f"Failed to launch screensaver:\n{e}"))
        else:
            # Running from source: use python + screensaver.py
            script = os.path.join(APP_DIR, 'screensaver.py')
            if os.path.exists(script):
                try:
                    subprocess.Popen([sys.executable, script,
                                      '--server', self.server])
                except Exception as e:
                    self.root.after(0, lambda: messagebox.showerror(
                        "Error", f"Failed to launch screensaver:\n{e}"))

    # ── Queue polling ──

    def _poll_queue(self):
        try:
            while True:
                event, data = self.msg_queue.get_nowait()
                self._handle_event(event, data)
        except queue.Empty:
            pass
        # Live session stats — show running total including current job time
        if self._compute_start_time is not None:
            elapsed = (time.time() - self._compute_start_time) / 3600
            total_h = self._session_hours + elapsed
            jobs_txt = f"{self._session_jobs}" if self._session_jobs else "0"
            self.compute_tab.stats_label.config(
                text=f"Session: {jobs_txt} jobs, {total_h:.2f}h compute"
                     f" (job running {time.time() - self._compute_start_time:.0f}s)")
            self.session_text.config(
                text=f"{jobs_txt} jobs | {total_h:.2f}h | computing...")
        self.root.after(100, self._poll_queue)

    def _shared_dir(self):
        if sys.platform == 'win32':
            return os.path.join(os.environ.get('LOCALAPPDATA', APP_DIR), 'WHome')
        return os.path.join(os.path.expanduser('~'), '.whome')

    def _write_scr_status(self):
        """Write compute state to shared location for screensaver to read."""
        try:
            shared = self._shared_dir()
            os.makedirs(shared, exist_ok=True)
            status_path = os.path.join(shared, 'compute_status.json')
            state = dict(self._scr_state)
            state['pid'] = os.getpid()
            state['source'] = 'gui'
            with open(status_path, 'w') as f:
                json.dump(state, f)
        except Exception:
            pass

    def _copy_config_to_shared(self):
        """Copy worker config to shared location so screensaver can find it."""
        try:
            shared = self._shared_dir()
            os.makedirs(shared, exist_ok=True)
            src = CONFIG_PATH
            dst = os.path.join(shared, 'worker_config.json')
            if os.path.exists(src):
                import shutil
                shutil.copy2(src, dst)
        except Exception:
            pass

    def _handle_event(self, event, data):
        tab = self.compute_tab
        if event == 'status':
            tab.set_status(data.get('text', ''))
            tab.log_msg(data.get('text', ''))
            state = data.get('state', 'idle')
            self.status_text.config(
                text=f"\u25cf  {data.get('text', '')}",
                foreground=C['green'] if state == 'computing' else
                           C['gold'] if state == 'paused' else C['dim'])
            self._scr_state['active'] = (state == 'computing')
            if state != 'computing':
                self._compute_start_time = None  # not actively computing
            if state == 'idle' and self.worker and not self.worker.is_alive():
                tab.set_computing(False)
                tab.reset_stages()
                self._scr_state['active'] = False
        elif event == 'job_start':
            self._update_tray_icon('computing')
            self._compute_start_time = time.time()
            tab.set_job(data['job_id'], data['lambda'], data['params'])
            tab.reset_stages()
            tab.log_msg(f"Job #{data['job_id']} started \u2014 \u03bb={data['lambda']:.6f}")
            self.status_text.config(text=f"\u25cf  Job #{data['job_id']}  \u03bb={data['lambda']:.6f}",
                                    foreground=C['green'])
            self._scr_state.update({
                'job_id': data['job_id'], 'lambda_val': data['lambda'],
                'stage': '', 'stage_num': 0, 'active': True,
            })
        elif event == 'stage':
            tab.set_stage(data['num'])
            detail = data.get('detail', '')
            tab.log_msg(f"  {data['name']}  {detail}", 'info')
            self.status_text.config(text=f"\u25cf  {data['name']}",
                                    foreground=C['green'])
            self._scr_state.update({
                'stage': data['name'], 'stage_num': data['num'],
            })
        elif event == 'job_done':
            self._compute_start_time = None  # job finished, stop live timer
            tab.set_status("Job complete")
            n = data.get('n_eigs', 0)
            dur = data.get('duration', 0)
            if data.get('hits'):
                for h in data['hits']:
                    tab.log_msg(f"  \u2605 RESONANCE: {h}", 'gold')
            else:
                tab.log_msg(f"  Complete \u2014 {n} eigenvalues in {dur:.1f}s", 'green')
            self._session_jobs = data.get('session_jobs', 0)
            self._session_hours = data.get('session_hours', 0)
            self._session_disc = data.get('session_disc', 0)
            tab.update_stats(self._session_jobs, self._session_hours, self._session_disc)
            self.session_text.config(
                text=f"{self._session_jobs} jobs | {self._session_hours:.2f}h")
            self._scr_state['jobs_done'] = self._session_jobs
            self._scr_state['stage'] = 'Complete'
            self._scr_state['stage_num'] = 6
        elif event == 'update_available':
            ver = data.get('version', '?')
            tab.log_msg(f"Update available: v{ver}", 'gold')
            self.settings_tab.save_status.config(
                text=f"Update available: v{ver}", style='Gold.TLabel')
            if self.tray:
                try:
                    self.tray.notify(f"W@Home v{ver} available",
                                     "Click Check for Update in Settings")
                except Exception:
                    pass
        elif event == 'server_alert':
            msg = data.get('message', '')
            if msg:
                tab.log_msg(f"[SERVER] {msg}", 'gold')
                self._set_alert(msg)
        elif event == 'error':
            tab.log_msg(data.get('text', 'Unknown error'), 'error')
            self._update_tray_icon('error')
        # Update screensaver status file
        self._write_scr_status()

    # ── Background pollers ──

    def _start_pollers(self):
        # Dashboard poller
        def poll_dashboard():
            while True:
                try:
                    code, progress = api_get(self.server, "/progress", self.api_key)
                    if code == 200:
                        self.root.after(0, self.dashboard_tab.update_progress, progress)

                    code, disc = api_get(self.server, "/discoveries", self.api_key)
                    if code == 200 and isinstance(disc, list):
                        self.root.after(0, self.dashboard_tab.update_discoveries, disc)

                    code, leaders = api_get(self.server, "/leaderboard", self.api_key)
                    if code == 200 and isinstance(leaders, list):
                        self.root.after(0, self.dashboard_tab.update_leaderboard, leaders)
                except Exception:
                    pass
                time.sleep(15)

        threading.Thread(target=poll_dashboard, daemon=True).start()

        # Chat poller
        def poll_chat():
            while True:
                self.chat_tab.poll_messages()
                time.sleep(3)

        threading.Thread(target=poll_chat, daemon=True).start()

    def run(self):
        self.root.mainloop()

# ═══════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════

def _ensure_single_instance():
    """Prevent multiple GUI instances (would cause duplicate compute workers)."""
    if sys.platform == 'win32':
        import ctypes
        mutex = ctypes.windll.kernel32.CreateMutexW(None, False, "WHome_Hive_Mutex")
        if ctypes.windll.kernel32.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
            # Try to find and foreground the existing window
            hwnd = ctypes.windll.user32.FindWindowW(None, "W@Home Hive")
            if hwnd:
                ctypes.windll.user32.ShowWindow(hwnd, 9)  # SW_RESTORE
                ctypes.windll.user32.SetForegroundWindow(hwnd)
            sys.exit(0)
        return mutex  # Must keep reference so mutex isn't garbage collected
    else:
        lock_path = os.path.join(os.path.expanduser('~'), '.whome', 'gui.lock')
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)
        import fcntl
        lock_file = open(lock_path, 'w')
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock_file.write(str(os.getpid()))
            lock_file.flush()
        except OSError:
            sys.exit(0)
        return lock_file  # Keep reference


def main():
    _instance_lock = _ensure_single_instance()

    # Set low process priority on Windows
    try:
        if sys.platform == 'win32':
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(
                ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL
        else:
            os.nice(10)
    except Exception:
        pass

    app = WHomeApp()
    app.run()

def _is_screensaver_mode():
    """Check if launched as Windows screensaver (.scr with /s, /c, /p flags)."""
    args = [a.lower().lstrip('/').lstrip('-') for a in sys.argv[1:]]
    if 'p' in args:
        sys.exit(0)  # Preview mode — skip
    if 'c' in args:
        # Config dialog
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(
                0, "W@Home Screensaver\n\nConfigure via the W@Home Hive app.",
                "W@Home Settings", 0x40)
        except Exception:
            pass
        sys.exit(0)
    if 's' in args or '--screensaver' in sys.argv[1:]:
        return True
    # Also detect if invoked as .scr without flags (double-click on .scr = /s)
    if getattr(sys, 'frozen', False) and sys.executable.lower().endswith('.scr'):
        return True
    return False


if __name__ == '__main__':
    if _is_screensaver_mode():
        # Launch screensaver
        try:
            scr_dir = os.path.dirname(os.path.abspath(__file__))
            if getattr(sys, 'frozen', False):
                scr_dir = getattr(sys, '_MEIPASS', scr_dir)
            sys.path.insert(0, scr_dir)
            from screensaver import run_screensaver
            run_screensaver(fullscreen=True)
        except Exception as e:
            try:
                import ctypes
                ctypes.windll.user32.MessageBoxW(
                    0, f"Screensaver error:\n{e}", "W@Home", 0x10)
            except Exception:
                print(f"Screensaver error: {e}")
    else:
        try:
            main()
        except Exception as e:
            messagebox.showerror("W@Home Error", str(e))
            if getattr(sys, 'frozen', False):
                input("Press Enter to exit...")
