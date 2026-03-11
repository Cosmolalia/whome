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
    python client.py --screensaver           # With live computation
    python screensaver.py --fullscreen       # Fullscreen mode
"""

import pygame
import numpy as np
import math
import sys
import threading
import time
import json
import os

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
    """Background thread that polls server for status."""
    def __init__(self, server_url, api_key=None):
        self.server_url = server_url
        self.api_key = api_key
        self.progress = {}
        self.discoveries = []
        self.leaderboard = []
        self.lock = threading.Lock()
        self._running = True

        self.thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.thread.start()

    def _poll_loop(self):
        while self._running:
            try:
                headers = {"x-api-key": self.api_key} if self.api_key else {}
                prog = requests.get(f"{self.server_url}/progress", timeout=5).json()
                disc = requests.get(f"{self.server_url}/discoveries", timeout=5).json()

                with self.lock:
                    self.progress = prog
                    self.discoveries = disc[:5]
            except Exception:
                pass
            time.sleep(5)

    def get(self):
        with self.lock:
            return dict(self.progress), list(self.discoveries)

    def stop(self):
        self._running = False

# ═══════════════════════════════════════════════════════════
# Live Computation State (shared with client worker thread)
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


def run_screensaver(api_key=None, server_url="http://localhost:8081",
                    worker_id=None, compute_state=None, fullscreen=False):
    """Main screensaver loop."""
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

        # Title
        s = fonts['title'].render("W@HOME HIVE", True, (160, 240, 255))
        hud.blit(s, (30, 25))
        s = fonts['info'].render("Akataleptos Distributed Spectral Search", True, (80, 80, 100))
        hud.blit(s, (30, 58))

        # Get server stats
        progress, discoveries = ({}, [])
        if poller:
            progress, discoveries = poller.get()

        # Sweep progress bar
        if progress:
            pct = progress.get('percent_complete', 0)
            bar_x, bar_y, bar_w, bar_h = 30, H - 80, W - 60, 20
            pygame.draw.rect(hud, (30, 30, 45), (bar_x, bar_y, bar_w, bar_h), border_radius=3)
            fill_w = int(bar_w * pct / 100)
            if fill_w > 0:
                # Gradient fill
                for px in range(fill_w):
                    frac = px / bar_w
                    r = int(196 * (1 - frac) + 160 * frac)
                    g = int(160 * (1 - frac) + 240 * frac)
                    b = int(255 * (1 - frac) + 255 * frac)
                    pygame.draw.line(hud, (r, g, b, 200), (bar_x + px, bar_y), (bar_x + px, bar_y + bar_h))

            # Lambda marker
            current_lam = progress.get('current_lambda', 0.4)
            marker_x = bar_x + int(bar_w * (current_lam - 0.4) / 0.2)
            pygame.draw.line(hud, (255, 208, 106), (marker_x, bar_y - 3), (marker_x, bar_y + bar_h + 3), 2)

            # Labels
            s = fonts['info'].render(f"λ = 0.400000", True, (70, 70, 90))
            hud.blit(s, (bar_x, bar_y + bar_h + 5))
            s = fonts['info'].render(f"{pct:.2f}%", True, (160, 240, 255))
            hud.blit(s, (bar_x + bar_w // 2 - s.get_width() // 2, bar_y + bar_h + 5))
            s = fonts['info'].render(f"λ = 0.600000", True, (70, 70, 90))
            hud.blit(s, (bar_x + bar_w - s.get_width(), bar_y + bar_h + 5))

            # Stats line
            workers = progress.get('active_workers', 0)
            completed = progress.get('completed', 0)
            total = progress.get('total_jobs', 200000)
            total_disc = progress.get('total_discoveries', 0)
            eta = progress.get('eta_hours')

            stats_y = H - 115
            stats = (f"{workers} workers online  |  {completed:,}/{total:,} jobs  |  "
                     f"{total_disc} discoveries")
            if eta:
                stats += f"  |  ETA: {eta}h"
            s = fonts['info'].render(stats, True, (80, 80, 100))
            hud.blit(s, (30, stats_y))

        # Current computation (if worker is running, or via file IPC)
        comp = compute_state.get() if compute_state else _read_status_file()
        if comp and comp.get('lambda_val') is not None:
            # Lambda value — big display
            lam_str = f"λ = {comp['lambda_val']:.6f}"
            s = fonts['lambda'].render(lam_str, True, (255, 208, 106))
            hud.blit(s, (W - s.get_width() - 30, 30))

            # Job info
            s = fonts['info'].render(f"Job #{comp['job_id']}", True, (80, 80, 100))
            hud.blit(s, (W - s.get_width() - 30, 60))

            # Stage progress
            stages = ['Graph', 'Glue', 'Merge', 'Laplacian', 'Spectrum']
            stage_y = 90
            for i, name in enumerate(stages):
                if i + 1 < comp.get('stage_num', 0):
                    color = (128, 255, 170)  # green
                    marker = "●"
                elif i + 1 == comp.get('stage_num', 0):
                    color = (160, 240, 255)  # cyan
                    marker = "◉"
                else:
                    color = (55, 55, 65)
                    marker = "○"
                s = fonts['stage'].render(f" {marker} {name}", True, color)
                hud.blit(s, (W - 180, stage_y + i * 22))

            # Eigenvalue spectrum visualization
            eigs = comp.get('eigenvalues')
            if eigs is not None and len(eigs) > 0:
                eigs_arr = np.array(eigs)
                eigs_nz = eigs_arr[eigs_arr > 1e-9]
                if len(eigs_nz) > 0:
                    spec_x = W - 250
                    spec_y = 250
                    spec_h = 200
                    max_eig = eigs_nz.max()
                    s = fonts['info'].render("Eigenvalue Spectrum", True, (70, 70, 90))
                    hud.blit(s, (spec_x, spec_y - 20))
                    for i, e in enumerate(eigs_nz[:40]):
                        bar_h = int((e / max_eig) * spec_h)
                        bx = spec_x + i * 5
                        # Color by position
                        frac = i / max(len(eigs_nz) - 1, 1)
                        r = int(160 * (1 - frac) + 255 * frac)
                        g = int(240 * (1 - frac) + 208 * frac)
                        b = int(255 * (1 - frac) + 106 * frac)
                        pygame.draw.rect(hud, (r, g, b, 180),
                                        (bx, spec_y + spec_h - bar_h, 3, bar_h))

        # Discovery feed (bottom-right)
        if discoveries:
            disc_x = W - 350
            disc_y = H - 200
            s = fonts['info'].render("Recent Discoveries", True, (255, 208, 106))
            hud.blit(s, (disc_x, disc_y - 20))
            for i, d in enumerate(discoveries[:4]):
                name = d.get('constant_name', '?')
                lam = d.get('lambda_val', 0)
                ratio = d.get('ratio_value', 0)
                s = fonts['disc'].render(
                    f"  {name} at λ={lam:.6f} (r={ratio:.5f})",
                    True, (196, 160, 255)
                )
                hud.blit(s, (disc_x, disc_y + i * 18))

        # Axiom
        a = int(128 + 40 * math.sin(t * 0.3))
        s = fonts['info'].render("1 = 0 = ∞", True, (a // 2, a // 2, a))
        hud.blit(s, (W // 2 - s.get_width() // 2, H - 30))

        # Upload HUD
        data = pygame.image.tostring(hud, 'RGBA', True)
        hud_tex.write(data)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        hud_tex.use()
        hud_vao.render(moderngl.TRIANGLE_STRIP)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE

        pygame.display.flip()

    if poller:
        poller.stop()
    pygame.quit()

# ═══════════════════════════════════════════════════════════
# Standalone entry
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://localhost:8081")
    parser.add_argument("--fullscreen", action="store_true")
    args = parser.parse_args()

    run_screensaver(server_url=args.server, fullscreen=args.fullscreen)
