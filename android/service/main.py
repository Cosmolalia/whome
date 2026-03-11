"""
W@Home Hive — Android Foreground Service
Runs eigenvalue computation in the background with a persistent notification.
Communicates status to the Kivy UI via a shared JSON file.
"""

import os
import sys
import json
import hashlib
import time
import threading
import traceback

# Setup path so we can import w_operator from parent dir
service_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.dirname(service_dir)
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

import numpy as np
import w_operator

# ═══════════════════════════════════════════════════════════
# HTTP helpers (pure urllib — no requests in service)
# ═══════════════════════════════════════════════════════════

import urllib.request
import urllib.error
import ssl

# Service process may not have system CA certs — use unverified context
# (we're only talking to our own server at wathome.akataleptos.com)
_ssl_ctx = ssl.create_default_context()
_ssl_ctx.check_hostname = False
_ssl_ctx.verify_mode = ssl.CERT_NONE


def http_post(url, json_data=None, headers=None, timeout=30):
    data = json.dumps(json_data).encode('utf-8') if json_data else None
    req = urllib.request.Request(url, data=data, method='POST')
    req.add_header('Content-Type', 'application/json')
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_ctx) as resp:
            body = resp.read().decode('utf-8')
            return resp.status, body
    except urllib.error.HTTPError as e:
        body = e.read().decode('utf-8')
        return e.code, body


def http_post_json(url, json_data=None, headers=None, timeout=30):
    code, body = http_post(url, json_data, headers, timeout)
    try:
        return code, json.loads(body)
    except Exception:
        return code, {"raw": body}


# ═══════════════════════════════════════════════════════════
# Sparse eigensolver fallback (same as main app)
# ═══════════════════════════════════════════════════════════

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def solve_spectrum_fallback(L_sparse, M=40, tol=1e-8):
    n = L_sparse.shape[0]
    k = min(M + 1, n)
    if n <= 2000:
        if hasattr(L_sparse, 'toarray'):
            dense = L_sparse.toarray()
        else:
            dense = np.array(L_sparse)
        vals = np.linalg.eigvalsh(dense)
        return np.sort(np.real(vals))[:k]
    else:
        return _lanczos_smallest(L_sparse, k, tol, max_iter=300)


def _lanczos_smallest(A, k, tol=1e-8, max_iter=300):
    n = A.shape[0]
    m = min(max(2 * k + 20, 60), n)
    v = np.random.randn(n).astype(np.float64)
    v = v / np.linalg.norm(v)
    V = np.zeros((n, m), dtype=np.float64)
    alpha = np.zeros(m, dtype=np.float64)
    beta = np.zeros(m, dtype=np.float64)
    V[:, 0] = v
    for j in range(m):
        # Check for stop between Lanczos iterations
        _check_stop()
        w = A.dot(V[:, j]).real
        alpha[j] = np.dot(V[:, j], w)
        if j == 0:
            w = w - alpha[j] * V[:, j]
        else:
            w = w - alpha[j] * V[:, j] - beta[j] * V[:, j - 1]
        for i in range(j + 1):
            w -= np.dot(V[:, i], w) * V[:, i]
        beta_next = np.linalg.norm(w)
        if beta_next < 1e-14:
            m = j + 1
            break
        if j + 1 < m:
            beta[j + 1] = beta_next
            V[:, j + 1] = w / beta_next
    T = np.diag(alpha[:m])
    for i in range(m - 1):
        T[i, i + 1] = beta[i + 1]
        T[i + 1, i] = beta[i + 1]
    eigs = np.linalg.eigvalsh(T)
    return np.sort(eigs)[:k]


# Monkey-patch w_operator if scipy missing
if not HAS_SCIPY:
    class SimpleSparse:
        def __init__(self, dense):
            self.data = np.array(dense, dtype=np.complex128)
            self.shape = self.data.shape
        def dot(self, v):
            return self.data.dot(v)
        def toarray(self):
            return self.data

    _orig_build = w_operator.build_magnetic_laplacian

    def _patched_build(vertices, edges, s, psi_by_id):
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

    w_operator.build_magnetic_laplacian = _patched_build
    w_operator.solve_spectrum = solve_spectrum_fallback


# ═══════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════

CONSTANTS = {
    "phi": 1.6180339887, "e": 2.718281828, "pi": 3.141592653,
    "alpha_inv": 137.035999, "proton_electron": 1836.15267,
    "sqrt2": 1.4142135624, "sqrt3": 1.7320508076, "ln2": 0.6931471806,
}
TOLERANCE = 1e-4
MIN_BACKOFF = 2
MAX_BACKOFF = 120

# Stop signal — checked between computation stages for immediate shutdown
_stop_event = threading.Event()


class StopRequested(Exception):
    """Raised when the user requests immediate stop mid-job."""
    pass


def _check_stop():
    """Raise StopRequested if the stop event is set."""
    if _stop_event.is_set():
        raise StopRequested("Stop requested by user")


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
# Status file — shared with Kivy UI
# ═══════════════════════════════════════════════════════════

def get_status_path():
    try:
        from android.storage import app_storage_path
        return os.path.join(app_storage_path(), "service_status.json")
    except ImportError:
        return os.path.join(app_dir, "service_status.json")


def write_status(status):
    try:
        with open(get_status_path(), 'w') as f:
            json.dump(status, f)
    except Exception:
        pass


def get_config_path():
    try:
        from android.storage import app_storage_path
        return os.path.join(app_storage_path(), "worker_config.json")
    except ImportError:
        return os.path.join(app_dir, "worker_config.json")


def load_config():
    p = get_config_path()
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


# ═══════════════════════════════════════════════════════════
# Android Foreground Notification
# ═══════════════════════════════════════════════════════════

notification_id = 1
channel_id = "wathome_compute"

# Service context — set in main(), used instead of mActivity (which is None in service process)
_service_ctx = None


def setup_notification_channel():
    """Create notification channel (required Android 8+). Uses service context."""
    try:
        from jnius import autoclass
        Context = autoclass('android.content.Context')
        NotificationChannel = autoclass('android.app.NotificationChannel')
        NotificationManager = autoclass('android.app.NotificationManager')

        manager = _service_ctx.getSystemService(Context.NOTIFICATION_SERVICE)
        channel = NotificationChannel(
            channel_id,
            "W@Home Computing",
            NotificationManager.IMPORTANCE_LOW  # No sound, just persistent icon
        )
        channel.setDescription("Eigenvalue computation in progress")
        channel.setShowBadge(True)
        manager.createNotificationChannel(channel)
        return manager
    except Exception as e:
        write_status({"state": "init", "message": f"Channel setup failed: {e}"})
        return None


# Cache autoclass refs so we don't re-resolve every update
_notif_classes = {}

def _get_notif_classes():
    if not _notif_classes:
        from jnius import autoclass
        _notif_classes['builder_cls'] = autoclass('android.app.Notification$Builder')
        _notif_classes['bigtext_cls'] = autoclass('android.app.Notification$BigTextStyle')
        _notif_classes['pending_cls'] = autoclass('android.app.PendingIntent')
        _notif_classes['intent_cls'] = autoclass('android.content.Intent')
        _notif_classes['string_cls'] = autoclass('java.lang.String')
        _notif_classes['notification_cls'] = autoclass('android.app.Notification')
    return _notif_classes


def build_notification(text="Computing...", jobs=0, hits=0, hours=0.0, stage=""):
    """Build a rich foreground notification with lock screen + expanded view."""
    try:
        c = _get_notif_classes()
        ctx = _service_ctx
        String = c['string_cls']

        # Intent to reopen app on tap
        intent = c['intent_cls']()
        intent.setClassName(
            ctx.getPackageName(),
            'org.kivy.android.PythonActivity'
        )
        intent.setFlags(c['intent_cls'].FLAG_ACTIVITY_SINGLE_TOP)
        pending = c['pending_cls'].getActivity(
            ctx, 0, intent,
            c['pending_cls'].FLAG_UPDATE_CURRENT | c['pending_cls'].FLAG_IMMUTABLE
        )

        # Short text for collapsed notification + taskbar
        short_text = text

        # Expanded text for notification shade pull-down
        expanded = f"{text}\n"
        if stage:
            expanded += f"Stage: {stage}\n"
        expanded += f"Session: {jobs} jobs \u2022 {hits} hits \u2022 {hours:.1f}h compute"

        builder = c['builder_cls'](ctx, channel_id)
        builder.setContentTitle(String("W@Home Hive"))
        builder.setContentText(String(short_text))
        # Use custom Menger L1 icon, fall back to app icon
        try:
            from jnius import autoclass
            R = autoclass(ctx.getPackageName() + '.R$drawable')
            builder.setSmallIcon(R.ic_menger)
        except Exception:
            builder.setSmallIcon(ctx.getApplicationInfo().icon)
        builder.setContentIntent(pending)
        builder.setOngoing(True)

        # Badge count on app icon = jobs done
        builder.setNumber(jobs)

        # Lock screen visibility — show full stats
        builder.setVisibility(c['notification_cls'].VISIBILITY_PUBLIC)

        # Accent color — cyan (#00d4ff)
        # 0xFF00D4FF as signed 32-bit int for Java
        builder.setColor(-16063233)
        builder.setColorized(True)

        # Expanded view when pulling down shade
        big_style = c['bigtext_cls']()
        big_style.bigText(String(expanded))
        big_style.setBigContentTitle(String(f"W@Home \u2022 {jobs} jobs"))
        builder.setStyle(big_style)

        return builder.build()
    except Exception as e:
        write_status({"state": "init", "message": f"build_notification failed: {e}"})
        return None


def update_notification(manager, text, jobs=0, hits=0, hours=0.0, stage=""):
    """Update the notification with current stats."""
    try:
        notif = build_notification(text, jobs, hits, hours, stage)
        if notif and manager:
            manager.notify(notification_id, notif)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════
# Compute loop
# ═══════════════════════════════════════════════════════════

def run_job(params):
    k = params['k']
    G1, G2 = params['G1'], params['G2']
    S = params['S']
    lam = params['lambda']
    w_glue = params['w_glue']

    _check_stop()
    vertices, edges, b_ids = w_operator.build_graph(k, G1, G2, S, 2)
    _check_stop()
    upd, psi_by = w_operator.add_glue_edges(vertices, b_ids, lam, w_glue, G1, G2)
    _check_stop()
    edges_merged = w_operator.merge_edges(edges, upd)
    _check_stop()
    L, _ = w_operator.build_magnetic_laplacian(vertices, edges_merged, s=(0, 0), psi_by_id=psi_by)
    _check_stop()

    if HAS_SCIPY:
        eigs = w_operator.solve_spectrum(L, M=40)
    else:
        eigs = solve_spectrum_fallback(L, M=40)
    return eigs


def is_charging():
    """Check if device is plugged in using sticky broadcast (reliable on all devices)."""
    try:
        from jnius import autoclass
        Intent = autoclass('android.content.Intent')
        IntentFilter = autoclass('android.content.IntentFilter')
        BatteryManager = autoclass('android.os.BatteryManager')
        ifilter = IntentFilter(Intent.ACTION_BATTERY_CHANGED)
        battery = _service_ctx.registerReceiver(None, ifilter)
        if battery is None:
            return True
        # EXTRA_PLUGGED: 0=unplugged, 1=AC, 2=USB, 4=wireless
        plugged = battery.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0)
        return plugged > 0
    except Exception:
        return True  # Assume charging on non-Android / error


def vibrate(ms=500):
    """Vibrate the device for ms milliseconds."""
    try:
        from jnius import autoclass
        Context = autoclass('android.content.Context')
        vibrator = _service_ctx.getSystemService(Context.VIBRATOR_SERVICE)
        vibrator.vibrate(ms)
    except Exception:
        pass


def compute_loop(cfg, manager):
    api_key = cfg['api_key']
    server = cfg.get('server', 'https://wathome.akataleptos.com')
    backoff = MIN_BACKOFF
    # Restore stats from previous session
    jobs_done = cfg.get('session_jobs', 0)
    discoveries = cfg.get('session_discoveries', 0)
    compute_hours = cfg.get('session_hours', 0.0)
    stop_path = get_status_path().replace('service_status.json', 'service_stop')

    while True:
        # Check for stop signal (file or event)
        if _stop_event.is_set() or os.path.exists(stop_path):
            if os.path.exists(stop_path):
                os.remove(stop_path)
            break

        # Re-read settings each loop (user can change via Settings screen)
        settings = load_config()
        charge_only = settings.get('charge_only', False)
        max_jobs = settings.get('max_jobs', 0)
        cooldown = settings.get('cooldown', 0)
        notify_hits = settings.get('notify_hits', True)

        # Charge-only check
        if charge_only and not is_charging():
            write_status({
                "state": "waiting",
                "jobs_done": jobs_done,
                "discoveries": discoveries,
                "compute_hours": round(compute_hours, 3),
                "message": "Waiting for charger...",
            })
            update_notification(manager, "Paused — waiting for charger",
                               jobs=jobs_done, hits=discoveries, hours=compute_hours)
            time.sleep(30)
            continue

        # Max jobs check
        if max_jobs > 0 and jobs_done >= max_jobs:
            write_status({
                "state": "stopped",
                "jobs_done": jobs_done,
                "discoveries": discoveries,
                "compute_hours": round(compute_hours, 3),
                "message": f"Max jobs reached ({max_jobs})",
            })
            update_notification(manager, f"Done — {jobs_done}/{max_jobs} jobs",
                               jobs=jobs_done, hits=discoveries, hours=compute_hours)
            break

        try:
            write_status({
                "state": "requesting",
                "jobs_done": jobs_done,
                "discoveries": discoveries,
                "compute_hours": round(compute_hours, 3),
            })
            update_notification(manager, f"Requesting job...",
                               jobs=jobs_done, hits=discoveries, hours=compute_hours)

            code, data = http_post_json(
                f"{server}/job",
                headers={"x-api-key": api_key, "x-device-type": "mobile"},
                timeout=30,
            )

            if code == 401:
                write_status({"state": "error", "message": "API key rejected"})
                break

            if code != 200:
                time.sleep(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF)
                continue

            if data.get('status') == 'no_jobs':
                write_status({
                    "state": "waiting",
                    "jobs_done": jobs_done,
                    "discoveries": discoveries,
                    "compute_hours": round(compute_hours, 3),
                })
                update_notification(manager, f"Waiting for new jobs...",
                                   jobs=jobs_done, hits=discoveries, hours=compute_hours)
                time.sleep(60)
                continue

            backoff = MIN_BACKOFF
            job_id = data['job_id']
            params = data['params']
            params['job_id'] = job_id
            lam = params.get('lambda', 0)
            k_val = params.get('k', '?')

            write_status({
                "state": "computing",
                "job_id": job_id,
                "lambda": lam,
                "k": k_val,
                "jobs_done": jobs_done,
                "discoveries": discoveries,
                "compute_hours": round(compute_hours, 3),
            })
            update_notification(manager, f"Computing job {job_id}",
                               jobs=jobs_done, hits=discoveries, hours=compute_hours,
                               stage=f"\u03bb={lam:.4f} k={k_val}")

            start_time = time.time()
            try:
                eigs = run_job(params)
            except StopRequested:
                write_status({"state": "stopped", "jobs_done": jobs_done,
                              "discoveries": discoveries, "message": "Stopped mid-job"})
                update_notification(manager, f"Stopped. {jobs_done} jobs completed.",
                                   jobs=jobs_done, hits=discoveries, hours=compute_hours)
                break
            duration = time.time() - start_time

            hits = check_for_gold(eigs)
            jobs_done += 1
            compute_hours += duration / 3600

            if hits:
                discoveries += len(hits)

            # Submit
            eig_hash = hash_eigenvalues(eigs.tolist())
            code2, resp2 = http_post_json(
                f"{server}/result",
                headers={"x-api-key": api_key},
                json_data={
                    "job_id": job_id,
                    "eigenvalues": eigs.tolist(),
                    "eigenvalues_hash": eig_hash,
                    "found_constants": hits,
                    "compute_seconds": duration,
                },
                timeout=30,
            )

            verified = resp2.get('verified', False) if code2 == 200 else False

            write_status({
                "state": "submitted",
                "job_id": job_id,
                "duration": round(duration, 1),
                "hits": len(hits),
                "verified": verified,
                "jobs_done": jobs_done,
                "discoveries": discoveries,
                "compute_hours": round(compute_hours, 3),
            })

            if hits:
                update_notification(manager, f"\u2728 HIT! {hits[0][:30]}",
                                   jobs=jobs_done, hits=discoveries, hours=compute_hours)
                if notify_hits:
                    vibrate(800)
            else:
                update_notification(manager, f"Job {job_id} done ({round(duration, 1)}s)",
                                   jobs=jobs_done, hits=discoveries, hours=compute_hours)

            # Persist stats to config (app reads on restart)
            try:
                scfg = load_config()
                scfg['session_jobs'] = jobs_done
                scfg['session_discoveries'] = discoveries
                scfg['session_hours'] = round(compute_hours, 4)
                p = get_config_path()
                with open(p, 'w') as f:
                    json.dump(scfg, f, indent=2)
            except Exception:
                pass

            # Cooldown between jobs
            if cooldown > 0:
                time.sleep(cooldown)

        except Exception as e:
            write_status({
                "state": "error",
                "message": str(e)[:200],
                "jobs_done": jobs_done,
                "discoveries": discoveries,
            })
            time.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF)

    write_status({"state": "stopped", "jobs_done": jobs_done, "discoveries": discoveries})
    update_notification(manager, f"Stopped. {jobs_done} jobs completed.")


# ═══════════════════════════════════════════════════════════
# Heartbeat (separate thread)
# ═══════════════════════════════════════════════════════════

def heartbeat_loop(cfg):
    api_key = cfg['api_key']
    server = cfg.get('server', 'https://wathome.akataleptos.com')
    while True:
        try:
            http_post(
                f"{server}/heartbeat",
                headers={"x-api-key": api_key},
                timeout=10,
            )
        except Exception:
            pass
        time.sleep(120)


# ═══════════════════════════════════════════════════════════
# Entry point — called by p4a PythonService
# ═══════════════════════════════════════════════════════════

def main():
    global _service_ctx

    write_status({"state": "init", "message": "Service main() entered"})

    # Get service context FIRST — mActivity is None in the service process,
    # we must use PythonService.mService as our Android context
    try:
        from android.config import SERVICE_CLASS_NAME
        from jnius import autoclass
        service_cls = autoclass(SERVICE_CLASS_NAME)
        _service_ctx = service_cls.mService
        write_status({"state": "init", "message": f"Service context acquired: {SERVICE_CLASS_NAME}"})
    except Exception as e:
        write_status({"state": "error", "message": f"Cannot get service context: {e}"})
        return

    # Now create notification channel (uses _service_ctx)
    manager = setup_notification_channel()
    write_status({"state": "init", "message": f"Notification channel: {'OK' if manager else 'FAILED'}"})

    # Start as foreground service
    try:
        notif = build_notification("Starting up...")
        if notif:
            service_cls.setAutoRestartService(False)
            _service_ctx.startForeground(notification_id, notif)
            write_status({"state": "init", "message": "Foreground started with notification"})
        else:
            write_status({"state": "init", "message": "Notification build returned None"})
    except Exception as e:
        write_status({"state": "init", "message": f"Foreground setup failed: {e}"})

    # Load config
    cfg = load_config()
    if not cfg.get('api_key'):
        write_status({"state": "error", "message": "No API key. Log in from the app first."})
        return

    write_status({"state": "init", "message": f"Config loaded, server={cfg.get('server', '?')}"})

    # Clear any stale stop signal
    _stop_event.clear()
    stop_path = get_status_path().replace('service_status.json', 'service_stop')
    if os.path.exists(stop_path):
        os.remove(stop_path)

    # Start stop-file watcher — polls every 0.5s so STOP responds fast
    def _watch_stop_file():
        while not _stop_event.is_set():
            if os.path.exists(stop_path):
                _stop_event.set()
                try:
                    os.remove(stop_path)
                except Exception:
                    pass
                break
            time.sleep(0.5)
    sw = threading.Thread(target=_watch_stop_file, daemon=True)
    sw.start()

    write_status({"state": "starting", "message": "Starting compute loop"})

    # Start heartbeat thread
    hb = threading.Thread(target=heartbeat_loop, args=(cfg,), daemon=True)
    hb.start()

    # Run compute loop (blocks)
    try:
        compute_loop(cfg, manager)
    except Exception as e:
        write_status({"state": "error", "message": f"compute_loop crashed: {e}"})


if __name__ == '__main__':
    main()
