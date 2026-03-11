"""
W@Home Screensaver — Windows .scr wrapper

Windows screensavers are .exe files renamed to .scr that respond to:
  /s         — run screensaver fullscreen
  /c         — show settings dialog
  /p HWND    — preview in settings thumbnail (we skip this)

Build:
  pyinstaller --onefile --windowed --name=WHome whome_scr.py
  rename dist\WHome.exe WHome.scr
  copy WHome.scr %WINDIR%\System32\
  Then: Settings > Personalization > Lock screen > Screen saver settings > select "WHome"
"""

import sys
import os

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)


def get_mode():
    """Parse Windows screensaver command-line flags."""
    for arg in sys.argv[1:]:
        a = arg.lower().lstrip('/-')
        if a.startswith('s'):
            return 'screensaver'
        elif a.startswith('c'):
            return 'configure'
        elif a.startswith('p'):
            return 'preview'
    # No args = run screensaver (double-click)
    return 'screensaver'


def run_screensaver():
    """Launch the Menger sponge screensaver fullscreen."""
    try:
        from screensaver import run_screensaver as _run
        # Load server URL from config if available
        server = "https://wathome.akataleptos.com"
        api_key = None
        config_path = os.path.join(APP_DIR, "worker_config.json")
        if os.path.exists(config_path):
            import json
            with open(config_path) as f:
                cfg = json.load(f)
            server = cfg.get('server', server)
            api_key = cfg.get('api_key')
        _run(api_key=api_key, server_url=server, fullscreen=True)
    except Exception as e:
        # Log errors since we have no console
        try:
            with open(os.path.join(APP_DIR, 'scr_error.log'), 'w') as f:
                import traceback
                traceback.print_exc(file=f)
        except Exception:
            pass


def show_settings():
    """Show a simple settings dialog."""
    try:
        import tkinter as tk
        from tkinter import ttk
        root = tk.Tk()
        root.title("W@Home Screensaver")
        root.geometry("300x150+200+200")
        root.configure(bg='#0d0d1a')
        root.resizable(False, False)
        tk.Label(root, text="W@Home Hive", fg='#60e8ff', bg='#0d0d1a',
                 font=('Consolas', 14, 'bold')).pack(pady=(20, 4))
        tk.Label(root, text="Menger Sponge Screensaver", fg='#606878', bg='#0d0d1a',
                 font=('Consolas', 10)).pack()
        tk.Label(root, text="akataleptos.com", fg='#606878', bg='#0d0d1a',
                 font=('Consolas', 9)).pack(pady=(8, 0))
        tk.Button(root, text="OK", command=root.destroy, width=10).pack(pady=12)
        root.mainloop()
    except Exception:
        pass


def main():
    mode = get_mode()
    if mode == 'screensaver':
        run_screensaver()
    elif mode == 'configure':
        show_settings()
    # 'preview' — do nothing (thumbnail preview not supported)


if __name__ == '__main__':
    main()
