"""
Build script for W@Home Windows installer.

Usage:
    python build_windows.py          # Build exe only (PyInstaller)
    python build_windows.py --inno   # Build exe + installer (requires Inno Setup)

Prerequisites:
    pip install pyinstaller pystray Pillow requests numpy scipy pygame moderngl pyrr
    Optional: pip install cupy-cuda12x  (GPU support)
    Optional: Install Inno Setup 6 (for --inno flag)
"""

import os
import sys
import shutil
import subprocess

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DIST_DIR = os.path.join(APP_DIR, 'dist')

def create_icon():
    """Generate whome.ico from Menger pattern."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("  [--] Pillow not installed, skipping icon generation")
        return None

    sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    images = []
    for w, h in sizes:
        img = Image.new('RGBA', (w, h), (13, 13, 26, 255))
        draw = ImageDraw.Draw(img)
        s = w // 3
        color = (96, 232, 255, 220)
        for x in range(3):
            for y in range(3):
                if x == 1 and y == 1:
                    continue
                draw.rectangle([x*s+1, y*s+1, (x+1)*s-2, (y+1)*s-2], fill=color)
        images.append(img)

    ico_path = os.path.join(APP_DIR, 'whome.ico')
    images[0].save(ico_path, format='ICO', sizes=[(s[0], s[1]) for s in sizes],
                   append_images=images[1:])
    print(f"  [OK] Icon: {ico_path}")
    return ico_path


def build_exe():
    """Build standalone .exe with PyInstaller — includes screensaver."""
    print("\n  Building W@Home executable...\n")

    ico = create_icon()

    cmd = [
        sys.executable, '-m', 'PyInstaller',
        '--onefile', '--windowed',
        '--name=WHome',
        '--add-data', f'w_operator.py{os.pathsep}.',
        '--add-data', f'w_cuda.py{os.pathsep}.',
        '--add-data', f'screensaver.py{os.pathsep}.',
    ]

    # Add icon if available
    icon_path = os.path.join(APP_DIR, 'icon-192.png')
    if os.path.exists(icon_path):
        cmd.extend(['--add-data', f'icon-192.png{os.pathsep}.'])

    if ico:
        cmd.extend(['--icon', ico])

    # Hidden imports — includes screensaver deps
    cmd.extend([
        '--hidden-import=pystray',
        '--hidden-import=PIL',
        '--hidden-import=PIL.Image',
        '--hidden-import=PIL.ImageDraw',
        '--hidden-import=pygame',
        '--hidden-import=moderngl',
        '--hidden-import=pyrr',
        '--hidden-import=pyrr.Matrix44',
        '--hidden-import=pyrr.Vector3',
    ])

    # Exclude heavy packages that get pulled in from global env
    for mod in ['torch', 'torchvision', 'tensorflow', 'qiskit',
                'Panda3D', 'panda3d', 'ursina', 'sympy',
                'symengine', 'matplotlib', 'flask', 'jinja2',
                'pywebview', 'beautifulsoup4', 'bs4', 'cryptography',
                'GitPython', 'git', 'rustworkx']:
        cmd.extend(['--exclude-module', mod])

    cmd.append('whome_gui.py')

    print(f"  Running: {' '.join(cmd[-3:])}")
    result = subprocess.run(cmd, cwd=APP_DIR)

    if result.returncode != 0:
        print("\n  [!!] PyInstaller build failed")
        sys.exit(1)

    exe_path = os.path.join(DIST_DIR, 'WHome.exe')
    if os.path.exists(exe_path):
        size_mb = os.path.getsize(exe_path) / (1024 * 1024)
        print(f"\n  [OK] Built: {exe_path} ({size_mb:.1f} MB)")
    else:
        print("\n  [!!] Expected output not found")
        sys.exit(1)

    # Copy exe as .scr (Windows screensavers are just renamed exes)
    scr_path = os.path.join(DIST_DIR, 'WHome.scr')
    shutil.copy2(exe_path, scr_path)
    print(f"  [OK] Copied: {scr_path}")

    # Copy supporting files to dist/ for Inno Setup
    for f in ['w_operator.py', 'w_cuda.py', 'screensaver.py', 'icon-192.png']:
        src = os.path.join(APP_DIR, f)
        if os.path.exists(src):
            shutil.copy2(src, DIST_DIR)

    return exe_path


def build_installer():
    """Build Windows installer using Inno Setup."""
    iss_path = os.path.join(APP_DIR, 'installer.iss')
    if not os.path.exists(iss_path):
        print("  [!!] installer.iss not found")
        return

    # Find Inno Setup compiler
    iscc = None
    for path in [
        r'C:\Program Files (x86)\Inno Setup 6\ISCC.exe',
        r'C:\Program Files\Inno Setup 6\ISCC.exe',
        'ISCC.exe',  # On PATH
    ]:
        if os.path.exists(path) or shutil.which(path):
            iscc = path
            break

    if not iscc:
        print("  [!!] Inno Setup not found. Install from https://jrsoftware.org/issetup.php")
        print("  [--] Exe is still available in dist/WHome.exe")
        return

    print(f"\n  Building installer with {iscc}...")
    result = subprocess.run([iscc, iss_path], cwd=APP_DIR)
    if result.returncode == 0:
        print("\n  [OK] Installer built: dist/WHome-Setup.exe")
    else:
        print("\n  [!!] Inno Setup build failed")


if __name__ == '__main__':
    print()
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║  W@HOME HIVE — Windows Build                           ║")
    print("  ╚══════════════════════════════════════════════════════════╝")

    build_exe()

    if '--inno' in sys.argv:
        build_installer()

    print()
    print("  Done. To distribute:")
    print(f"    Standalone:  dist/WHome.exe")
    print(f"    Screensaver: dist/WHome.scr")
    if '--inno' in sys.argv:
        print(f"    Installer:   dist/WHome-Setup.exe")
    print()
