#!/bin/bash
# W@Home Hive — One-Click Installer (Linux + macOS)
# curl -sSL https://wathome.akataleptos.com/static/install.sh | bash

set -e

# ═══════════════════════════════════════════════
# Colors — use printf for macOS compatibility
# ═══════════════════════════════════════════════
CYAN='\033[96m'
GOLD='\033[93m'
GREEN='\033[92m'
RED='\033[91m'
DIM='\033[90m'
BOLD='\033[1m'
RESET='\033[0m'

say() { printf "$@\n"; }

say ""
say "${CYAN}╔══════════════════════════════════════════════════════════╗"
say "║  ${BOLD}W@HOME HIVE — Installer${RESET}${CYAN}                                ║"
say "║  ${DIM}Akataleptos Distributed Spectral Search${RESET}${CYAN}                 ║"
say "╚══════════════════════════════════════════════════════════╝${RESET}"
say ""

# ═══════════════════════════════════════════════
# OS Detection
# ═══════════════════════════════════════════════
OS="unknown"
ARCH=$(uname -m)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    if command -v apt-get &>/dev/null; then PKG="apt"
    elif command -v dnf &>/dev/null; then PKG="dnf"
    elif command -v pacman &>/dev/null; then PKG="pacman"
    else PKG="unknown"; fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
    PKG="brew"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    OS="windows"
    PKG="none"
fi

say "  ${DIM}OS:${RESET} $OS ($ARCH)"
say "  ${DIM}Package Manager:${RESET} $PKG"
say ""

# ═══════════════════════════════════════════════
# Check Python
# ═══════════════════════════════════════════════
PYTHON=""
for cmd in python3 python; do
    if command -v $cmd &>/dev/null; then
        ver=$($cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
        major=$(echo $ver | cut -d. -f1)
        minor=$(echo $ver | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 8 ]; then
            PYTHON=$cmd
            say "  ${GREEN}✓${RESET} Python: $($cmd --version)"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    say "  ${RED}✗${RESET} Python 3.8+ not found"
    say ""
    if [ "$OS" == "linux" ] && [ "$PKG" == "apt" ]; then
        say "  ${CYAN}Install with:${RESET} sudo apt install python3 python3-pip python3-venv"
    elif [ "$OS" == "macos" ]; then
        say "  ${CYAN}Install with:${RESET} brew install python3"
    fi
    say ""
    read -p "  Install Python now? [Y/n] " yn < /dev/tty
    case $yn in
        [Nn]* ) echo "  Cannot continue without Python."; exit 1;;
        * )
            if [ "$OS" == "linux" ] && [ "$PKG" == "apt" ]; then
                sudo apt-get update && sudo apt-get install -y python3 python3-pip python3-venv
            elif [ "$OS" == "linux" ] && [ "$PKG" == "dnf" ]; then
                sudo dnf install -y python3 python3-pip
            elif [ "$OS" == "linux" ] && [ "$PKG" == "pacman" ]; then
                sudo pacman -S --noconfirm python python-pip
            elif [ "$OS" == "macos" ]; then
                if command -v brew &>/dev/null; then
                    brew install python3
                else
                    say "  ${RED}Homebrew not found.${RESET} Install from https://brew.sh first."
                    exit 1
                fi
            fi
            PYTHON=python3
            ;;
    esac
fi

# ═══════════════════════════════════════════════
# Check tkinter (macOS Homebrew needs separate pkg)
# ═══════════════════════════════════════════════
if ! $PYTHON -c "import tkinter" 2>/dev/null; then
    say "  ${CYAN}Installing tkinter...${RESET}"
    if [ "$OS" == "macos" ]; then
        brew install python-tk@3.$minor 2>/dev/null || brew install python-tk 2>/dev/null || true
    elif [ "$OS" == "linux" ] && [ "$PKG" == "apt" ]; then
        sudo apt-get install -y python3-tk 2>/dev/null || true
    elif [ "$OS" == "linux" ] && [ "$PKG" == "dnf" ]; then
        sudo dnf install -y python3-tkinter 2>/dev/null || true
    elif [ "$OS" == "linux" ] && [ "$PKG" == "pacman" ]; then
        sudo pacman -S --noconfirm tk 2>/dev/null || true
    fi
fi

# ═══════════════════════════════════════════════
# GPU Detection
# ═══════════════════════════════════════════════
GPU_INFO="CPU"
HAS_CUDA=false
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    if [ -n "$GPU_NAME" ]; then
        GPU_INFO="NVIDIA $GPU_NAME"
        HAS_CUDA=true
        say "  ${GREEN}✓${RESET} GPU: $GPU_INFO"
    fi
else
    say "  ${DIM}○${RESET} No NVIDIA GPU detected (CPU mode)"
fi
say ""

# ═══════════════════════════════════════════════
# Install Location
# ═══════════════════════════════════════════════
INSTALL_DIR="$HOME/.whome"
say "  ${DIM}Install directory:${RESET} $INSTALL_DIR"
say ""

if [ -d "$INSTALL_DIR" ]; then
    say "  ${GOLD}Existing installation found.${RESET}"
    read -p "  Update? [Y/n] " yn < /dev/tty
    case $yn in
        [Nn]* ) echo "  Keeping existing installation.";;
        * ) echo "  Updating...";;
    esac
else
    mkdir -p "$INSTALL_DIR"
fi

# ═══════════════════════════════════════════════
# Download / Copy Files
# ═══════════════════════════════════════════════
say ""
say "  ${CYAN}Downloading W@Home...${RESET}"

HIVE_SERVER="${HIVE_SERVER:-https://wathome.akataleptos.com}"

# Try downloading from server, fall back to local copy
for f in whome_gui.py client.py w_operator.py w_cuda.py screensaver.py icon-menger-256.png; do
    if curl -sSf "$HIVE_SERVER/static/$f" -o "$INSTALL_DIR/$f" 2>/dev/null; then
        say "  ${GREEN}✓${RESET} $f"
    elif [ -f "$(dirname "$0")/$f" ]; then
        cp "$(dirname "$0")/$f" "$INSTALL_DIR/$f"
        say "  ${GREEN}✓${RESET} $f (local)"
    else
        say "  ${RED}✗${RESET} $f — not found"
    fi
done

# ═══════════════════════════════════════════════
# Virtual Environment
# ═══════════════════════════════════════════════
say ""
say "  ${CYAN}Setting up Python environment...${RESET}"

if [ ! -d "$INSTALL_DIR/venv" ]; then
    if [ "$OS" == "linux" ]; then
        $PYTHON -m venv --system-site-packages "$INSTALL_DIR/venv"
    else
        $PYTHON -m venv "$INSTALL_DIR/venv"
    fi
elif [ "$OS" == "linux" ] && ! "$INSTALL_DIR/venv/bin/python3" -c "import gi" 2>/dev/null; then
    say "  ${DIM}Recreating venv with system-site-packages for tray support...${RESET}"
    rm -rf "$INSTALL_DIR/venv"
    $PYTHON -m venv --system-site-packages "$INSTALL_DIR/venv"
fi

source "$INSTALL_DIR/venv/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet numpy scipy requests pystray Pillow pygame pyrr

# Linux: system tray deps (gi + AppIndicator)
if [ "$OS" == "linux" ]; then
    NEED_PKGS=""
    $PYTHON -c "import gi" 2>/dev/null || NEED_PKGS="yes"
    if [ -n "$NEED_PKGS" ]; then
        say "  ${CYAN}Installing system tray support...${RESET}"
        if [ "$PKG" == "apt" ]; then
            sudo apt-get install -y python3-gi gir1.2-ayatanaappindicator3-0.1 2>/dev/null || \
            sudo apt-get install -y python3-gi gir1.2-appindicator3-0.1 2>/dev/null || true
        elif [ "$PKG" == "dnf" ]; then
            sudo dnf install -y python3-gobject libappindicator-gtk3 2>/dev/null || true
        elif [ "$PKG" == "pacman" ]; then
            sudo pacman -S --noconfirm python-gobject libappindicator-gtk3 2>/dev/null || true
        fi
    fi
fi

if $HAS_CUDA; then
    say "  ${CYAN}Installing CUDA support...${RESET}"
    pip install --quiet cupy-cuda12x 2>/dev/null || \
    pip install --quiet cupy-cuda11x 2>/dev/null || \
    say "  ${DIM}CuPy install failed — using CPU mode${RESET}"
fi

say "  ${GREEN}✓${RESET} Dependencies installed"

# ═══════════════════════════════════════════════
# Wizard
# ═══════════════════════════════════════════════
say ""
say "${CYAN}════════════════════════════════════════════════${RESET}"
say "  ${BOLD}Setup Wizard${RESET}"
say "${CYAN}════════════════════════════════════════════════${RESET}"
say ""

DEFAULT_NAME="$(hostname)-$(whoami)"
DEFAULT_SERVER="https://wathome.akataleptos.com"
SETUP_AUTOSTART=false

# Check if we have a terminal for interactive prompts
TTY_OK=false
if (echo < /dev/tty) >/dev/null 2>&1; then
    TTY_OK=true
fi

if [ "$TTY_OK" = true ]; then
    # Interactive — ask questions
    read -p "  Node name [$DEFAULT_NAME]: " NODE_NAME < /dev/tty
    NODE_NAME="${NODE_NAME:-$DEFAULT_NAME}"
    say "  ${GREEN}✓${RESET} Name: $NODE_NAME"
    say ""

    read -p "  Hive server [$DEFAULT_SERVER]: " SERVER < /dev/tty
    SERVER="${SERVER:-$DEFAULT_SERVER}"
    say "  ${GREEN}✓${RESET} Server: $SERVER"
    say ""

    read -p "  Start on login? [y/N] " yn < /dev/tty
    case $yn in
        [Yy]* ) SETUP_AUTOSTART=true; say "  ${GREEN}✓${RESET} Autostart: yes";;
        * ) say "  ${DIM}○${RESET} Autostart: no";;
    esac
else
    # Non-interactive — use defaults
    say "  ${DIM}(non-interactive — using defaults)${RESET}"
    NODE_NAME="$DEFAULT_NAME"
    SERVER="$DEFAULT_SERVER"
    say "  ${GREEN}✓${RESET} Name: $NODE_NAME"
    say "  ${GREEN}✓${RESET} Server: $SERVER"
    say "  ${DIM}○${RESET} Autostart: no (run whome-gui to configure)"
fi
say ""

# ═══════════════════════════════════════════════
# Write launcher scripts
# ═══════════════════════════════════════════════
cat > "$INSTALL_DIR/whome" << LAUNCHER
#!/bin/bash
cd "$INSTALL_DIR"
source venv/bin/activate
exec python3 client.py --server "$SERVER" --name "$NODE_NAME" "\$@"
LAUNCHER
chmod +x "$INSTALL_DIR/whome"

cat > "$INSTALL_DIR/whome-gui" << LAUNCHER
#!/bin/bash
cd "$INSTALL_DIR"
exec "$INSTALL_DIR/venv/bin/python3" "$INSTALL_DIR/whome_gui.py" "\$@"
LAUNCHER
chmod +x "$INSTALL_DIR/whome-gui"

cat > "$INSTALL_DIR/whome-screensaver" << LAUNCHER
#!/bin/bash
cd "$INSTALL_DIR"
exec "$INSTALL_DIR/venv/bin/python3" "$INSTALL_DIR/screensaver.py" "\$@"
LAUNCHER
chmod +x "$INSTALL_DIR/whome-screensaver"

# Symlink to PATH
if [ "$OS" == "macos" ]; then
    # macOS: /usr/local/bin is in PATH by default
    LINK_DIR="/usr/local/bin"
    sudo mkdir -p "$LINK_DIR" 2>/dev/null || LINK_DIR="$HOME/.local/bin"
    if [ "$LINK_DIR" == "/usr/local/bin" ]; then
        sudo ln -sf "$INSTALL_DIR/whome" "$LINK_DIR/whome"
        sudo ln -sf "$INSTALL_DIR/whome-gui" "$LINK_DIR/whome-gui"
        sudo ln -sf "$INSTALL_DIR/whome-screensaver" "$LINK_DIR/whome-screensaver"
    else
        mkdir -p "$LINK_DIR"
        ln -sf "$INSTALL_DIR/whome" "$LINK_DIR/whome"
        ln -sf "$INSTALL_DIR/whome-gui" "$LINK_DIR/whome-gui"
        ln -sf "$INSTALL_DIR/whome-screensaver" "$LINK_DIR/whome-screensaver"
        say "  ${GOLD}Note:${RESET} Add $LINK_DIR to your PATH: export PATH=\"\$HOME/.local/bin:\$PATH\""
    fi
else
    LINK_DIR="$HOME/.local/bin"
    mkdir -p "$LINK_DIR"
    ln -sf "$INSTALL_DIR/whome" "$LINK_DIR/whome"
    ln -sf "$INSTALL_DIR/whome-gui" "$LINK_DIR/whome-gui"
    ln -sf "$INSTALL_DIR/whome-screensaver" "$LINK_DIR/whome-screensaver"
fi
say "  ${GREEN}✓${RESET} Added 'whome' and 'whome-gui' to PATH"

# ═══════════════════════════════════════════════
# App integration (desktop entry / .app bundle)
# ═══════════════════════════════════════════════
if [ "$OS" == "linux" ]; then
    APPS_DIR="$HOME/.local/share/applications"
    mkdir -p "$APPS_DIR"
    cat > "$APPS_DIR/whome.desktop" << DESKTOP
[Desktop Entry]
Type=Application
Name=W@Home Hive
Comment=Akataleptos Distributed Spectral Search
Exec=$INSTALL_DIR/whome-gui
Icon=$INSTALL_DIR/icon-menger-256.png
StartupWMClass=whome
Terminal=false
Categories=Science;Math;
Keywords=compute;distributed;eigenvalue;
DESKTOP
    chmod +x "$APPS_DIR/whome.desktop"
    update-desktop-database "$APPS_DIR" 2>/dev/null || true
    say "  ${GREEN}✓${RESET} Added to application menu"
elif [ "$OS" == "macos" ]; then
    BUNDLE="$HOME/Applications/W@Home Hive.app"
    mkdir -p "$BUNDLE/Contents/MacOS" "$BUNDLE/Contents/Resources"
    # Copy icon for bundle
    cp "$INSTALL_DIR/icon-menger-256.png" "$BUNDLE/Contents/Resources/icon.png" 2>/dev/null || true
    cat > "$BUNDLE/Contents/MacOS/W@Home Hive" << MACAPP
#!/bin/bash
cd "$INSTALL_DIR"
exec "$INSTALL_DIR/venv/bin/python3" "$INSTALL_DIR/whome_gui.py" "\$@" 2>"$INSTALL_DIR/gui_error.log"
MACAPP
    chmod +x "$BUNDLE/Contents/MacOS/W@Home Hive"
    cat > "$BUNDLE/Contents/Info.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>W@Home Hive</string>
    <key>CFBundleExecutable</key>
    <string>W@Home Hive</string>
    <key>CFBundleIdentifier</key>
    <string>com.akataleptos.whome</string>
    <key>CFBundleVersion</key>
    <string>1.0.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleIconFile</key>
    <string>icon.png</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSPrincipalClass</key>
    <string>NSApplication</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.13</string>
</dict>
</plist>
PLIST
    say "  ${GREEN}✓${RESET} Created W@Home Hive.app in ~/Applications"
fi

# ═══════════════════════════════════════════════
# Autostart
# ═══════════════════════════════════════════════
if $SETUP_AUTOSTART; then
    if [ "$OS" == "linux" ]; then
        AUTOSTART_DIR="$HOME/.config/autostart"
        mkdir -p "$AUTOSTART_DIR"
        cat > "$AUTOSTART_DIR/whome.desktop" << DESKTOP
[Desktop Entry]
Type=Application
Name=W@Home Hive
Comment=Akataleptos Distributed Spectral Search
Exec=$INSTALL_DIR/whome-gui
Icon=$INSTALL_DIR/icon-menger-256.png
Terminal=false
Hidden=false
X-GNOME-Autostart-enabled=true
DESKTOP
        say "  ${GREEN}✓${RESET} Autostart configured"
    elif [ "$OS" == "macos" ]; then
        AGENT_DIR="$HOME/Library/LaunchAgents"
        mkdir -p "$AGENT_DIR"
        cat > "$AGENT_DIR/com.akataleptos.whome.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.akataleptos.whome</string>
    <key>ProgramArguments</key>
    <array>
        <string>$INSTALL_DIR/whome-gui</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
</dict>
</plist>
PLIST
        say "  ${GREEN}✓${RESET} Autostart configured (LaunchAgent)"
    fi
fi

# ═══════════════════════════════════════════════
# Done
# ═══════════════════════════════════════════════
say ""
say "${GREEN}╔══════════════════════════════════════════════════════════╗"
say "║  ${BOLD}Installation Complete!${RESET}${GREEN}                                  ║"
say "╚══════════════════════════════════════════════════════════╝${RESET}"
say ""
say "  ${CYAN}Launch GUI:${RESET}          whome-gui"
say "  ${CYAN}Headless mode:${RESET}       whome"
say "  ${CYAN}Screensaver mode:${RESET}    whome-screensaver"
if [ "$OS" == "linux" ]; then
    say "  ${CYAN}App menu:${RESET}            Search 'W@Home' in your application launcher"
    say "  ${CYAN}Uninstall:${RESET}           rm -rf $INSTALL_DIR ~/.local/bin/whome*"
elif [ "$OS" == "macos" ]; then
    say "  ${CYAN}App:${RESET}                 ~/Applications/W@Home Hive.app"
    say "  ${CYAN}Uninstall:${RESET}           rm -rf $INSTALL_DIR ~/Applications/W@Home\\ Hive.app"
fi
say ""

if [ "$TTY_OK" = true ]; then
    read -p "  Launch W@Home now? [Y/n] " yn < /dev/tty
    case $yn in
        [Nn]* ) say "  Run ${CYAN}whome-gui${RESET} whenever you're ready.";;
        * ) exec "$INSTALL_DIR/whome-gui";;
    esac
else
    say "  Run ${CYAN}whome-gui${RESET} to launch."
fi
