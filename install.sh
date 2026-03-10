#!/bin/bash
# W@Home Hive — One-Click Installer
# curl -sSL https://akataleptos.com/hive/install.sh | bash

set -e

# ═══════════════════════════════════════════════
# Colors
# ═══════════════════════════════════════════════
CYAN='\033[96m'
GOLD='\033[93m'
GREEN='\033[92m'
RED='\033[91m'
DIM='\033[90m'
BOLD='\033[1m'
RESET='\033[0m'

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════╗"
echo -e "║  ${BOLD}W@HOME HIVE — Installer${RESET}${CYAN}                                ║"
echo -e "║  ${DIM}Akataleptos Distributed Spectral Search${RESET}${CYAN}                 ║"
echo -e "╚══════════════════════════════════════════════════════════╝${RESET}"
echo ""

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

echo -e "  ${DIM}OS:${RESET} $OS ($ARCH)"
echo -e "  ${DIM}Package Manager:${RESET} $PKG"
echo ""

# ═══════════════════════════════════════════════
# Check Python
# ═══════════════════════════════════════════════
PYTHON=""
for cmd in python3 python; do
    if command -v $cmd &>/dev/null; then
        ver=$($cmd --version 2>&1 | grep -oP '\d+\.\d+')
        major=$(echo $ver | cut -d. -f1)
        minor=$(echo $ver | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 8 ]; then
            PYTHON=$cmd
            echo -e "  ${GREEN}✓${RESET} Python: $($cmd --version)"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo -e "  ${RED}✗${RESET} Python 3.8+ not found"
    echo ""
    if [ "$OS" == "linux" ] && [ "$PKG" == "apt" ]; then
        echo -e "  ${CYAN}Install with:${RESET} sudo apt install python3 python3-pip python3-venv"
    elif [ "$OS" == "macos" ]; then
        echo -e "  ${CYAN}Install with:${RESET} brew install python3"
    fi
    echo ""
    read -p "  Install Python now? [Y/n] " yn
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
                brew install python3
            fi
            PYTHON=python3
            ;;
    esac
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
        echo -e "  ${GREEN}✓${RESET} GPU: $GPU_INFO"
    fi
else
    echo -e "  ${DIM}○${RESET} No NVIDIA GPU detected (CPU mode)"
fi
echo ""

# ═══════════════════════════════════════════════
# Install Location
# ═══════════════════════════════════════════════
INSTALL_DIR="$HOME/.whome"
echo -e "  ${DIM}Install directory:${RESET} $INSTALL_DIR"
echo ""

if [ -d "$INSTALL_DIR" ]; then
    echo -e "  ${GOLD}Existing installation found.${RESET}"
    read -p "  Update? [Y/n] " yn
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
echo ""
echo -e "  ${CYAN}Downloading W@Home...${RESET}"

HIVE_SERVER="${HIVE_SERVER:-https://akataleptos.com/hive}"

# Try downloading from server, fall back to local copy
for f in client.py w_operator.py w_cuda.py screensaver.py; do
    if curl -sSf "$HIVE_SERVER/static/$f" -o "$INSTALL_DIR/$f" 2>/dev/null; then
        echo -e "  ${GREEN}✓${RESET} $f"
    elif [ -f "$(dirname "$0")/$f" ]; then
        cp "$(dirname "$0")/$f" "$INSTALL_DIR/$f"
        echo -e "  ${GREEN}✓${RESET} $f (local)"
    else
        echo -e "  ${RED}✗${RESET} $f — not found"
    fi
done

# ═══════════════════════════════════════════════
# Virtual Environment
# ═══════════════════════════════════════════════
echo ""
echo -e "  ${CYAN}Setting up Python environment...${RESET}"

if [ ! -d "$INSTALL_DIR/venv" ]; then
    $PYTHON -m venv "$INSTALL_DIR/venv"
fi

source "$INSTALL_DIR/venv/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet numpy scipy requests

if $HAS_CUDA; then
    echo -e "  ${CYAN}Installing CUDA support...${RESET}"
    pip install --quiet cupy-cuda12x 2>/dev/null || \
    pip install --quiet cupy-cuda11x 2>/dev/null || \
    echo -e "  ${DIM}CuPy install failed — using CPU mode${RESET}"
fi

echo -e "  ${GREEN}✓${RESET} Dependencies installed"

# ═══════════════════════════════════════════════
# Wizard
# ═══════════════════════════════════════════════
echo ""
echo -e "${CYAN}════════════════════════════════════════════════${RESET}"
echo -e "  ${BOLD}Setup Wizard${RESET}"
echo -e "${CYAN}════════════════════════════════════════════════${RESET}"
echo ""

# Name
DEFAULT_NAME="$(hostname)-$(whoami)"
read -p "  Node name [$DEFAULT_NAME]: " NODE_NAME
NODE_NAME="${NODE_NAME:-$DEFAULT_NAME}"
echo -e "  ${GREEN}✓${RESET} Name: $NODE_NAME"
echo ""

# Server
DEFAULT_SERVER="https://akataleptos.com/hive"
read -p "  Hive server [$DEFAULT_SERVER]: " SERVER
SERVER="${SERVER:-$DEFAULT_SERVER}"
echo -e "  ${GREEN}✓${RESET} Server: $SERVER"
echo ""

# Screensaver
SETUP_SCREENSAVER=false
if [ "$OS" == "linux" ]; then
    read -p "  Set up as screensaver? [y/N] " yn
    case $yn in
        [Yy]* ) SETUP_SCREENSAVER=true; echo -e "  ${GREEN}✓${RESET} Screensaver: yes";;
        * ) echo -e "  ${DIM}○${RESET} Screensaver: no";;
    esac
    echo ""
fi

# Autostart
SETUP_AUTOSTART=false
read -p "  Start on login? [y/N] " yn
case $yn in
    [Yy]* ) SETUP_AUTOSTART=true; echo -e "  ${GREEN}✓${RESET} Autostart: yes";;
    * ) echo -e "  ${DIM}○${RESET} Autostart: no";;
esac
echo ""

# ═══════════════════════════════════════════════
# Write launcher script
# ═══════════════════════════════════════════════
cat > "$INSTALL_DIR/whome" << LAUNCHER
#!/bin/bash
cd "$INSTALL_DIR"
source venv/bin/activate
exec python3 client.py --server "$SERVER" --name "$NODE_NAME" "\$@"
LAUNCHER
chmod +x "$INSTALL_DIR/whome"

cat > "$INSTALL_DIR/whome-screensaver" << LAUNCHER
#!/bin/bash
cd "$INSTALL_DIR"
source venv/bin/activate
exec python3 client.py --server "$SERVER" --name "$NODE_NAME" --screensaver "\$@"
LAUNCHER
chmod +x "$INSTALL_DIR/whome-screensaver"

# Symlink to PATH
if [ -d "$HOME/.local/bin" ]; then
    ln -sf "$INSTALL_DIR/whome" "$HOME/.local/bin/whome"
    ln -sf "$INSTALL_DIR/whome-screensaver" "$HOME/.local/bin/whome-screensaver"
    echo -e "  ${GREEN}✓${RESET} Added 'whome' to PATH"
fi

# ═══════════════════════════════════════════════
# Screensaver (Linux/XScreenSaver)
# ═══════════════════════════════════════════════
if $SETUP_SCREENSAVER && [ "$OS" == "linux" ]; then
    XSCR_DIR="$HOME/.xscreensaver"
    if [ -f "$XSCR_DIR" ] || command -v xscreensaver &>/dev/null; then
        # Add to xscreensaver config
        if ! grep -q "whome-screensaver" "$XSCR_DIR" 2>/dev/null; then
            echo "  \"W@Home Hive\" $INSTALL_DIR/whome-screensaver \\n\\" >> "$XSCR_DIR.d/whome" 2>/dev/null || true
        fi
        echo -e "  ${GREEN}✓${RESET} XScreenSaver configured"
    else
        echo -e "  ${DIM}○${RESET} XScreenSaver not found — install with: sudo apt install xscreensaver"
    fi
fi

# ═══════════════════════════════════════════════
# Autostart
# ═══════════════════════════════════════════════
if $SETUP_AUTOSTART; then
    AUTOSTART_DIR="$HOME/.config/autostart"
    mkdir -p "$AUTOSTART_DIR"
    cat > "$AUTOSTART_DIR/whome.desktop" << DESKTOP
[Desktop Entry]
Type=Application
Name=W@Home Hive
Comment=Akataleptos Distributed Spectral Search
Exec=$INSTALL_DIR/whome
Icon=system-run
Terminal=false
Hidden=false
X-GNOME-Autostart-enabled=true
DESKTOP
    echo -e "  ${GREEN}✓${RESET} Autostart configured"
fi

# ═══════════════════════════════════════════════
# Done
# ═══════════════════════════════════════════════
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗"
echo -e "║  ${BOLD}Installation Complete!${RESET}${GREEN}                                  ║"
echo -e "╚══════════════════════════════════════════════════════════╝${RESET}"
echo ""
echo -e "  ${CYAN}Start computing:${RESET}     whome"
echo -e "  ${CYAN}Screensaver mode:${RESET}    whome-screensaver"
echo -e "  ${CYAN}With options:${RESET}         whome --help"
echo -e "  ${CYAN}Uninstall:${RESET}           rm -rf $INSTALL_DIR"
echo ""

read -p "  Start computing now? [Y/n] " yn
case $yn in
    [Nn]* ) echo -e "  Run ${CYAN}whome${RESET} whenever you're ready.";;
    * ) exec "$INSTALL_DIR/whome";;
esac
