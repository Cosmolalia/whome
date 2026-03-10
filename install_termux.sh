#!/data/data/com.termux/files/usr/bin/bash
# W@Home Hive — Termux (Android) Installer
# Install Termux from F-Droid (NOT Play Store — the Play Store version is outdated)
# Then run: curl -sSL https://akataleptos.com/hive/install_termux.sh | bash

set -e

echo ""
echo "  ╔══════════════════════════════════════════════════════════╗"
echo "  ║  W@HOME HIVE — Android/Termux Installer                ║"
echo "  ╚══════════════════════════════════════════════════════════╝"
echo ""

# Install system deps
echo "  Installing system packages..."
pkg update -y
pkg install -y python python-pip git curl

echo "  Installing Python libraries..."
pip install --quiet numpy scipy requests

# Install dir
INSTALL_DIR="$HOME/.whome"
mkdir -p "$INSTALL_DIR"

# Download files
SERVER="${HIVE_SERVER:-https://akataleptos.com/hive}"
echo ""
echo "  Downloading W@Home..."
for f in client.py w_operator.py; do
    if curl -sSf "$SERVER/static/$f" -o "$INSTALL_DIR/$f" 2>/dev/null; then
        echo "  [OK] $f"
    elif [ -f "$(dirname "$0")/$f" ]; then
        cp "$(dirname "$0")/$f" "$INSTALL_DIR/$f"
        echo "  [OK] $f (local)"
    else
        echo "  [!!] $f — not found"
    fi
done

# Launcher
cat > "$INSTALL_DIR/whome" << 'LAUNCHER'
#!/data/data/com.termux/files/usr/bin/bash
cd "$HOME/.whome"
exec python client.py "$@"
LAUNCHER
chmod +x "$INSTALL_DIR/whome"

# Add to PATH
if ! grep -q ".whome" "$HOME/.bashrc" 2>/dev/null; then
    echo 'export PATH="$HOME/.whome:$PATH"' >> "$HOME/.bashrc"
fi
export PATH="$HOME/.whome:$PATH"

# Setup
echo ""
echo "  ════════════════════════════════════════════════"
echo "    Setup"
echo "  ════════════════════════════════════════════════"
echo ""

DEFAULT_NAME="android-$(whoami)"
read -p "  Node name [$DEFAULT_NAME]: " NODE_NAME
NODE_NAME="${NODE_NAME:-$DEFAULT_NAME}"

DEFAULT_SERVER="https://akataleptos.com/hive"
read -p "  Hive server [$DEFAULT_SERVER]: " SERVER_URL
SERVER_URL="${SERVER_URL:-$DEFAULT_SERVER}"

# Save config into launcher
cat > "$INSTALL_DIR/whome" << LAUNCHER
#!/data/data/com.termux/files/usr/bin/bash
cd "\$HOME/.whome"
exec python client.py --server "$SERVER_URL" --name "$NODE_NAME" "\$@"
LAUNCHER
chmod +x "$INSTALL_DIR/whome"

# Termux:Boot autostart (optional)
if [ -d "$HOME/.termux/boot" ] || pkg list-installed 2>/dev/null | grep -q termux-boot; then
    read -p "  Start on boot? [y/N] " yn
    case $yn in
        [Yy]* )
            mkdir -p "$HOME/.termux/boot"
            cp "$INSTALL_DIR/whome" "$HOME/.termux/boot/whome"
            echo "  [OK] Autostart configured"
            ;;
    esac
fi

echo ""
echo "  ╔══════════════════════════════════════════════════════════╗"
echo "  ║  Installation Complete!                                  ║"
echo "  ╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  Start computing:  whome"
echo "  With options:      whome --help"
echo "  Uninstall:         rm -rf ~/.whome"
echo ""

read -p "  Start computing now? [Y/n] " yn
case $yn in
    [Nn]* ) echo "  Run 'whome' whenever you're ready.";;
    * ) exec "$INSTALL_DIR/whome";;
esac
