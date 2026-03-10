#!/bin/bash
# Download Pyodide + numpy + scipy for local serving
# Run once on the server. Files total ~71MB.
# After this, the hive server serves all Pyodide files locally —
# no CDN dependency, works through VPN, on Hostinger, anywhere.

set -e

PYODIDE_VER="0.25.1"
BASE="https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VER}/full"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIR="${SCRIPT_DIR}/pyodide"

mkdir -p "$DIR"

echo "Downloading Pyodide v${PYODIDE_VER} for local serving..."
echo ""

# Core runtime files
CORE_FILES="pyodide.js pyodide.asm.js pyodide.asm.wasm pyodide-lock.json python_stdlib.zip"
for f in $CORE_FILES; do
    if [ -f "$DIR/$f" ]; then
        echo "  [cached] $f"
    else
        echo -n "  $f... "
        curl -sSL "$BASE/$f" -o "$DIR/$f"
        du -h "$DIR/$f" | awk '{print $1}'
    fi
done

# Package files (numpy, scipy, openblas)
PACKAGES="numpy-1.26.4-cp311-cp311-emscripten_3_1_46_wasm32.whl openblas-0.3.23.zip scipy-1.11.2-cp311-cp311-emscripten_3_1_46_wasm32.whl"
for f in $PACKAGES; do
    if [ -f "$DIR/$f" ]; then
        echo "  [cached] $f"
    else
        echo -n "  $f... "
        curl -sSL "$BASE/$f" -o "$DIR/$f"
        du -h "$DIR/$f" | awk '{print $1}'
    fi
done

echo ""
echo "Done! Total size:"
du -sh "$DIR"
echo ""
echo "The hive server will serve these at /pyodide/ — no CDN needed."
