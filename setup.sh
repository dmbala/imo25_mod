#!/usr/bin/env bash
# Create .venv and install dependencies with uv.
#
# uv takes an exclusive flock on the target environment. On some cluster mounts
# -- /n/netscratch here -- that lock never returns, so `uv sync` hangs forever
# with no output. When that happens this script resolves and installs into a
# staging venv on local disk (where uv is fine) and relocates the result.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3.12}"
VENV="$ROOT/.venv"
STAGE="${TMPDIR:-/tmp}/imo25-stage-$$"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${TMPDIR:-/tmp}/uv-cache-$USER}"

command -v uv >/dev/null || { echo "uv not found on PATH" >&2; exit 1; }
[ -x "$PYTHON_BIN" ] || { echo "no interpreter at $PYTHON_BIN (override with PYTHON_BIN=)" >&2; exit 1; }

echo "==> creating $VENV"
rm -rf "$VENV"
uv venv "$VENV" --python "$PYTHON_BIN" >/dev/null

echo "==> trying uv sync directly (hangs on some cluster mounts; 90s cap)"
if (cd "$ROOT" && timeout 90 uv sync --extra dev >/dev/null 2>&1); then
    echo "    ok"
else
    echo "    uv could not write to this filesystem; staging on local disk instead"
    trap 'rm -rf "$STAGE"' EXIT
    uv venv "$STAGE" --python "$PYTHON_BIN" >/dev/null
    uv pip install --python "$STAGE/bin/python" anthropic openai pyyaml pytest
    uv pip install --python "$STAGE/bin/python" --no-deps -e "$ROOT"

    PYVER="$("$PYTHON_BIN" -c 'import sys; print("python%d.%d" % sys.version_info[:2])')"
    cp -a "$STAGE/lib/$PYVER/site-packages/." "$VENV/lib/$PYVER/site-packages/"
    for f in "$STAGE"/bin/*; do
        b=$(basename "$f")
        case "$b" in python*|activate*|deactivate*|pydoc*|*.bat|*.ps1|*.csh|*.fish|*.nu) continue;; esac
        sed "1s|^#!.*|#!$VENV/bin/python|" "$f" > "$VENV/bin/$b"
        chmod +x "$VENV/bin/$b"
    done
fi

echo "==> verifying"
"$VENV/bin/python" -c "import anthropic, openai, yaml; print('    imports ok')"
"$VENV/bin/imo-solve" --validate
echo
echo "Done. Run:  $VENV/bin/imo-solve problems/imo01.txt --model claude-opus-5"
