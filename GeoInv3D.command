#!/bin/bash
# GeoInv3D: start the local API server (unless it is already running) and open the upload page.
# The macOS counterpart of GeoInv3D.bat: double-click it in Finder.  This Terminal window holds
# the server and this machine's AWS access; close it (or press Ctrl-C) to stop the server.
# While the server runs the Mac is kept from sleeping (caffeinate; closing the lid on battery
# still sleeps it), so running jobs keep being followed and their results fetched.
cd "$(dirname "$0")" || exit 1
PORT="${GEOINV3D_PORT:-8000}"
URL="http://localhost:$PORT/"
HEALTH="http://127.0.0.1:$PORT/api/health"
PYTHON=".venv/bin/python"

running() { curl -s --max-time 2 "$HEALTH" 2>/dev/null | grep -q '"service":"GeoInv3D'; }

fail() {
    echo
    echo "$1"
    read -n 1 -s -r -p "Press any key to close this window."
    echo
    exit 1
}

if running; then
    echo "GeoInv3D is already running on port $PORT."
    open "$URL"
    exit 0
fi

if [ ! -x "$PYTHON" ]; then
    fail "No Python environment in $(pwd)/.venv.  Create it here with:
  uv venv --python 3.13
  uv pip install -e \".[full,dev,cloud]\" rasterio"
fi
if ! "$PYTHON" -c "import fastapi, uvicorn, boto3, paramiko" 2>/dev/null; then
    fail "The server needs the cloud extras.  Install them here with:
  uv pip install -e \".[cloud]\""
fi
if [ -z "$AWS_ACCESS_KEY_ID$AWS_PROFILE" ] && [ ! -f "$HOME/.aws/credentials" ]; then
    echo "Warning: no AWS credentials (~/.aws/credentials).  The page opens, but jobs cannot"
    echo "start until you run 'aws configure'."
    echo
fi

echo "Starting the GeoInv3D server on port $PORT ..."
"$PYTHON" -m geoinv3d.api --host 127.0.0.1 --port "$PORT" &
SERVER=$!
# closing the window (SIGHUP) or Ctrl-C stops the server with it
trap 'kill "$SERVER" 2>/dev/null' EXIT
trap 'exit 130' INT TERM HUP

for _ in $(seq 60); do
    running && break
    kill -0 "$SERVER" 2>/dev/null || fail "The server stopped (see the messages above)."
    sleep 0.5
done
running || fail "The server did not start within 30 s (see the messages above)."

open "$URL"
caffeinate -i -s -w "$SERVER" &
echo
echo "Upload page: $URL"
echo "Close this window, or press Ctrl-C, to stop the server."
wait "$SERVER"
