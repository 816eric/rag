#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PATH="$HOME/development/flutter/bin:$PATH"

echo "Stopping backend (port 8001)..."
fuser -k 8001/tcp 2>/dev/null

echo "Stopping frontend (port 5173)..."
fuser -k 5173/tcp 2>/dev/null

echo "Stopping any leftover flutter/dart processes for this project..."
pkill -f "flutter.*rag_frontend" 2>/dev/null
pkill -f "dart.*rag_frontend" 2>/dev/null

sleep 2

mkdir -p logs

echo "Starting backend..."
nohup "$SCRIPT_DIR/env/bin/python" -m uvicorn backend.main:app --host 127.0.0.1 --port 8001 \
    > logs/backend.log 2>&1 &
echo "Backend PID: $!"

sleep 3

echo "Starting frontend..."
(cd "$SCRIPT_DIR/frontend" && nohup flutter run -d web-server --web-hostname=127.0.0.1 --web-port=5173 \
    > "$SCRIPT_DIR/logs/frontend.log" 2>&1 &)
echo "Frontend PID: $!"

echo
echo "Backend and frontend are starting in the background."
echo "Backend log:  $SCRIPT_DIR/logs/backend.log"
echo "Frontend log: $SCRIPT_DIR/logs/frontend.log"
echo "Backend:  http://127.0.0.1:8001/api/health"
echo "Frontend: http://127.0.0.1:5173  (open manually in your browser — no Chrome installed to auto-launch)"
