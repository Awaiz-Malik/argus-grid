#!/usr/bin/env bash
# Starts (or stops) all 6 Argus Grid services as local background processes,
# for iterating without podman-compose. Logs go to data/logs/<service>.log.
#
# Usage:
#   scripts/run_local.sh start
#   scripts/run_local.sh stop

set -euo pipefail
cd "$(dirname "$0")/.."

VENV_PY=".venv/bin/uvicorn"
LOG_DIR="data/logs"
PID_FILE="data/logs/run_local.pids"

start() {
    mkdir -p "$LOG_DIR"
    : > "$PID_FILE"

    launch() {
        local name="$1" module="$2" port="$3"
        shift 3
        env "$@" "$VENV_PY" "$module" --port "$port" --log-level info \
            > "$LOG_DIR/$name.log" 2>&1 &
        echo "$!" >> "$PID_FILE"
        echo "started $name (pid $!) -> $LOG_DIR/$name.log"
    }

    launch vision-agent-site-a services.vision_agent.app:app 9001 ARGUS_SITE_ID=site-a
    launch vision-agent-site-b services.vision_agent.app:app 9002 ARGUS_SITE_ID=site-b
    launch vision-agent-site-c services.vision_agent.app:app 9003 ARGUS_SITE_ID=site-c
    launch triage-agent services.triage_agent.app:app 8090
    launch reporting-agent services.reporting_agent.app:app 8091
    launch orchestrator services.orchestrator.app:app 8080

    echo
    echo "Dashboard: http://localhost:8080"
    echo "Stop with: scripts/run_local.sh stop"
}

stop() {
    if [[ ! -f "$PID_FILE" ]]; then
        echo "No PID file found ($PID_FILE) - nothing to stop."
        exit 0
    fi
    while read -r pid; do
        [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
    done < "$PID_FILE"
    rm -f "$PID_FILE"
    echo "Stopped."
}

case "${1:-}" in
    start) start ;;
    stop) stop ;;
    *)
        echo "Usage: $0 {start|stop}"
        exit 1
        ;;
esac
