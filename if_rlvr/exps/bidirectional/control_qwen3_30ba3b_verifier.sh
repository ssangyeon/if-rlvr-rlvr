#!/usr/bin/env bash

set -euo pipefail

BASE_URL=${IF_LLM_VERIFIER_CONTROL_URL:-http://127.0.0.1:22900}
BASE_URL=${BASE_URL%/}
BASE_URL=${BASE_URL%/v1}
ACTION=${1:-status}

request() {
    curl --fail --silent --show-error "$@"
}

status() {
    request "${BASE_URL}/is_sleeping"
    echo
}

case "${ACTION}" in
    status)
        status
        ;;
    sleep)
        LEVEL=${2:-1}
        if [[ "${LEVEL}" != "1" && "${LEVEL}" != "2" ]]; then
            echo "ERROR: sleep level must be 1 or 2." >&2
            exit 2
        fi
        request -X POST "${BASE_URL}/sleep?level=${LEVEL}" >/dev/null
        status
        ;;
    wake|wake_up)
        request -X POST "${BASE_URL}/wake_up" >/dev/null
        status
        ;;
    *)
        echo "Usage: $0 {status|sleep [1|2]|wake}" >&2
        exit 2
        ;;
esac
