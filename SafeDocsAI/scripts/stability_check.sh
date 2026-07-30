#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8001}"
API_PREFIX="${API_PREFIX:-/api/v1}"
LOGIN_URL="${BASE_URL}${API_PREFIX}/auth/login"
CHAT_URL="${BASE_URL}${API_PREFIX}/chat/"

STABILITY_USERNAME="${STABILITY_USERNAME:-admin}"
STABILITY_PASSWORD="${STABILITY_PASSWORD:-admin123}"
DURATION_SECONDS="${DURATION_SECONDS:-600}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-5}"

echo "Stability check started"
echo "Base URL: ${BASE_URL}"
echo "Duration: ${DURATION_SECONDS}s, Interval: ${INTERVAL_SECONDS}s"

login_response="$(curl -sS -X POST \
  -F "username=${STABILITY_USERNAME}" \
  -F "password=${STABILITY_PASSWORD}" \
  "${LOGIN_URL}")"

token="$(printf '%s' "${login_response}" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("access_token",""))')"
if [[ -z "${token}" ]]; then
  echo "Failed to obtain token from ${LOGIN_URL}"
  echo "Response: ${login_response}"
  exit 1
fi

start_ts="$(date +%s)"
ok_count=0
fail_count=0

while true; do
  now_ts="$(date +%s)"
  elapsed=$((now_ts - start_ts))
  if (( elapsed >= DURATION_SECONDS )); then
    break
  fi

  status_code="$(curl -sS -o /tmp/andozai_stability_response.json -w "%{http_code}" \
    -X POST "${CHAT_URL}" \
    -H "Authorization: Bearer ${token}" \
    -H "Content-Type: application/json" \
    -d '{"question":"Какая ставка НДС в Таджикистане?"}')"

  if [[ "${status_code}" == "200" ]]; then
    ok_count=$((ok_count + 1))
    echo "[OK] request #$((ok_count + fail_count))"
  else
    fail_count=$((fail_count + 1))
    echo "[FAIL] status=${status_code} body=$(cat /tmp/andozai_stability_response.json)"
  fi

  sleep "${INTERVAL_SECONDS}"
done

echo "Stability check finished"
echo "Successful requests: ${ok_count}"
echo "Failed requests: ${fail_count}"

if (( fail_count > 0 )); then
  exit 1
fi
