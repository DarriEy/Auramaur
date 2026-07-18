#!/bin/sh
set -eu

pgrep -f 'ibgateway|java' >/dev/null
nc -z 127.0.0.1 "${IB_GATEWAY_HEALTH_PORT:-4002}"
