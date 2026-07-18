#!/bin/sh
set -eu

if [ "${IB_GATEWAY_DROPPED_PRIVILEGES:-false}" != "true" ]; then
    runtime_uid=${AURAMAUR_RUNTIME_UID:-1000}
    runtime_gid=${AURAMAUR_RUNTIME_GID:-1000}
    groupmod -o -g "$runtime_gid" ibgateway
    usermod -o -u "$runtime_uid" -g "$runtime_gid" ibgateway
    install -d -m 1777 /tmp/.X11-unix
    if ! gosu ibgateway test -w "$HOME/Jts"; then
        echo "IB Gateway settings are not writable by host UID $runtime_uid: $HOME/Jts" >&2
        exit 1
    fi
    export IB_GATEWAY_DROPPED_PRIVILEGES=true
    exec gosu ibgateway env HOME=/home/ibgateway "$0" "$@"
fi

password_file=${IB_GATEWAY_VNC_PASSWORD_FILE:-/run/secrets/ibgateway_vnc_password}
if [ ! -s "$password_file" ]; then
    echo "Missing non-empty VNC password secret: $password_file" >&2
    exit 1
fi

vnc_runtime=/tmp/auramaur-ibgateway
mkdir -p "$HOME/Jts" "$vnc_runtime"
x11vnc -storepasswd "$(cat "$password_file")" "$vnc_runtime/vnc.pass" >/dev/null
chmod 0600 "$vnc_runtime/vnc.pass"

Xvfb "$DISPLAY" -screen 0 1280x800x24 -nolisten tcp &
xvfb_pid=$!
x_socket="/tmp/.X11-unix/X${DISPLAY#:}"
x_wait=0
while [ ! -S "$x_socket" ]; do
    if ! kill -0 "$xvfb_pid" 2>/dev/null; then
        echo "Xvfb exited before display $DISPLAY became ready" >&2
        exit 1
    fi
    x_wait=$((x_wait + 1))
    if [ "$x_wait" -ge 100 ]; then
        echo "Timed out waiting for Xvfb display $DISPLAY" >&2
        exit 1
    fi
    sleep 0.1
done
fluxbox >/tmp/fluxbox.log 2>&1 &
x11vnc -display "$DISPLAY" -rfbauth "$vnc_runtime/vnc.pass" \
    -localhost -forever -shared >/tmp/x11vnc.log 2>&1 &
websockify --web=/usr/share/novnc/ 6080 localhost:5900 >/tmp/novnc.log 2>&1 &

gateway=$(find "${IB_GATEWAY_HOME:-/opt/ibgateway}" -type f -name ibgateway -perm -u+x | head -1)
if [ -z "$gateway" ]; then
    echo "IB Gateway executable not found" >&2
    exit 1
fi

echo "IB Gateway GUI ready on container port 6080; authenticate manually."
exec "$gateway"
