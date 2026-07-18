# IB Gateway image

IBKR requires a GUI for authentication and does not support unattended
credential entry. Download the current **offline Linux x86-64 IB Gateway**
installer from IBKR into this directory as
`ibgateway-latest-standalone-linux-x64.sh`, then build with Compose.

The image runs Gateway on a virtual display and exposes noVNC on port 6080.
Compose binds it to host loopback only. Reach a cloud host through Tailscale or
an SSH tunnel; never publish 6080 or the API ports to the internet.

After login, configure API ports 4002 (paper) and/or 4001 (live), automatic
weekly restarts, trusted localhost/API access, and the intended read-only mode.
Use a dedicated IBKR username so another Client Portal/TWS login does not break
Gateway reconnection. Weekly interactive reauthentication remains required.
