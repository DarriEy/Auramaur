FROM ubuntu:24.04

ARG IB_GATEWAY_INSTALLER=ibgateway-latest-standalone-linux-x64.sh

ENV DEBIAN_FRONTEND=noninteractive \
    DISPLAY=:1 \
    IB_GATEWAY_HOME=/opt/ibgateway \
    HOME=/home/ibgateway

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates fluxbox gosu netcat-openbsd novnc openjdk-17-jre-headless \
       procps websockify x11vnc xauth xdotool xvfb \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --uid 10002 --create-home ibgateway \
    && mkdir -p /home/ibgateway/Jts /run/ibgateway "${IB_GATEWAY_HOME}" \
    && chown -R ibgateway:ibgateway /home/ibgateway /run/ibgateway "${IB_GATEWAY_HOME}"

# IBKR does not support unattended credential entry or a GUI-less Gateway.
# Download the current offline Linux Gateway installer from IBKR, place it in
# this directory under the ARG filename, then build. It is intentionally not
# redistributed by this repository.
COPY ${IB_GATEWAY_INSTALLER} /tmp/ibgateway-installer.sh
RUN chown ibgateway:ibgateway /tmp/ibgateway-installer.sh \
    && chmod 0700 /tmp/ibgateway-installer.sh \
    && gosu ibgateway env HOME=/home/ibgateway \
       xvfb-run -a /tmp/ibgateway-installer.sh -q -dir "${IB_GATEWAY_HOME}" \
    && find "${IB_GATEWAY_HOME}" -type f -name ibgateway -perm -u+x | grep -q . \
    && rm -f /tmp/ibgateway-installer.sh

COPY --chown=ibgateway:ibgateway entrypoint.sh healthcheck.sh /usr/local/bin/
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=4 \
    CMD ["/usr/local/bin/healthcheck.sh"]
