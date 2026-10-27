# The dashboard and the OpenAI-compatible proxy as an image: no engine, no GPU, no torch.
#
#   docker build -f ops/Dockerfile.dash -t tandemllm-dash .        (from the repository's root)
#   docker run -d --name tandem-dash -p 127.0.0.1:8080:8080 -v tandem-dash:/data \
#       --env-file ops/dash.env tandemllm-dash
#
# Everything the proxy keeps (the config database, the usage ledger, the key file when no
# TANDEM_SECRETS_KEY is given) lives in the /data volume. The server binds 0.0.0.0 inside the
# container; what is reachable from outside is what `-p` publishes. Publish it to 127.0.0.1 or a
# trusted LAN address, never to the internet: with no TANDEM_ADMIN_TOKEN set every route is open
# (docs/proxy.md).
#
# The System view's host collector sees the container, not the machine. For the machine and the
# processes behind backends on it, run the container with `--pid=host` (and `--gpus all` for
# NVML), or run `tandem-dash hostcollect` on that machine and point the backend's host at it.

FROM python:3.12-slim AS build
WORKDIR /src
RUN pip install --no-cache-dir build==1.3.0
COPY pyproject.toml VERSION LICENSE COMMERCIAL-LICENSE.md NOTICE ./
COPY docs/proxy.md docs/proxy.md
COPY proxy proxy
COPY server server
COPY dashboard/dist dashboard/dist
RUN python -m build --wheel -o /wheel .

FROM python:3.12-slim
RUN useradd --create-home --uid 10001 tandem && mkdir -p /data && chown tandem /data
COPY --from=build /wheel/*.whl /tmp/
RUN pip install --no-cache-dir "$(ls /tmp/*.whl)[host]" && rm /tmp/*.whl
USER tandem
ENV TANDEM_CONFIG_DB=/data/config.sqlite3 \
    TANDEM_USAGE_LEDGER=/data/ledger.sqlite3 \
    TANDEM_HOST=0.0.0.0 \
    TANDEM_PORT=8080 \
    PYTHONUNBUFFERED=1
VOLUME ["/data"]
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8080/health', timeout=4)"
STOPSIGNAL SIGTERM
ENTRYPOINT ["tandem-dash"]
CMD ["serve", "--no-keychain"]
