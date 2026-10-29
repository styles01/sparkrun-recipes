#!/usr/bin/env python3
"""Engine-log relay: `docker logs --follow dsv41-serve` -> /home/jaita/sparkgateway/engine.log

Runs on the HOST (the sparkDash container has no docker binary/socket).
sparkDash reads the file through the /app/gateway bind mount and streams new
lines to the /live pane over the /ws/llm socket.
"""
import os
import subprocess
import sys

CONTAINER = os.environ.get("ENGINE_CONTAINER", "dsv41-serve")
OUT = os.environ.get("ENGINE_LOG", "/home/jaita/sparkgateway/engine.log")
MAX_BYTES = int(os.environ.get("ENGINE_LOG_MAX_BYTES", str(64 * 1024 * 1024)))


def rotate_if_big():
    try:
        if os.path.getsize(OUT) > MAX_BYTES:
            os.replace(OUT, OUT + ".1")
    except OSError:
        pass


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rotate_if_big()
    with open(OUT, "ab", buffering=0) as out:
        while True:
            proc = subprocess.Popen(
                ["docker", "logs", "--follow", "--tail", "200", CONTAINER],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            try:
                for line in proc.stdout:
                    out.write(line)
            except Exception as e:  # noqa: BLE001
                out.write(f"[engine-log] relay error: {e}\n".encode())
            proc.wait()
            out.write(b"[engine-log] follow ended; container not running -- retrying in 5s\n")
            import time

            time.sleep(5)
            rotate_if_big()


if __name__ == "__main__":
    sys.exit(main())
