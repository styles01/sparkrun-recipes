"""NOTIFY: one push to the operator's phone through the ntfy stack, out of band.

The one action that leaves the box. It is off unless the policy's `notify.enabled` says so (the
stall watchdog's setting; the owner allowed it on 2026-09-30, off by default and on only in the
overnight profile), it is never sent in shadow, and the publish token lives in a file on the box
(`token_file`), never in the policy, the journal or a log line. `send` returns what happened as
a short string for the journal; it never raises.

The server's monitor holds the sender (`Monitor.notifier`); tests and the fake engine put their
own there, so no test can reach the network.
"""

from __future__ import annotations

import os
import ssl
import urllib.error
import urllib.request


#: the topic URL when the policy's `notify.url` is empty (the shipped policy names none)
URL_ENV = "TANDEM_NOTIFY_URL"


def target(cfg: dict) -> str:
    """The topic URL: the policy's `url`, else `$TANDEM_NOTIFY_URL`, else empty."""
    return str(cfg.get("url") or os.environ.get(URL_ENV) or "").strip()


def send(cfg: dict, title: str, message: str, opener=None) -> str:
    """POST `message` to the topic URL (`target`) with the ntfy headers; "sent 200" or
    "failed: why"."""
    url = target(cfg)
    if not url:
        return "failed: no url"
    path = os.path.expanduser(cfg.get("token_file") or "")
    try:
        with open(path) as f:
            token = f.read().strip()
    except OSError:
        return "failed: no token file"
    if not token:
        return "failed: empty token file"
    headers = {"Title": title, "Priority": cfg.get("priority") or "default",
               "Authorization": "Bearer " + token, "Content-Type": "text/plain; charset=utf-8"}
    if cfg.get("tags"):
        headers["Tags"] = cfg["tags"]
    req = urllib.request.Request(url, data=message.encode(), method="POST",
                                 headers=headers)
    try:
        ctx = (ssl.create_default_context(cafile=os.path.expanduser(cfg["ca_file"]))
               if cfg.get("ca_file") else None)
        with (opener or urllib.request.urlopen)(req, timeout=float(cfg.get("timeout_s") or 10),
                                                context=ctx) as r:
            return f"sent {r.status}"
    except urllib.error.HTTPError as exc:
        return f"failed: HTTP {exc.code}"
    except (OSError, ValueError, ssl.SSLError) as exc:
        return f"failed: {type(exc).__name__}"
