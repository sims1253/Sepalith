#!/usr/bin/env python3
"""Night-runner notify hook: POST the runner event to SEPALITH_NOTIFY_URL.

    night-runner.sh run --notify-command /usr/bin/python3 -I -B <repo>/scripts/night/notify-webhook.py

The runner sets SEPALITH_EVENT (idle, failed, needs_user), SEPALITH_DETAIL and
SEPALITH_REPORT. The URL (a T3 scheduled-task webhook that wakes the
orchestrating thread) is a credential: it lives in
~/.config/sepalith/secrets.env, never in the repo. Without it this is a no-op.
"""
import json
import os
import socket
import sys
import urllib.request


def main() -> int:
    url = os.environ.get("SEPALITH_NOTIFY_URL", "")
    if not url:
        return 0
    body = {"event": os.environ.get("SEPALITH_EVENT", ""), "detail": os.environ.get("SEPALITH_DETAIL", ""),
            "report": os.environ.get("SEPALITH_REPORT", ""), "host": socket.gethostname()}
    request = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json",
                                              "User-Agent": "sepalith-night-runner/1"})  # the relay rejects urllib's default UA (403)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return 0 if 200 <= response.status < 300 else 1
    except OSError as error:
        print(f"notify-webhook: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
