#!/usr/bin/env python3
"""Start one owned CPU server, check health, then exercise OpenAI completions."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import time
import urllib.request


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--server', type=Path, required=True)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--log', type=Path, required=True)
    args = ap.parse_args()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    with args.log.open('w') as log:
        child = subprocess.Popen([str(args.server.resolve()), '-m', str(args.model.resolve()), '--host', '127.0.0.1', '--port', str(port), '-c', '512', '-t', '2', '-ngl', '0'], stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if child.poll() is not None:
                    raise RuntimeError(f'server exited {child.returncode}; see {args.log}')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2) as response:
                        assert response.status == 200
                    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/completions', data=json.dumps(dict(prompt='The answer is', max_tokens=2, temperature=0)).encode(), headers={'Content-Type': 'application/json'})
                    with urllib.request.urlopen(req, timeout=10) as response:
                        result = json.load(response)
                    assert isinstance(result['choices'][0]['text'], str)
                    print(json.dumps(dict(ok=True, model=str(args.model), completion=result)))
                    return
                except (OSError, KeyError):
                    time.sleep(1)
            raise RuntimeError('Completion readiness timeout')
        finally:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == '__main__':
    main()
