"""server.py + completion client for the H1 rig.

Serving convention (plan §1.2, eval_scenarios verbatim): llama-b10453
CPU-only build, -t 8 -ngl 0 --parallel 1 -c 8192, --host 127.0.0.1, tracked
PID only, readiness = a POST /v1/completions returning HTTP 200 (never
/health). H1 ports: 1830x. No CUDA context is ever created.
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_SERVER = HERE.parent / "bin" / "llama" / "llama-b10453" / "llama-server"
H1_PORT = 18310


def port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def complete(port: int, prompt: str, max_tokens: int, stop: list[str],
             timeout: int = 600):
    """(text, latency_s) — temperature 0, the plan's decode convention."""
    body = json.dumps({"prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0, "stop": stop,
                       "stream": False}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    return data["choices"][0]["text"], time.time() - t0


class Server:
    """Own one llama-server child (tracked PID; never pkill by name)."""

    def __init__(self, model, port=H1_PORT, threads=8, ctx=8192,
                 binary=DEFAULT_SERVER, log_path=None):
        self.binary, self.model, self.port = str(binary), str(model), port
        self.threads, self.ctx = threads, ctx
        self.log_path = Path(log_path or HERE / f"llama-server-h1-{port}.log")
        self.proc = None

    def start(self, ready_timeout=1800):
        if port_open(self.port):
            raise RuntimeError(
                f"port {self.port} in use; refusing to touch a server we did "
                f"not start (tracked-PID-only policy)")
        cmd = [self.binary, "-m", self.model, "--port", str(self.port),
               "--host", "127.0.0.1", "-t", str(self.threads),
               "--parallel", "1", "-c", str(self.ctx), "-ngl", "0"]
        with open(self.log_path, "ab") as log:
            log.write(f"\n==== {time.strftime('%F %T')} {' '.join(cmd)}\n".encode())
            log.flush()
            self.proc = subprocess.Popen(cmd, stdout=log,
                                         stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL,
                                         start_new_session=True)
        t0 = time.time()
        while time.time() - t0 < ready_timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server rc={self.proc.returncode}; "
                                   f"log tail: {self.log_tail()}")
            try:
                body = json.dumps({"prompt": "readiness", "max_tokens": 1,
                                   "temperature": 0}).encode()
                req = urllib.request.Request(
                    f"http://127.0.0.1:{self.port}/v1/completions", data=body,
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    if r.status == 200:
                        return
            except (urllib.error.URLError, urllib.error.HTTPError,
                    OSError, json.JSONDecodeError):
                pass
            time.sleep(2)
        raise RuntimeError("llama-server not ready in time; "
                           f"log tail: {self.log_tail()}")

    def log_tail(self, n=1500):
        try:
            return self.log_path.read_text(errors="replace")[-n:]
        except OSError:
            return "<no log>"

    def stop(self):
        if self.proc is None:
            return
        pid = self.proc.pid
        try:
            os.kill(pid, signal.SIGTERM)
            for _ in range(75):
                if self.proc.poll() is not None:
                    break
                time.sleep(0.2)
            if self.proc.poll() is None:
                os.kill(pid, signal.SIGKILL)
            self.proc.wait(10)
        except (ProcessLookupError, PermissionError):
            pass
        finally:
            self.proc = None


class PairClient:
    """Completion cache with the plan's 2-rollout noise control.

    Key = (prompt, stops, max_tokens). First evaluation of a key issues TWO
    SEQUENTIAL temp-0 completions (the nondeterminism probe); later
    candidates hitting the same key reuse the stored pair (counted as a
    pair_reuse, not a fresh rollout — this is the archive semantics:
    identical requests are not re-spun). Ledger counts real completions.
    Subclasses override _do_complete for offline tests.
    """

    def __init__(self, port: int, fresh: bool = False):
        self.port = port
        self.fresh = fresh  # baseline re-measure mode: bypass reads, overwrite
        self.cache: dict[tuple, list[dict]] = {}
        self.ledger = dict(new_completions=0, pair_reuses=0, cache_entries=0)

    def _do_complete(self, prompt, max_tokens, stops):
        return complete(self.port, prompt, max_tokens, stops)

    @staticmethod
    def key(prompt: str, stops: list[str], max_tokens: int):
        return (prompt, tuple(stops), max_tokens)

    def pair(self, prompt: str, stops: list[str], max_tokens: int) -> list[dict]:
        k = self.key(prompt, stops, max_tokens)
        if k in self.cache and not self.fresh:
            self.ledger["pair_reuses"] += 1
            return self.cache[k]
        out = []
        for _ in range(2):  # sequential: a fair nondeterminism probe
            text, lat = self._do_complete(prompt, max_tokens, stops)
            out.append(dict(text=text, latency=round(lat, 3)))
        self.cache[k] = out
        self.ledger["new_completions"] += 2
        self.ledger["cache_entries"] = len(self.cache)
        return out

    def single(self, prompt: str, stops: list[str], max_tokens: int) -> dict:
        """Verdict-battery mode: reuse the cached pair's first rollout when
        available, else one fresh completion (the eval_scenarios 1-shot
        convention). Fresh mode never writes the cache from a single."""
        k = self.key(prompt, stops, max_tokens)
        if k in self.cache and not self.fresh:
            self.ledger["pair_reuses"] += 1
            return self.cache[k][0]
        text, lat = self._do_complete(prompt, max_tokens, stops)
        if not self.fresh:
            self.ledger["new_completions"] += 1
        return dict(text=text, latency=round(lat, 3))

    def dump(self, path: Path):
        """Persist the cache for resume (key hashed; texts kept for reuse)."""
        import hashlib
        with open(path, "w") as f:
            for (prompt, stops, mt), rolls in self.cache.items():
                kh = hashlib.sha1(
                    f"{prompt}\x00{stops}\x00{mt}".encode()).hexdigest()[:16]
                f.write(json.dumps(dict(k=kh, stops=list(stops), max_tokens=mt,
                                        prompt=prompt,
                                        texts=[r["text"] for r in rolls],
                                        latencies=[r["latency"] for r in rolls])) + "\n")

    def load(self, path: Path):
        if not Path(path).exists():
            return
        for line in open(path):
            r = json.loads(line)
            k = self.key(r["prompt"], r["stops"], r["max_tokens"])
            self.cache[k] = [dict(text=t, latency=l) for t, l in
                             zip(r["texts"], r["latencies"])]
        self.ledger["cache_entries"] = len(self.cache)


def dump_all(clients, path: Path):
    """Merge-dump every client's cache into one file (keys are port-free,
    so they are globally unique)."""
    seen = {}
    for c in clients:
        seen.update(c.cache)
    with open(path, "w") as f:
        for (prompt, stops, mt), rolls in seen.items():
            f.write(json.dumps(dict(stops=list(stops), max_tokens=mt,
                                    prompt=prompt,
                                    texts=[r["text"] for r in rolls],
                                    latencies=[r["latency"] for r in rolls])) + "\n")


def wait_port_gone(port: int, timeout: float = 30.0):
    t0 = time.time()
    while port_open(port) and time.time() - t0 < timeout:
        time.sleep(0.5)
