"""Temporary loopback HTTPS service for exact root-admitted runtime assets."""
from pathlib import Path
import argparse
import json
import ssl
from http.server import HTTPServer, BaseHTTPRequestHandler

p = argparse.ArgumentParser()
p.add_argument('--directory', type=Path, required=True)
a = p.parse_args()
root = a.directory.resolve()
assets = json.loads((root / 'asset-map.json').read_text())
manifest = (root / 'primary-manifest.json').read_bytes()

class Handler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        self.send_asset(False)

    def do_GET(self):
        self.send_asset(True)

    def send_asset(self, body):
        file = None
        if self.path == '/manifest.json':
            size = len(manifest)
        elif self.path in assets:
            file = Path(assets[self.path])
            size = file.stat().st_size
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Length', str(size))
        self.send_header('Content-Type', 'application/octet-stream' if file else 'application/json')
        self.end_headers()
        if not body:
            return
        if file:
            with file.open('rb') as source:
                for chunk in iter(lambda: source.read(1 << 20), b''):
                    self.wfile.write(chunk)
        else:
            self.wfile.write(manifest)

    def log_message(self, fmt, *args):
        print(json.dumps({'method': self.command, 'path': self.path,
                          'message': fmt % args}), flush=True)

server = HTTPServer(('127.0.0.1', 18443), Handler)
server.timeout = 5
tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
tls.load_cert_chain(root / 'asset-ca.pem', root / 'asset-key.pem')
server.socket = tls.wrap_socket(server.socket, server_side=True)
server.serve_forever()
