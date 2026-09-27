"""Loopback-only HTTP server; static assets are explicitly allowlisted."""

from __future__ import annotations

import argparse
import json
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from ui.runtime import SessionBusy, SessionController

ASSETS = Path(__file__).parent / "static"


def make_handler(controller, token):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, code, body=b"", mime="application/json; charset=utf-8", attachment=None):
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            if attachment:
                self.send_header("Content-Disposition", f'attachment; filename="{attachment}"')
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; "
                             "style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; "
                             "connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def json(self, code, data):
            self.send(code, json.dumps(data).encode("utf-8"))

        def valid_host(self):
            port = self.server.server_port
            return self.headers.get("Host") in (f"127.0.0.1:{port}", f"localhost:{port}")

        def do_GET(self):
            if not self.valid_host():
                return self.json(403, {"error": "Use the localhost dashboard address."})
            path = urlsplit(self.path).path
            if path == "/api/state":
                return self.json(200, controller.snapshot())
            if path == "/api/frame":
                with controller.lock:
                    jpeg = controller.jpeg
                return self.send(200 if jpeg else 204, jpeg or b"", "image/jpeg")
            if path == "/api/export":
                return self.send(200, controller.export(), "text/csv; charset=utf-8",
                                 attachment="guidesense-activity.csv")
            assets = {"/": ("index.html", "text/html; charset=utf-8"),
                      "/app.css": ("app.css", "text/css; charset=utf-8"),
                      "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                      "/scene.svg": ("scene.svg", "image/svg+xml"),
                      "/mark.svg": ("mark.svg", "image/svg+xml")}
            if path not in assets:
                return self.json(404, {"error": "Not found"})
            name, mime = assets[path]
            body = (ASSETS / name).read_bytes()
            if path == "/":
                body = body.replace(b"__SESSION_TOKEN__", token.encode())
            self.send(200, body, mime)

        def do_POST(self):
            if not self.valid_host() or not secrets.compare_digest(
                self.headers.get("X-GuideSense-Token", ""), token
            ):
                return self.json(403, {"error": "Refresh the dashboard before trying again."})
            origin = self.headers.get("Origin")
            if origin and origin != f"http://{self.headers.get('Host')}":
                return self.json(403, {"error": "Origin not allowed."})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 <= length <= 4096:
                    return self.json(413, {"error": "Request too large."})
                body = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/api/start":
                    controller.start(body)
                elif self.path == "/api/stop":
                    controller.stop()
                else:
                    return self.json(404, {"error": "Not found"})
            except (ValueError, UnicodeDecodeError) as exc:
                return self.json(400, {"error": str(exc)})
            except SessionBusy as exc:
                return self.json(409, {"error": str(exc)})
            self.json(200, controller.snapshot())

    return Handler


def main():
    parser = argparse.ArgumentParser(description="GuideSense local dashboard")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    controller = SessionController()
    server = ThreadingHTTPServer(("127.0.0.1", args.port),
                                make_handler(controller, secrets.token_hex(24)))
    print(f"GuideSense dashboard: http://127.0.0.1:{server.server_port}")
    print("Open that address in your browser. Ctrl+C stops the server and session.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        controller.shutdown()
        server.server_close()
