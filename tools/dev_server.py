"""Local stand-in for the whole setup: serves the page, the corpus and a vote endpoint
that behaves like the Apps Script one (password "test"; votes appended to votes.tsv).

    pixi run serve            # http://192.168.1.211:8770 (pass another host/port: serve HOST PORT)
"""
from __future__ import annotations

import json
import sys
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
PASSWORD = "test"
VOTES = ROOT / "votes.tsv"
HOST = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.211"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8770


class Handler(SimpleHTTPRequestHandler):
    def translate_path(self, path: str) -> str:
        p = urlparse(path).path
        if p.startswith("/corpus/"):
            return str(ROOT / "corpus" / p[len("/corpus/"):])
        return str(ROOT / "docs" / (p.lstrip("/") or "index.html"))

    def _json(self, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        u = urlparse(self.path)
        if u.path == "/config.js":  # point the page back at whatever address it was opened with
            base = f"http://{self.headers['Host']}"
            body = f'export const CORPUS_URL = "{base}/corpus/";\nexport const VOTE_URL = "{base}/vote";\n'.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/javascript")
            self.end_headers()
            self.wfile.write(body)
        elif u.path == "/vote":
            self._json({"ok": parse_qs(u.query).get("pw", [""])[0] == PASSWORD})
        else:
            super().do_GET()

    def do_POST(self) -> None:
        d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        ok = d.get("pw") == PASSWORD and d.get("vote") in ("up", "down")
        if ok:
            new = not VOTES.exists()
            with VOTES.open("a") as fh:
                if new:
                    fh.write("time\titem\tvote\tvoter\tdevice\tbatch\n")
                fh.write("\t".join([time.strftime("%Y-%m-%d %H:%M:%S"), d["item"], d["vote"], d["voter"], d["device"], d["batch"]]) + "\n")
        self._json({"ok": ok})

    def log_message(self, *args) -> None:
        pass


if __name__ == "__main__":
    print(f"http://{HOST}:{PORT}")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
