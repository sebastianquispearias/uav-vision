"""Safety proxy for bench missions against the REAL uav_api.

Telemetry endpoints are forwarded verbatim to the real uav_api (real Pixhawk
data). Command and movement endpoints are answered locally with success and
NEVER forwarded — a bench mission must not be able to arm motors or command
movement, whatever the runner asks for. This is the honest split: everything
testable on the desk runs real; what physically requires a flight is stubbed.

Runner points at this proxy (port 8001); the real uav_api stays on 8000.
"""

import json
import urllib.request
from http import server

REAL = "http://localhost:8000"


class Handler(server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _json(self, payload, status=200):
        cuerpo = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(cuerpo)))
        self.end_headers()
        self.wfile.write(cuerpo)

    def do_GET(self):
        ruta = self.path.split("?")[0]
        if ruta.startswith("/telemetry/"):
            try:
                with urllib.request.urlopen(REAL + self.path, timeout=3) as r:
                    datos = r.read()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(datos)))
                self.end_headers()
                self.wfile.write(datos)
            except Exception as e:
                self._json({"error": str(e)}, 502)
        elif ruta.startswith("/command/"):
            # Never forwarded: a bench mission must not arm or command RTL.
            self._json({"result": "Success", "info": "simulado en banco"})
        else:
            self._json({"error": f"sin ruta {ruta}"}, 404)

    def do_POST(self):
        largo = int(self.headers.get("Content-Length", 0))
        self.rfile.read(largo)
        ruta = self.path.split("?")[0].rstrip("/")
        if ruta.startswith("/movement"):
            self._json({"result": "Success", "info": "simulado en banco"})
        else:
            self._json({"error": f"sin ruta {ruta}"}, 404)


if __name__ == "__main__":
    print("proxy de banco en :8001 -> telemetria real de :8000, comandos simulados")
    server.ThreadingHTTPServer(("127.0.0.1", 8001), Handler).serve_forever()
