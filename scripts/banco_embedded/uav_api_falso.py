"""Fake uav_api for bench runs: the flight-controller side of the stack, mocked.

Serves the exact endpoints the gradys-embedded runner and UavApiYaw consume, with
the drone "hovering" at a fixed pose on the desk. Everything else in the stack
(runner, encapsulator, transport, protocol, camera) is the real thing — this is
the only mocked piece, because there is no autopilot on the desk.

State: arm/takeoff always succeed; after /command/rtl the reported relative_alt
drops to 0 so the runner's landing detector can close the mission out.
"""

import json
from http import server

LAT, LON = -22.9793, -43.2325   # PUC-Rio, so the geo math runs on real numbers
ALT_VUELO = 1.2                 # desk-mounted camera height, meters
HEADING = 0.0                   # camera facing "north" of the local frame

estado = {"alt": ALT_VUELO}


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
        if ruta == "/telemetry/gps":
            self._json({"info": {
                "position": {"lat": LAT, "lon": LON,
                             "relative_alt": estado["alt"]},
                "heading": HEADING,
            }})
        elif ruta == "/telemetry/general":
            # Same shape as the real uav_api: heading nested under "info".
            self._json({"result": "Success", "info": {"heading": HEADING}})
        elif ruta == "/command/arm":
            self._json({"status": "armed"})
        elif ruta == "/command/takeoff":
            estado["alt"] = ALT_VUELO
            self._json({"status": "airborne"})
        elif ruta == "/command/rtl":
            estado["alt"] = 0.0
            self._json({"status": "landed"})
        elif ruta == "/command/set_air_speed":
            self._json({"status": "ok"})
        else:
            self._json({"error": f"sin ruta {ruta}"}, 404)

    def do_POST(self):
        largo = int(self.headers.get("Content-Length", 0))
        self.rfile.read(largo)
        ruta = self.path.split("?")[0].rstrip("/")
        if ruta in ("/movement/go_to_gps", "/movement/go_to_gps_wait"):
            self._json({"status": "ok"})
        else:
            self._json({"error": f"sin ruta {ruta}"}, 404)


if __name__ == "__main__":
    print("uav_api falso en :8000 (dron posado en el escritorio)")
    server.ThreadingHTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
