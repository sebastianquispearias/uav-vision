"""Fake uav_api for bench runs: the flight-controller side of the stack, mocked.

Serves the exact endpoints the gradys-embedded runner and UavApiYaw consume, with
a drone that "flies" on the desk: it reports a pose that travels toward whatever
waypoint the runner last commanded, at the last commanded speed. Everything else
in the stack (runner, encapsulator, transport, protocol, camera) is the real
thing -- this is the only mocked piece, because there is no autopilot on the desk.

Why it moves instead of sitting still: MissionMobilityPlugin only advances to the
next waypoint once telemetry says the current one was reached. A fixed pose makes
the mission freeze on leg 1, so a waypoint mission cannot be validated against it.

Every request is logged with its payload, in local metres relative to the mission
origin, so the log alone shows which waypoints left the protocol.

State: arm/takeoff always succeed; after /command/rtl the vehicle flies home and
the reported relative_alt drops to 0 so the runner's landing detector can close
the mission out.
"""

import json
import math
import time
from http import server
from urllib.parse import parse_qs, urlparse

LAT, LON = -22.9793, -43.2325   # PUC-Rio, so the geo math runs on real numbers
HEADING = 0.0                   # camera facing "north" of the local frame
GROUND_ALT = 0.0                # relative_alt reported while landed

# Same flat-earth constants cartesian_to_geo uses, so a waypoint commanded at
# x metres north is logged back as x metres north and not x plus a rounding.
M_PER_DEG_LAT = 111320.0
M_PER_DEG_LON = 111320.0 * math.cos(math.radians(LAT))

estado = {
    "lat": LAT, "lon": LON, "alt": GROUND_ALT,
    "target": None,          # (lat, lon, alt) or None while holding position
    "speed": 5.0,            # m/s, overwritten by /command/set_air_speed
    "t": time.monotonic(),
}


def _local(lat, lon, alt):
    """Waypoint in metres (north, east, up) from the mission origin."""
    return ((lat - LAT) * M_PER_DEG_LAT, (lon - LON) * M_PER_DEG_LON, alt)


def _log(texto):
    print(f"[{time.strftime('%H:%M:%S')}] {texto}", flush=True)


def _advance():
    """Move the reported pose toward the target for the time actually elapsed."""
    ahora = time.monotonic()
    dt, estado["t"] = ahora - estado["t"], ahora
    destino = estado["target"]
    if destino is None:
        return

    dn = (destino[0] - estado["lat"]) * M_PER_DEG_LAT
    de = (destino[1] - estado["lon"]) * M_PER_DEG_LON
    du = destino[2] - estado["alt"]
    falta = math.sqrt(dn * dn + de * de + du * du)

    paso = estado["speed"] * dt
    if falta <= paso or falta == 0.0:
        estado["lat"], estado["lon"], estado["alt"] = destino
        estado["target"] = None
        _log(f"    llegado a {tuple(round(v, 1) for v in _local(*destino))}")
        return

    f = paso / falta
    estado["lat"] += (dn * f) / M_PER_DEG_LAT
    estado["lon"] += (de * f) / M_PER_DEG_LON
    estado["alt"] += du * f


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
        partes = urlparse(self.path)
        ruta = partes.path.rstrip("/") or "/"
        args = parse_qs(partes.query)

        if ruta == "/telemetry/gps":
            # The only endpoint polled in a loop, so it is the one that ticks
            # the pose forward. Not logged: 2 Hz would bury the waypoints.
            _advance()
            self._json({"info": {
                "position": {"lat": estado["lat"], "lon": estado["lon"],
                             "relative_alt": estado["alt"]},
                "heading": HEADING,
            }})
        elif ruta == "/telemetry/general":
            # Same shape as the real uav_api: heading nested under "info".
            self._json({"result": "Success", "info": {"heading": HEADING}})
        elif ruta == "/command/arm":
            _log("ARMA motores")
            self._json({"status": "armed"})
        elif ruta == "/command/takeoff":
            alt = float(args.get("alt", ["0"])[0])
            _log(f"DESPEGA a {alt} m")
            estado["target"] = (estado["lat"], estado["lon"], alt)
            self._json({"status": "airborne"})
        elif ruta == "/command/rtl":
            _log("RTL: vuelve a casa y aterriza")
            estado["target"] = (LAT, LON, GROUND_ALT)
            self._json({"status": "landed"})
        elif ruta == "/command/set_air_speed":
            estado["speed"] = float(args.get("new_v", [estado["speed"]])[0])
            _log(f"velocidad {estado['speed']} m/s")
            self._json({"status": "ok"})
        else:
            _log(f"RUTA DESCONOCIDA GET {ruta}")
            self._json({"error": f"sin ruta {ruta}"}, 404)

    def do_POST(self):
        largo = int(self.headers.get("Content-Length", 0))
        crudo = self.rfile.read(largo)
        ruta = urlparse(self.path).path.rstrip("/")

        if ruta in ("/movement/go_to_gps", "/movement/go_to_gps_wait"):
            d = json.loads(crudo or b"{}")
            destino = (d["lat"], d["long"], d["alt"])
            estado["target"] = destino
            n, e, u = (round(v, 1) for v in _local(*destino))
            _log(f"WAYPOINT -> norte {n} m, este {e} m, altura {u} m")
            self._json({"status": "ok"})
        else:
            _log(f"RUTA DESCONOCIDA POST {ruta}")
            self._json({"error": f"sin ruta {ruta}"}, 404)


if __name__ == "__main__":
    _log("uav_api falso en :8000 (dron de escritorio, vuela sobre el papel)")
    server.ThreadingHTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
