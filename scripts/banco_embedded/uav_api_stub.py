"""Fake uav_api for bench runs: the flight-controller side of the stack, mocked.

Serves the exact endpoints the gradys-embedded runner, UavApiYaw and UavApiBattery consume, with
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

THE ORIGIN HAS TO BE THE MISSION'S. The local metres of this fake autopilot are measured from
it, and if the two disagree every waypoint this file logs is offset by the distance between
them and reads as a wild number: commanded 30 m from a target, the log once said
'llegado a (112.0, 51.1)' because the two origins were 120 m apart. The default is PUC-Rio so
the geo maths runs on real numbers when nobody says otherwise, and the bench launcher passes the
mission's. The flat-earth constants are the ones cartesian_to_geo uses, so a waypoint commanded
at x metres north is logged back as x metres north and not x plus a rounding.

The fake camera faces "north" of the local frame, and the altitude reported while landed is the
relative one. The destination is (lat, lon, alt) or None while holding position, and the speed
is in m/s, overwritten by /command/set_air_speed.

/telemetry/general is THE ONLY ENDPOINT POLLED IN A LOOP, so it is the one that ticks the pose
forward. It is not logged, because 2 Hz would bury the waypoints. Its shape is the real
uav_api's, with the heading nested under "info".
"""

import json
import math
import os
import sys
import time
from http import server
from urllib.parse import parse_qs, urlparse

_ORIGEN = os.environ.get("UAV_API_ORIGEN", "")
try:
    LAT, LON = (float(v) for v in _ORIGEN.split(",")[:2])
except (ValueError, TypeError):
    LAT, LON = -22.9793, -43.2325
HEADING = 0.0
GROUND_ALT = 0.0

M_PER_DEG_LAT = 111320.0
M_PER_DEG_LON = 111320.0 * math.cos(math.radians(LAT))

BATERIA_LLENA_MV = 16800.0
BATERIA_VACIA_MV = 14000.0
BATERIA_MINUTOS = 20.0

estado = {
    "lat": LAT, "lon": LON, "alt": GROUND_ALT,
    "target": None,
    "speed": 5.0,
    "t": time.monotonic(),
    "t0": time.monotonic(),
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


def _bateria():
    """A pack that drains, in the shape the real endpoint returns it.

    The voltage is MAVLink's, in MILLIVOLTS, because that is what uav_api forwards out of
    SYS_STATUS and a stub that answered in volts would hide a unit bug instead of exposing it:
    the consumer divides by a thousand, and against a stub serving 15.0 the bench would read
    0.015 V and nobody would know whether the fault was here or on the aircraft.

    It falls instead of sitting still for the same reason the fake pose moves: a constant is
    indistinguishable from a field that is never updated. Twenty minutes from full to the point
    where a 4S pack should already be on the ground, which is roughly a real endurance, so a
    long bench run walks the station's warning thresholds instead of parking above them.
    """
    t = min(1.0, (time.monotonic() - estado["t0"]) / (BATERIA_MINUTOS * 60.0))
    mv = BATERIA_LLENA_MV + t * (BATERIA_VACIA_MV - BATERIA_LLENA_MV)
    return {"voltage": int(round(mv)),
            "current": 1200,
            "battery_remaining": int(round(100 * (1.0 - t)))}


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
            _advance()
            self._json({"info": {
                "position": {"lat": estado["lat"], "lon": estado["lon"],
                             "relative_alt": estado["alt"]},
                "heading": HEADING,
            }})
        elif ruta == "/telemetry/general":
            self._json({"result": "Success", "info": {"heading": HEADING}})
        elif ruta == "/telemetry/battery_info":
            self._json({"result": "success", "info": _bateria()})
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
    # The port is 8000 because that is what the bench launcher and runner_banco.toml agree on,
    # and it is overridable only so a gate can run this against a free port without racing a
    # bench that is already up.
    puerto = 8000
    if "--puerto" in sys.argv:
        puerto = int(sys.argv[sys.argv.index("--puerto") + 1])
    _log(f"uav_api falso en :{puerto} (dron de escritorio, vuela sobre el papel)")
    server.ThreadingHTTPServer(("127.0.0.1", puerto), Handler).serve_forever()
