"""Ground-station stand-in: listens on the fleet data plane and prints POIs.

Occupies node 2 in the mission's node_ip_dict, so the drone's broadcast reaches
it over the real HTTP transport — the same wire format any GrADyS node uses.
Prints every vision_poi as it lands.
"""

import json
from datetime import datetime
from http import server


class Handler(server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        largo = int(self.headers.get("Content-Length", 0))
        crudo = self.rfile.read(largo)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status": "ok"}')

        try:
            payload = json.loads(crudo)
            mensaje = json.loads(payload["message"])
        except Exception:
            print("mensaje no-JSON:", crudo[:200])
            return

        hora = datetime.now().strftime("%H:%M:%S")
        if mensaje.get("type") == "vision_poi":
            pois = mensaje.get("pois", [])
            print(f"[{hora}] vision_poi del dron {payload.get('source')} | "
                  f"frames vistos: {mensaje.get('frames_seen')} | {len(pois)} POI(s)")
            for p in pois:
                movil = p.get("movil")
                etiqueta = ("MOVIL " if movil else "estatico " if movil is not None
                            else "consenso ")
                print(f"    {etiqueta} x={p.get('x')} y={p.get('y')} "
                      f"n_obs={p.get('n_obs')} conf={p.get('conf', p.get('conf_mean'))}")
        else:
            print(f"[{hora}] mensaje:", mensaje)


if __name__ == "__main__":
    print("Ground Station de banco escuchando en :8300 — esperando vision_poi ...")
    server.ThreadingHTTPServer(("0.0.0.0", 8300), Handler).serve_forever()
