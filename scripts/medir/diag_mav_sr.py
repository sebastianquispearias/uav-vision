"""Read-only MAVLink probe for the stream-rate diagnosis.

Connects to the mavlink-router UDP server endpoint as an extra GCS client, so it
does not compete with uav_api for the 14552 socket. It counts every message type
that actually arrives over a fixed window and then reads the SRx_* stream-rate
parameters for all four serial channels. Nothing is written to the autopilot.
"""

import sys
import time
from collections import Counter

from pymavlink import mavutil

WINDOW_S = 15.0
PARAMS = [
    f"SR{ch}_{grp}"
    for ch in range(4)
    for grp in ("EXT_STAT", "EXTRA1", "EXTRA2", "EXTRA3", "POSITION", "RAW_SENS")
]

m = mavutil.mavlink_connection("udpout:127.0.0.1:14550", source_system=200, source_component=190)
m.mav.heartbeat_send(6, 8, 0, 0, 0)  # announce ourselves as a GCS so the router registers us
hb = m.wait_heartbeat(timeout=15)
if hb is None:
    print("SIN HEARTBEAT en 15 s por 14550")
    sys.exit(1)
tgt_sys, tgt_comp = m.target_system, m.target_component
print(f"heartbeat de sysid={tgt_sys} compid={tgt_comp} tipo={hb.type} autopilot={hb.autopilot}")

print(f"\n=== lo que LLEGA en {WINDOW_S:.0f} s (sin pedir nada) ===")
counts = Counter()
t0 = time.time()
while time.time() - t0 < WINDOW_S:
    msg = m.recv_match(blocking=True, timeout=1.0)
    if msg is not None:
        counts[msg.get_type()] += 1
dur = time.time() - t0
print(f"{'mensaje':<26}{'paquetes':>9}{'Hz':>8}")
for name, n in counts.most_common():
    print(f"{name:<26}{n:>9}{n / dur:>8.2f}")
print(f"total {sum(counts.values())} paquetes en {dur:.1f} s")

print("\n=== las tasas de stream configuradas ===")
got = {}
for name in PARAMS:
    m.mav.param_request_read_send(tgt_sys, tgt_comp, name.encode(), -1)
deadline = time.time() + 12.0
while time.time() < deadline and len(got) < len(PARAMS):
    msg = m.recv_match(type="PARAM_VALUE", blocking=True, timeout=1.0)
    if msg is None:
        continue
    pid = msg.param_id.rstrip("\x00") if isinstance(msg.param_id, str) else msg.param_id.decode().rstrip("\x00")
    if pid in PARAMS:
        got[pid] = msg.param_value

print(f"{'parametro':<18}{'valor':>8}")
for name in PARAMS:
    v = got.get(name)
    print(f"{name:<18}{'(sin respuesta)' if v is None else f'{v:>8.0f}'}")
