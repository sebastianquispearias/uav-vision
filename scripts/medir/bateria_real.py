"""Read the real pack straight off the MAVLink bus, bypassing uav_api entirely.

BATTERY_STATUS rides in ArduPilot's EXTRA3 stream and arrives at 2 Hz on this link whether or
not uav_api is running, so the aircraft's own voltage can be read while the bench holds port
8000 with its fake autopilot. The two numbers are deliberately not mixed: this one is the
battery, the one on the bench screen is a simulation.
"""

import time

from pymavlink import mavutil

m = mavutil.mavlink_connection("udpout:127.0.0.1:14550", source_system=201, source_component=191)
m.mav.heartbeat_send(6, 8, 0, 0, 0)
m.wait_heartbeat(timeout=15)

leidas = []
t0 = time.time()
while time.time() - t0 < 10.0 and len(leidas) < 6:
    msg = m.recv_match(type="BATTERY_STATUS", blocking=True, timeout=2.0)
    if msg is None:
        continue
    mv = msg.voltages[0]
    if mv in (0, 65535):
        continue
    leidas.append((mv / 1000.0, msg.current_battery / 100.0, msg.battery_remaining))

if not leidas:
    print("no llego ni un BATTERY_STATUS con tension valida")
    raise SystemExit(1)

print("%-10s %-12s %-12s %s" % ("volts", "V/celda(4S)", "amperios", "restante"))
for v, a, pct in leidas:
    print("%-10.3f %-12.3f %-12.2f %s %%" % (v, v / 4.0, a, pct))
v = sum(x[0] for x in leidas) / len(leidas)
print()
print("PACK REAL: %.3f V = %.3f V por celda sobre 4S" % (v, v / 4.0))
print("umbrales:  3.50 volver   3.30 aterrizar")
print("estado:    %s" % ("ATERRIZAR" if v / 4 <= 3.30 else
                         ("VOLVER" if v / 4 <= 3.50 else "bien")))
