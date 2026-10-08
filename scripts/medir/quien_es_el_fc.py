"""Identify every MAVLink heartbeat source behind the router.

Several components announce themselves on the shared bus (the autopilot, the
router's own clients and this probe), so a parameter write has to be addressed to
the one that reports MAV_AUTOPILOT_ARDUPILOTMEGA rather than to whichever
heartbeat arrives first.
"""

import time
from collections import Counter

from pymavlink import mavutil

AUTOPILOT = {0: "GENERIC", 3: "ARDUPILOTMEGA", 8: "INVALID"}

m = mavutil.mavlink_connection("udpout:127.0.0.1:14550", source_system=200, source_component=190)
m.mav.heartbeat_send(6, 8, 0, 0, 0)
m.wait_heartbeat(timeout=15)

seen = Counter()
t0 = time.time()
while time.time() - t0 < 8.0:
    msg = m.recv_match(type="HEARTBEAT", blocking=True, timeout=1.0)
    if msg is None:
        continue
    seen[(msg.get_srcSystem(), msg.get_srcComponent(), msg.type, msg.autopilot)] += 1

print(f"{'sysid':>6}{'compid':>8}{'type':>6}  {'autopilot':<16}{'latidos':>8}")
for (s, c, t, a), n in sorted(seen.items()):
    print(f"{s:>6}{c:>8}{t:>6}  {AUTOPILOT.get(a, str(a)):<16}{n:>8}")
