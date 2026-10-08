"""Find out which runtime request actually makes EXTENDED_STATUS flow.

The stream-rate parameter alone is latched when the channel initialises, so a
value written while the autopilot is already running may never be re-read. This
walks two runtime requests one at a time, counting SYS_STATUS after each, so the
mechanism that works is identified by observation instead of by firmware lore.
"""

import time
from collections import Counter

from pymavlink import mavutil

TGT_SYS, TGT_COMP = 3, 1
VENTANA = 8.0
TESTIGOS = ("SYS_STATUS", "POWER_STATUS", "MEMINFO", "GPS_RAW_INT", "MISSION_CURRENT")


def contar(conn, etiqueta):
    counts = Counter()
    t0 = time.time()
    while time.time() - t0 < VENTANA:
        msg = conn.recv_match(blocking=True, timeout=1.0)
        if msg is not None:
            counts[msg.get_type()] += 1
    dur = time.time() - t0
    linea = "  ".join(f"{n}={counts.get(n, 0)}" for n in TESTIGOS)
    print(f"{etiqueta:<34}{linea}   (total {sum(counts.values())} en {dur:.1f} s)")
    return counts


m = mavutil.mavlink_connection("udpout:127.0.0.1:14550", source_system=200, source_component=190)
m.mav.heartbeat_send(6, 8, 0, 0, 0)
m.wait_heartbeat(timeout=15)

contar(m, "0. tal como esta")

m.mav.command_long_send(
    TGT_SYS, TGT_COMP, mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
    0, mavutil.mavlink.MAVLINK_MSG_ID_SYS_STATUS, 500000, 0, 0, 0, 0, 0,
)
ack = m.recv_match(type="COMMAND_ACK", blocking=True, timeout=4.0)
print(f"1. SET_MESSAGE_INTERVAL -> ack={getattr(ack, 'result', 'sin ack')}")
contar(m, "1. tras SET_MESSAGE_INTERVAL")

m.mav.request_data_stream_send(
    TGT_SYS, TGT_COMP, mavutil.mavlink.MAV_DATA_STREAM_EXTENDED_STATUS, 2, 1,
)
print("2. REQUEST_DATA_STREAM(EXTENDED_STATUS, 2 Hz) enviado")
contar(m, "2. tras REQUEST_DATA_STREAM")
