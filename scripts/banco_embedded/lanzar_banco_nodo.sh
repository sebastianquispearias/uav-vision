#!/usr/bin/env bash
# Starts ONE board of the desk bench, for any number of boards, up to a mission that is set up.
#
# Generalises lanzar_banco_dos_drones.sh, which hardwired two drones plus the station into
# node_ip_dict. Here the whole dictionary is passed in, so the same script serves one board or
# five, and the caller is levantar_banco.sh on the laptop rather than a human in three terminals.
#
# It stops short of /mission/start on purpose, exactly as the two-drone version did: every board
# is set up first and then they are all started together, so their replays of the recording line
# up. Starting them one at a time means each drone is looking at a different moment of the flight
# and there is nothing to fuse.
#
# Expects, in the home of user pi: ~/banco (the uav_vision package plus datos/), ~/gradys-embedded,
# ~/uav_api_falso.py, and the mission module in ~/gradys_protocols.
#
#   ssh pi@<board> 'bash -s' -- <node_id> <desde_s> <n1=ip:8200> <n2=ip:8200> ... <est=ip:8300> \
#       < lanzar_banco_nodo.sh
#
# The node ids in the dictionary are the positions of the addresses given, starting at 1, so the
# station is simply the last one. Example with three boards and the station:
#   ssh pi@10.0.0.11 'bash -s' -- 1 0   10.0.0.11:8200 10.0.0.12:8200 10.0.0.13:8200 10.0.0.9:8300
# The third argument is which mission to load. The one that replays the recording needs no camera
# and gives the same scene every run, which is what a gate wants. mision_vision:ProtocoloVisionLAC
# uses the REAL camera instead: live detection on whatever the board is pointed at. With it the
# detection is real and the position is NOT, because a board on a desk has no autopilot, so the
# metres on the map come from the fake one and mean nothing. Worth saying out loud in a demo.
#
# THE ARGUMENTS AND THE ORDER OF START-UP, AND WHY EACH IS AS IT IS
#
# The fourth argument is extra environment for the mission, "KEY=value KEY=value" or the
# sentinel. It exists because the two boards of this bench are NOT the same machine: the
# appearance model costs several times more on one than on the other, measured in NOTES.md, so
# turning it on for both at the same rate is not a decision, it is an oversight. Whatever is
# passed here reaches the mission as environment variables.
#
# THE SENTINEL "-" MEANS "NO EXTRA ENVIRONMENT", and an empty string cannot be used: ssh joins
# its arguments into ONE command and the remote shell parses that again, so an empty argument
# DISAPPEARS and every argument behind it shifts one place. Verified by sending seven arguments
# and receiving six. The symptom is env trying to execute an address:
#     env: 192.168.1.125:8200: No such file or directory
# which looks like an argument error in the launcher and says nothing about the empty one.
#
# The assignments arrive COMMA-separated and not space-separated, and are split again here.
# An argument with spaces does not survive ssh either: the command is re-parsed on the board and
# what was ONE argument arrives as several, so the shift eats an address and the node dictionary
# loses an entry. With two variables that left the STATION out of the map, and then the drone
# broadcasts to its neighbours and never to the ground: a symptom that reads as "the protocol
# does not report" and is a badly delivered argument.
#
# node_ip_dict as the runner wants it: {"1": "...", "2": "...", ...} over every address given.
#
# Whatever is left from a previous session goes first, and then we WAIT for the ports. A stub that
# is still holding 8000 makes the new one die on bind, and the old one keeps answering with the old
# configuration: the bench then runs on settings nobody passed and the only clue is a log that
# prints waypoints in the wrong frame. Measured the hard way.
#
# The wait pattern ends in a SPACE and not in a word boundary. A backslash-b written through
# several layers becomes a literal backspace, the pattern then matches nothing ever, and this
# loop exits on its first turn so the wait does not wait. It was like that until a sweep for
# control characters found it.
#
# The fake autopilot FIRST, because the runner refuses to start without one answering on 8000.
# uav_api_stub.py is the one that simulates movement; uav_api_falso.py may not, and then a drone
# told to go and look from another side never arrives and the orbit times out.
#
# The stub measures its local metres from this origin, and it has to be the mission's or every
# waypoint it logs is offset by the distance between the two.
#
# And WAIT for it to answer before the runner starts. Without this the runner comes up first, its
# first telemetry fetch fails with 'Cannot connect to host localhost:8000', it asks for an RTL that
# also fails, and it shuts itself down. The mission looks loaded and set up and then the board is
# simply gone, with the camera having opened correctly a second earlier, which sends you looking at
# the camera.
#
# A runner that is already running a mission refuses to load another, and the refusal looks like a
# launcher bug: 'Cannot load a mission while running'. Asking it to stop first is idempotent and
# costs nothing when there is nothing to stop.
set -u
N="$1"; DESDE="$2"; MISION_CLASE="${3:-mision_banco_dos_drones:ProtocoloVisionBanco}"
EXTRA="${4:-}"; shift 4
[ "$EXTRA" = "-" ] && EXTRA=""
EXTRA="$(printf "%s" "$EXTRA" | tr "," " ")"
DIRS=("$@")
EST="${DIRS[${#DIRS[@]}-1]}"

DICT="{"
for i in "${!DIRS[@]}"; do
    [ "$i" -gt 0 ] && DICT="$DICT,"
    DICT="$DICT\"$((i + 1))\":\"${DIRS[$i]}\""
done
DICT="$DICT}"

cd ~/banco
pkill -f gradys_embedded.runner.cli 2>/dev/null || true
pkill -f uav_api_stub 2>/dev/null || true
pkill -f uav_api_falso 2>/dev/null || true
for i in $(seq 1 20); do
    ss -ltn 2>/dev/null | grep -qE ':(8000|8100|8200)[[:space:]]' || break
    sleep 1
done
printf 'node_id = %s\nuav_api_port = 8000\ncontrol_api_port = 8100\ndata_port = 8200\n' "$N" > runner_banco.toml

STUB=~/uav_api_stub.py
[ -f "$STUB" ] || STUB=~/uav_api_falso.py
ORIGEN="-22.978029946,-43.23214256266666"
setsid nohup env UAV_API_ORIGEN="$ORIGEN" python3 "$STUB" > ~/banco/uav_api.log 2>&1 < /dev/null &
echo "dron $N: piloto falso $(basename "$STUB")"

for i in $(seq 1 30); do
    curl -s -o /dev/null -m 2 localhost:8000/telemetry/gps && break
    sleep 1
done
echo "dron $N: piloto falso responde tras ${i}s"

[ -n "$EXTRA" ] && echo "dron $N entorno: $EXTRA"
# shellcheck disable=SC2086  -- EXTRA is a list of assignments and must be word-split
setsid nohup env PYTHONPATH="$HOME/banco:$HOME/gradys-embedded:$HOME/gradys_protocols" \
    BANCO_DESDE_S="$DESDE" BANCO_ESTACION="http://$EST" $EXTRA \
    python3 -m gradys_embedded.runner.cli --config "$HOME/banco/runner_banco.toml" \
    > ~/banco/runner.log 2>&1 < /dev/null &

for i in $(seq 1 90); do curl -s -o /dev/null localhost:8100/mission/status && break; sleep 1; done
echo "dron $N: control API $(curl -s -o /dev/null -w %{http_code} localhost:8100/mission/status) tras ${i}s"

curl -s -m 10 -X POST localhost:8100/mission/stop > /dev/null 2>&1 || true
sleep 1

echo "dron $N mision: $MISION_CLASE"
MISION=$(printf '{"protocol":"%s","initial_position":[0,0,30],"origin_gps_coordinates":[-22.978029946,-43.23214256266666,0],"x_axis_degrees":0,"node_ip_dict":%s,"communication_protocol":"http","label":"banco_%s_nodos"}' "$MISION_CLASE" "$DICT" "${#DIRS[@]}")
echo "dron $N load:  $(curl -s -X POST localhost:8100/mission/load -H 'Content-Type: application/json' -d "$MISION" | cut -c1-70)"
echo "dron $N setup: $(curl -s -m 60 -X POST localhost:8100/mission/setup | cut -c1-70)"
echo "dron $N ve la estacion: $(curl -s -m 4 -o /dev/null -w %{http_code} "http://$EST/buscar")"
