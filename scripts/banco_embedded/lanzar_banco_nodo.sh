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
set -u
N="$1"; DESDE="$2"; shift 2
DIRS=("$@")
EST="${DIRS[${#DIRS[@]}-1]}"

# node_ip_dict as the runner wants it: {"1": "...", "2": "...", ...} over every address given.
DICT="{"
for i in "${!DIRS[@]}"; do
    [ "$i" -gt 0 ] && DICT="$DICT,"
    DICT="$DICT\"$((i + 1))\":\"${DIRS[$i]}\""
done
DICT="$DICT}"

cd ~/banco
printf 'node_id = %s\nuav_api_port = 8000\ncontrol_api_port = 8100\ndata_port = 8200\n' "$N" > runner_banco.toml

# The fake autopilot FIRST, because the runner refuses to start without one answering on 8000.
# uav_api_stub.py is the one that simulates movement; uav_api_falso.py may not, and then a drone
# told to go and look from another side never arrives and the orbit times out.
STUB=~/uav_api_stub.py
[ -f "$STUB" ] || STUB=~/uav_api_falso.py
setsid nohup python3 "$STUB" > ~/banco/uav_api.log 2>&1 < /dev/null &
echo "dron $N: piloto falso $(basename "$STUB")"

setsid nohup env PYTHONPATH="$HOME/banco:$HOME/gradys-embedded:$HOME/gradys_protocols" \
    BANCO_DESDE_S="$DESDE" BANCO_ESTACION="http://$EST" \
    python3 -m gradys_embedded.runner.cli --config "$HOME/banco/runner_banco.toml" \
    > ~/banco/runner.log 2>&1 < /dev/null &

for i in $(seq 1 90); do curl -s -o /dev/null localhost:8100/mission/status && break; sleep 1; done
echo "dron $N: control API $(curl -s -o /dev/null -w %{http_code} localhost:8100/mission/status) tras ${i}s"

MISION=$(printf '{"protocol":"mision_banco_dos_drones:ProtocoloVisionBanco","initial_position":[0,0,30],"origin_gps_coordinates":[-22.978029946,-43.23214256266666,0],"x_axis_degrees":0,"node_ip_dict":%s,"communication_protocol":"http","label":"banco_%s_nodos"}' "$DICT" "${#DIRS[@]}")
echo "dron $N load:  $(curl -s -X POST localhost:8100/mission/load -H 'Content-Type: application/json' -d "$MISION" | cut -c1-70)"
echo "dron $N setup: $(curl -s -m 60 -X POST localhost:8100/mission/setup | cut -c1-70)"
echo "dron $N ve la estacion: $(curl -s -m 4 -o /dev/null -w %{http_code} "http://$EST/buscar")"
