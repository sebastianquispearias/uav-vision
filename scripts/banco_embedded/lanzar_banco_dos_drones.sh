#!/usr/bin/env bash
# Starts one board of the two-drone desk bench, up to a mission that is loaded and set up.
#
# Runs on each Raspberry: the fake uav_api (so the runner can start without an autopilot), the
# gradys-embedded runner, and POST /mission/load + /mission/setup for mision_banco_dos_drones.
# It stops short of /mission/start on purpose: both boards are started together, afterwards, so
# their replays of the two takeoffs overlap. See mision_banco_dos_drones.py for what is replayed.
#
# Expects, in the home of user pi: ~/banco (uav_vision package + datos/), ~/gradys-embedded,
# ~/uav_api_falso.py, and the mission module in ~/gradys_protocols.
#
#   bash -s -- <node_id> <desde_s> <estacion ip:puerto> <nodo1 ip:8200> <nodo2 ip:8200> < lanzar_banco_dos_drones.sh
#   e.g. node 1: 1 0   10.237.186.80:8300 10.237.186.141:8200 10.237.186.186:8200
#        node 2: 2 700 10.237.186.80:8300 10.237.186.141:8200 10.237.186.186:8200
set -u
N="$1"; DESDE="$2"; ESTACION="$3"; NODO1="$4"; NODO2="$5"
cd ~/banco
printf 'node_id = %s\nuav_api_port = 8000\ncontrol_api_port = 8100\ndata_port = 8200\n' "$N" > runner_banco.toml
setsid nohup python3 ~/uav_api_falso.py > ~/banco/uav_api_falso.log 2>&1 < /dev/null &
setsid nohup env PYTHONPATH="$HOME/banco:$HOME/gradys-embedded:$HOME/gradys_protocols" \
    BANCO_DESDE_S="$DESDE" BANCO_ESTACION="http://$ESTACION" \
    python3 -m gradys_embedded.runner.cli --config "$HOME/banco/runner_banco.toml" \
    > ~/banco/runner.log 2>&1 < /dev/null &
for i in $(seq 1 90); do curl -s -o /dev/null localhost:8100/mission/status && break; sleep 1; done
echo "dron $N: control API $(curl -s -o /dev/null -w %{http_code} localhost:8100/mission/status) after ${i}s"
MISION=$(printf '{"protocol":"mision_banco_dos_drones:ProtocoloVisionBanco","initial_position":[0,0,30],"origin_gps_coordinates":[-22.978029946,-43.23214256266666,0],"x_axis_degrees":0,"node_ip_dict":{"1":"%s","2":"%s","3":"%s"},"communication_protocol":"http","label":"banco_dos_drones"}' "$NODO1" "$NODO2" "$ESTACION")
echo "load:  $(curl -s -X POST localhost:8100/mission/load -H 'Content-Type: application/json' -d "$MISION" | cut -c1-80)"
echo "setup: $(curl -s -m 60 -X POST localhost:8100/mission/setup | cut -c1-80)"
echo "station reachable from here: $(curl -s -m 4 -o /dev/null -w %{http_code} "http://$ESTACION/buscar")"
