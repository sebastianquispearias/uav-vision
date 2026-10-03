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
set -u
N="$1"; DESDE="$2"; MISION_CLASE="${3:-mision_banco_dos_drones:ProtocoloVisionBanco}"
# Cuarto argumento: entorno extra para la mision, "CLAVE=valor CLAVE=valor" o vacio. Existe
# porque las dos placas de este banco NO son la misma maquina: el modelo de apariencia cuesta
# unos 30 ms por caja en la Pi 5 y entre 170 y 330 en la Pi 4 (medido el 3oct con
# medir_osnet.py), asi que encenderlo en las dos al mismo ritmo no es una decision, es un
# descuido. Lo que se pasa aqui llega a la mision como variables de entorno.
# El centinela "-" significa "sin entorno extra". NO se puede pasar una cadena vacia por ssh:
# ssh une sus argumentos en UNA orden y el shell remoto la vuelve a parsear, asi que un
# argumento vacio desaparece y todos los siguientes se corren un lugar. Comprobado el 3oct
# mandando 7 argumentos y recibiendo 6. El sintoma es env intentando ejecutar una direccion:
#     env: 192.168.1.125:8200: No such file or directory
# que parece un error de argumentos del lanzador y no dice nada del argumento vacio.
EXTRA="${4:-}"; shift 4
[ "$EXTRA" = "-" ] && EXTRA=""
# Las asignaciones vienen separadas por COMAS, no por espacios, y aqui se vuelven a separar.
# Un argumento con espacios tampoco sobrevive a ssh: la orden se vuelve a parsear en la placa y
# lo que era UN argumento llega como varios, asi que shift 4 se come una direccion y el
# diccionario de nodos pierde una entrada. Con dos variables eso dejaba a la ESTACION fuera del
# mapa, y entonces el dron difunde a sus vecinos y nunca a tierra: un sintoma que se lee como
# "el protocolo no reporta" y es un argumento mal entregado.
EXTRA="$(printf "%s" "$EXTRA" | tr "," " ")"
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
# Whatever is left from a previous session goes first, and then we WAIT for the ports. A stub that
# is still holding 8000 makes the new one die on bind, and the old one keeps answering with the old
# configuration: the bench then runs on settings nobody passed and the only clue is a log that
# prints waypoints in the wrong frame. Measured the hard way.
pkill -f gradys_embedded.runner.cli 2>/dev/null || true
pkill -f uav_api_stub 2>/dev/null || true
pkill -f uav_api_falso 2>/dev/null || true
for i in $(seq 1 20); do
    # El patron termina en un espacio y no en un limite de palabra: un \b escrito a traves de
    # varias capas se convierte en un retroceso literal, el patron deja de coincidir nunca, y
    # entonces este bucle sale en la primera vuelta y la espera no espera. Estuvo asi hasta el
    # 3oct, cuando un barrido de caracteres de control lo encontro.
    ss -ltn 2>/dev/null | grep -qE ':(8000|8100|8200)[[:space:]]' || break
    sleep 1
done
printf 'node_id = %s\nuav_api_port = 8000\ncontrol_api_port = 8100\ndata_port = 8200\n' "$N" > runner_banco.toml

# The fake autopilot FIRST, because the runner refuses to start without one answering on 8000.
# uav_api_stub.py is the one that simulates movement; uav_api_falso.py may not, and then a drone
# told to go and look from another side never arrives and the orbit times out.
STUB=~/uav_api_stub.py
[ -f "$STUB" ] || STUB=~/uav_api_falso.py
# The stub measures its local metres from this origin, and it has to be the mission's or every
# waypoint it logs is offset by the distance between the two.
ORIGEN="-22.978029946,-43.23214256266666"
setsid nohup env UAV_API_ORIGEN="$ORIGEN" python3 "$STUB" > ~/banco/uav_api.log 2>&1 < /dev/null &
echo "dron $N: piloto falso $(basename "$STUB")"

# And WAIT for it to answer before the runner starts. Without this the runner comes up first, its
# first telemetry fetch fails with 'Cannot connect to host localhost:8000', it asks for an RTL that
# also fails, and it shuts itself down. The mission looks loaded and set up and then the board is
# simply gone, with the camera having opened correctly a second earlier, which sends you looking at
# the camera.
for i in $(seq 1 30); do
    curl -s -o /dev/null -m 2 localhost:8000/telemetry/gps && break
    sleep 1
done
echo "dron $N: piloto falso responde tras ${i}s"

[ -n "$EXTRA" ] && echo "dron $N entorno: $EXTRA"
# shellcheck disable=SC2086  -- $EXTRA es una lista de asignaciones y debe partirse
setsid nohup env PYTHONPATH="$HOME/banco:$HOME/gradys-embedded:$HOME/gradys_protocols" \
    BANCO_DESDE_S="$DESDE" BANCO_ESTACION="http://$EST" $EXTRA \
    python3 -m gradys_embedded.runner.cli --config "$HOME/banco/runner_banco.toml" \
    > ~/banco/runner.log 2>&1 < /dev/null &

for i in $(seq 1 90); do curl -s -o /dev/null localhost:8100/mission/status && break; sleep 1; done
echo "dron $N: control API $(curl -s -o /dev/null -w %{http_code} localhost:8100/mission/status) tras ${i}s"

# A runner that is already running a mission refuses to load another, and the refusal looks like a
# launcher bug: 'Cannot load a mission while running'. Asking it to stop first is idempotent and
# costs nothing when there is nothing to stop.
curl -s -m 10 -X POST localhost:8100/mission/stop > /dev/null 2>&1 || true
sleep 1

echo "dron $N mision: $MISION_CLASE"
MISION=$(printf '{"protocol":"%s","initial_position":[0,0,30],"origin_gps_coordinates":[-22.978029946,-43.23214256266666,0],"x_axis_degrees":0,"node_ip_dict":%s,"communication_protocol":"http","label":"banco_%s_nodos"}' "$MISION_CLASE" "$DICT" "${#DIRS[@]}")
echo "dron $N load:  $(curl -s -X POST localhost:8100/mission/load -H 'Content-Type: application/json' -d "$MISION" | cut -c1-70)"
echo "dron $N setup: $(curl -s -m 60 -X POST localhost:8100/mission/setup | cut -c1-70)"
echo "dron $N ve la estacion: $(curl -s -m 4 -o /dev/null -w %{http_code} "http://$EST/buscar")"
