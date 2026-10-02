#!/usr/bin/env bash
# Brings the whole desk bench up from the laptop, in one command, for however many boards there are.
#
# This is the script that was missing. Everything it does was already possible by hand, and doing
# it by hand is where the bench went wrong before: the boards have to know each other's addresses
# because autodiscovery does not work on this LAN (configuration.py:117,164), their clocks have to
# agree or the three logs cannot be read against each other, and they have to be STARTED TOGETHER
# or each drone is replaying a different moment of the flight and there is nothing to fuse.
#
#   bash scripts/banco_embedded/levantar_banco.sh <estacion ip:8300> <pi1> <pi2> [pi3 ...]
#
# e.g.  bash scripts/banco_embedded/levantar_banco.sh 10.0.0.9:8300 \
#           pi@10.0.0.11 pi@10.0.0.12 pi@10.0.0.13
#
# Each board gets a replay offset, in seconds of the recording, from BANCO_DESDE_S below. The
# recording holds two takeoffs about 700 s apart, so 0 and 700 put two boards on a takeoff at the
# same instant; a third board repeating 700 gives two aircraft reporting the SAME target at the
# same moment, which is what the fusion demo needs. Override with:
#   BANCO_DESDE_S="0 700 700" bash scripts/banco_embedded/levantar_banco.sh ...
set -u
# Named so the gate can hand it a recording stub instead of the real client: a test that had to win
# a PATH race against the system ssh would be testing the PATH, and on Windows it would lose it.
# Left unquoted where it is used so it can be more than one word, which is how the gate passes
# "bash <stub>"; a path with spaces in it would therefore not work, and nothing here has one.
SSH="${SSH:-ssh}"
DESDE_DEF="0 700 700 0 700"
EST="$1"; shift
PIS=("$@")
read -r -a DESDE <<< "${BANCO_DESDE_S:-$DESDE_DEF}"
# Which mission every board loads. The default replays the recording: no camera needed and the same
# scene every run. For a demo on real cameras:
#   BANCO_MISION=mision_vision:ProtocoloVisionLAC bash ...levantar_banco.sh ...
# With that the detection is real and the position is not, because a board on a desk has no
# autopilot: the metres come from the fake one. Say so rather than let anyone assume otherwise.
MISION="${BANCO_MISION:-mision_banco_dos_drones:ProtocoloVisionBanco}"

# The addresses the boards exchange reports on, plus the station last: this is node_ip_dict, and
# the node ids are the positions in it.
DIRS=()
for host in "${PIS[@]}"; do DIRS+=("${host#*@}:8200"); done
DIRS+=("$EST")

echo "=============================================================="
echo "BANCO DE ${#PIS[@]} PLACAS   estacion $EST"
echo "  mision: $MISION"
echo "  node_ip_dict: ${DIRS[*]}"
echo "=============================================================="

# 1. The clock. A Raspberry comes up with the wrong time after a cold boot, and then the three
#    boards' logs cannot be crossed with the laptop's or with each other's.
AHORA="$(date -u '+%Y-%m-%d %H:%M:%S')"
for host in "${PIS[@]}"; do
    echo "-- reloj de $host"
    $SSH -n "$host" "sudo date -u -s '$AHORA' >/dev/null && date" || echo "   FALLO el reloj de $host"
done

# 2. Set up every board. One ssh per board, and -n because an ssh that launches background
#    processes does not return the prompt even when they are setsid-ed.
for i in "${!PIS[@]}"; do
    echo "-- preparando ${PIS[$i]} como nodo $((i + 1)), desde ${DESDE[$i]:-0} s"
    $SSH "${PIS[$i]}" 'bash -s' -- "$((i + 1))" "${DESDE[$i]:-0}" "$MISION" "${DIRS[@]}" \
        < "$(dirname "$0")/lanzar_banco_nodo.sh" || echo "   FALLO ${PIS[$i]}"
done

# 3. Only now, start them together. This is the whole reason the per-board script stops short.
echo "-- arrancando las ${#PIS[@]} juntas"
for host in "${PIS[@]}"; do
    $SSH -n "$host" "curl -s -m 10 -X POST localhost:8100/mission/start | cut -c1-60" &
done
wait

echo "=============================================================="
echo "LISTO. Abrir la estacion en 127.0.0.1:${EST##*:}  (no localhost: cuesta ~2 s por peticion)"
echo "Comprobar antes que no haya otra estacion viva:  netstat -ano | findstr :${EST##*:}"
echo "Logs de cada placa:  ssh <pi> 'tail -f ~/banco/runner.log'"
echo "=============================================================="
