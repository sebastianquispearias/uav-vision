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
# From PowerShell, which is where this is used, on ONE single line: PowerShell does not read a
# backslash as a line continuation, and the error it gives does not mention the backslash.
#   bash scripts/banco_embedded/levantar_banco.sh 192.168.1.121:8300 pi@192.168.1.125 pi@192.168.1.126
#
# e.g.  bash scripts/banco_embedded/levantar_banco.sh 10.0.0.9:8300 \
#           pi@10.0.0.11 pi@10.0.0.12 pi@10.0.0.13
#
# Each board gets a replay offset, in seconds of the recording, from BANCO_DESDE_S below. The
# recording holds two takeoffs about 700 s apart, so 0 and 700 put two boards on a takeoff at the
# same instant; a third board repeating 700 gives two aircraft reporting the SAME target at the
# same moment, which is what the fusion demo needs. Override with:
#   BANCO_DESDE_S="0 700 700" bash scripts/banco_embedded/levantar_banco.sh ...
#
# THE KNOBS AND THE FOUR STEPS, AND WHY EACH IS AS IT IS
#
# Named so the gate can hand it a recording stub instead of the real client: a test that had to win
# a PATH race against the system ssh would be testing the PATH, and on Windows it would lose it.
# Left unquoted where it is used so it can be more than one word, which is how the gate passes
# "bash <stub>"; a path with spaces in it would therefore not work, and nothing here has one.
#
# Which mission every board loads. The default replays the recording: no camera needed and the same
# scene every run. For a demo on real cameras:
#   BANCO_MISION=mision_vision:ProtocoloVisionLAC bash ...levantar_banco.sh ...
# With that the detection is real and the position is not, because a board on a desk has no
# autopilot: the metres come from the fake one. Say so rather than let anyone assume otherwise.
#
# Appearance and rate, ONE ENTRY PER BOARD, in the same order as the addresses. The two boards of
# this bench are not the same machine, and the appearance model costs several times more on one
# than on the other; NOTES.md has the measurement.
# BANCO_REID: "si" leaves the mission's own default, "no" turns it off on that board.
# BANCO_FPS:  that board's DECLARED rate, which has to resemble the one it really achieves. The
#             identity layer scales ALL its maturity thresholds by the declared rate, so a board
#             slower than it claims silently stretches what "eight looks of evidence" means.
#   BANCO_REID="si no" BANCO_FPS="4 2" bash ...levantar_banco.sh ...
#
# BANCO_MODELO: which detector each board loads. "coco" (the default), "visdrone", or a whole
#             path. THE SCENE DECIDES: the aerial model sees NOTHING indoors, which NOTES.md
#             measures.
#   BANCO_MODELO="coco visdrone" bash ...levantar_banco.sh ...
# CAREFUL: whatever goes into EXTRA is split on spaces when it reaches the board, so a path WITH
#      spaces breaks the start-up and the error does not name the space. The aliases have none.
#
# The addresses the boards exchange reports on, plus the station last: this is node_ip_dict, and
# the node ids are the positions in it.
# Nothing starts until every argument looks like a board. A line pasted twice into a terminal
# arrives as six boards, three of which are a filename, the station's own address and a hostname
# with 'bash' stuck to the end -- and the old version gave each of them a node id, tried to set
# its clock, and started the two real boards TWICE under different ids. In the field that is a
# fleet where two aircraft answer to one name and the dictionary is nonsense, and the only sign is
# a few lines of 'Name or service not known' scrolling past.
#
# BANCO_SIN_ESTACION=1 skips step 0, and it is a seam for the gate, like SSH is. On a laptop there
# is usually a station already answering, so the curl below finds it and launches nothing; on a
# clean machine the launcher starts a REAL one and leaves it running, which is the difference
# between a gate that passes on the desk and one that hangs in CI. What the gate is about is the
# ORCHESTRATION -- which board gets which dictionary, which node id, which replay offset, the
# clock before any of it, and every board set up before any is started -- and none of that needs a
# station to exist.
#
# 0. The station, unless one is already answering. Here and not in a second command because the
#    arguments are easy to get wrong in a way that is hard to see: without --nodos the station
#    cannot send anything and answers 'ningun dron conectado', and two stations can hold the same
#    port without complaining, so the one that answers may not be the one just started. Set
#    ESTACION_FLAGS to change what it is started with.
#
# 1. The clock. A Raspberry comes up with the wrong time after a cold boot, and then the three
#    boards' logs cannot be crossed with the laptop's or with each other's.
#
# 2. Set up every board. One ssh per board, and -n because an ssh that launches background
#    processes does not return the prompt even when they are setsid-ed.
#
# 3. Only now, start them together. This is the whole reason the per-board script stops short.
#    The environment assignments travel COMMA-separated and never empty: neither an empty
#    argument nor one containing spaces survives ssh, which re-parses the command on the board.
#    See lanzar_banco_nodo.sh.
set -u
SSH="${SSH:-ssh}"
DESDE_DEF="0 700 700 0 700"
EST="$1"; shift
PIS=("$@")
read -r -a DESDE <<< "${BANCO_DESDE_S:-$DESDE_DEF}"
MISION="${BANCO_MISION:-mision_banco_dos_drones:ProtocoloVisionBanco}"
read -r -a REID <<< "${BANCO_REID:-}"
read -r -a FPS <<< "${BANCO_FPS:-}"
read -r -a MODELO <<< "${BANCO_MODELO:-}"

for host in "${PIS[@]}"; do
    case "${host#*@}" in
        *[!0-9.]*|*..*|"") echo "NO ARRANCO: '$host' no parece una placa."
                           echo "Se espera  usuario@A.B.C.D  por cada placa, y nada mas."
                           echo "Si pegaste la linea dos veces, el terminal las unio: borra y repite."
                           exit 1 ;;
    esac
done
if [ "$(printf '%s
' "${PIS[@]}" | sort | uniq -d | wc -l)" -gt 0 ]; then
    echo "NO ARRANCO: hay una placa repetida en la lista."
    echo "Dos node_id para la misma placa es una flota donde dos aeronaves contestan al mismo nombre."
    exit 1
fi

DIRS=()
for host in "${PIS[@]}"; do DIRS+=("${host#*@}:8200"); done
DIRS+=("$EST")

echo "=============================================================="
echo "BANCO DE ${#PIS[@]} PLACAS   estacion $EST"
echo "  mision: $MISION"
echo "  node_ip_dict: ${DIRS[*]}"
echo "=============================================================="

PUERTO="${EST##*:}"
if [ "${BANCO_SIN_ESTACION:-0}" = 1 ]; then
    echo "-- estacion SALTADA por BANCO_SIN_ESTACION=1"
elif curl -s -m 3 -o /dev/null "http://127.0.0.1:$PUERTO/estado"; then
    echo "-- ya hay una estacion viva en $PUERTO, se usa esa"
else
    echo "-- levantando la estacion en $PUERTO"
    NODOS=""
    for i in "${!PIS[@]}"; do
        [ -n "$NODOS" ] && NODOS="$NODOS,"
        NODOS="$NODOS$((i + 1))=${PIS[$i]#*@}:8200"
    done
    ( cd "$(dirname "$0")/../.." && nohup python scripts/banco_embedded/gs_mapa.py         --puerto "$PUERTO" --origen="${ORIGEN_GPS:--22.978029946,-43.23214256266666}"         --nodos "$NODOS" ${ESTACION_FLAGS:---segunda-opinion --banco}         > /tmp/estacion.log 2>&1 & )
    for i in $(seq 1 40); do
        curl -s -m 2 -o /dev/null "http://127.0.0.1:$PUERTO/estado" && break
        sleep 1
    done
    echo "   estacion responde tras ${i}s   (--nodos $NODOS)"
fi

AHORA="$(date -u '+%Y-%m-%d %H:%M:%S')"
for host in "${PIS[@]}"; do
    echo "-- reloj de $host"
    $SSH -n "$host" "sudo date -u -s '$AHORA' >/dev/null && date" || echo "   FALLO el reloj de $host"
done

for i in "${!PIS[@]}"; do
    echo "-- preparando ${PIS[$i]} como nodo $((i + 1)), desde ${DESDE[$i]:-0} s"
    EXTRA=""
    case "${REID[$i]:-}" in no|NO|off) EXTRA="BANCO_REID=" ;; esac
    [ -n "${FPS[$i]:-}" ] && EXTRA="${EXTRA:+$EXTRA,}BANCO_FPS=${FPS[$i]}"
    [ -n "${MODELO[$i]:-}" ] && EXTRA="${EXTRA:+$EXTRA,}BANCO_MODELO=${MODELO[$i]}"
    [ -z "$EXTRA" ] && EXTRA="-"
    $SSH "${PIS[$i]}" 'bash -s' -- "$((i + 1))" "${DESDE[$i]:-0}" "$MISION" "$EXTRA" "${DIRS[@]}" \
        < "$(dirname "$0")/lanzar_banco_nodo.sh" || echo "   FALLO ${PIS[$i]}"
done

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
