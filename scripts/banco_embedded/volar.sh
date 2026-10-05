#!/usr/bin/env bash
# One command for a flight day: provision the boards, bring up the station, start the mission on
# every aircraft, and REFUSE TO SAY IT IS READY until it has checked that each drone is reporting.
#
#   bash scripts/banco_embedded/volar.sh auto pi@rpanion.local pi@drone4.local
#
# "auto" works out the station's address by asking the first board, and the .local names work on
# any network. Those two together are what make the flight-day command IDENTICAL with a cable
# and with the hotspot, which is the only thing that makes it memorable. Everything can also be
# given by hand:
#
#   bash scripts/banco_embedded/volar.sh 192.168.1.121:8300 pi@192.168.1.125 pi@192.168.1.126
#
# From PowerShell, which is where this is actually typed, the wrapper beside it does the same:
#   .\scripts\banco_embedded\volar.ps1 192.168.1.121:8300 pi@192.168.1.125 pi@192.168.1.126
#
# Why this exists and levantar_banco.sh is not enough: the launcher starts things. On a flight day
# the question is not "did it start" but "can I take off", and those differ in three places that
# all bit on 2026-10-03.
#
#   THE CODE DOES NOT TRAVEL WHEN A MISSION STARTS. Only preparar_placa.sh copies it. A board can
#   hold the new file on disk and keep running the old module in memory, and the only symptom is
#   a new field arriving as null. So this provisions first, every time, and the cost is 16 MB.
#
#   THE FLIGHT MISSION IS NOT THE ONE WHOSE NAME SOUNDS LIKE IT. mision_vision.py is the desk
#   bench: no crops, no appearance, no preliminaries, and the indoor detector. Flying that means
#   an operator with nothing to judge and no way to say "not it". The default here is
#   mision_barrido, which is the one that declares itself the configuration to fly.
#
#   "STARTED" IS NOT "REPORTING". The station only prints a line per report that CARRIES a find;
#   an empty beat updates the drone's card and prints nothing, on purpose. Reading the log for
#   health therefore shows silence from a perfectly healthy aircraft that has not seen anybody
#   yet, which is exactly the wrong conclusion to reach while deciding whether to take off. So
#   the check at the end asks /estado how long ago each drone last spoke, which is the number
#   that means health.
#
# FRESCO_S is how long a drone may go without speaking before this refuses to call it ready. The
# report period is 2 s, so ten seconds is four misses: late enough not to trip on one lost
# packet, early enough that nobody walks to the field with a dead board.
#
# NOTHING STARTS until every argument looks like a board. What is demanded is the user@ and not a
# numeric address: without a cable the hotspot hands out the IPs and they change every time,
# while the names do not. Both boards run avahi, so rpanion.local and drone4.local work on any
# network and are the only way to have a command that is typed the same way every day. The user@
# is also the filter that matters: a line pasted twice brings filenames and bare addresses along,
# and none of those carries an at sign.
#
# THE STATION'S ADDRESS can be given, or asked for with "auto". Without a cable the hotspot hands
# it out and it changes every time, so typing it by hand is the part of the command that gets
# typed wrong on exactly the day it matters. It is worked out by asking the BOARD, which is who
# has the correct answer by definition: SSH_CLIENT carries the address this connection arrived
# from, which is the laptop as seen from the network the two share. A laptop with Wi-Fi and
# Ethernet at once has several addresses and only one of them is any use; this is always that
# one, with no guessing about interfaces. The -4 is deliberate: without it ssh may negotiate
# IPv6 and SSH_CLIENT returns a link-local address, which is correct and no use at all for
# building a URL or for letting the other board reach the station. Measured over the cable on
# 2026-10-03.
#
# THE FIVE STEPS, and why each is where it is:
#
#   1. That they answer at all, before spending four minutes finding out one of them does not.
#      VOLAR_SIN_PING is a seam for the test, like SSH: the gate runs this whole script against
#      addresses that do not exist, and a real ping would be slow and would fail for reasons
#      nobody is testing.
#   2. The code, every time, because it does not travel on its own and because a board with the
#      new file and the old module in memory is the most expensive failure this bench has
#      produced.
#   3. The station, reusing whichever one is alive: two can hold the same port without
#      complaining, and the one that answers need not be the one just opened.
#   4. The aircraft, all PREPARED first and STARTED together, which is the only thing that makes
#      their clocks and their replays talk about the same instant. The sentinel "-" is passed
#      rather than an empty string, because an empty argument does NOT survive ssh: ssh joins the
#      command and lets the board parse it again. The same goes for spaces.
#   5. And the only question that matters: are they reporting? The station's log is no use for
#      this, since it only prints the reports that carry a find, so a healthy drone that has not
#      seen anybody yet reads as a dead one. /estado says how long ago each one spoke, which is
#      what health means.
set -u
SSH="${SSH:-ssh}"
MISION="${VOLAR_MISION:-mision_barrido:ProtocoloBarridoLAC}"
FRESCO_S="${VOLAR_FRESCO_S:-10}"
SALTAR_PREPARAR="${VOLAR_SIN_PREPARAR:-0}"

if [ "$#" -lt 2 ]; then
    echo "uso: bash $0 <estacion ip:puerto> <usuario@placa> [usuario@placa ...]"
    exit 2
fi
EST="$1"; shift
PIS=("$@")
AQUI="$(cd "$(dirname "$0")" && pwd)"
RAIZ="$(cd "$AQUI/../.." && pwd)"

for host in "${PIS[@]}"; do
    case "$host" in
        *@*) ;;
        *) echo "NO ARRANCO: '$host' no parece una placa."
           echo "Se espera  usuario@nombre  o  usuario@A.B.C.D  por cada placa."
           echo "Con el hotspot conviene el nombre: pi@rpanion.local, pi@drone4.local."
           exit 1 ;;
    esac
    case "${host#*@}" in
        ""|*' '*) echo "NO ARRANCO: '$host' no tiene host despues del arroba."; exit 1 ;;
    esac
done

if [ "$(printf '%s\n' "${PIS[@]}" | sort | uniq -d | wc -l)" -gt 0 ]; then
    echo "NO ARRANCO: hay una placa repetida."
    exit 1
fi

if [ "${EST%%:*}" = "auto" ] || [ "$EST" = "auto" ]; then
    PUERTO_PEDIDO="8300"
    case "$EST" in auto:*) PUERTO_PEDIDO="${EST#auto:}" ;; esac
    echo "-- 0/5 averiguando con que direccion me ve la primera placa"
    MIA="$($SSH -4 -n -o ConnectTimeout=15 "${PIS[0]}" 'echo $SSH_CLIENT' 2>/dev/null | awk '{print $1}')"
    case "$MIA" in
        *[!0-9.]*|"") echo "NO ARRANCO: ${PIS[0]} no supo decirme mi direccion (dijo '$MIA')."
                      echo "Pasar la direccion a mano:  <ip de la laptop>:$PUERTO_PEDIDO"
                      exit 1 ;;
    esac
    EST="$MIA:$PUERTO_PEDIDO"
    echo "   la placa me ve como $MIA, asi que la estacion va en $EST"
fi

echo "=============================================================="
echo "VOLAR   ${#PIS[@]} aeronaves   estacion $EST"
echo "  mision: $MISION"
echo "=============================================================="

echo "-- 1/5 las placas contestan?"
falta=0
if [ "${VOLAR_SIN_PING:-0}" = 1 ]; then
    echo "   comprobacion saltada por VOLAR_SIN_PING=1"
fi
for host in "${PIS[@]}"; do
    [ "${VOLAR_SIN_PING:-0}" = 1 ] && break
    ip="${host#*@}"
    if ping -n 2 -w 2000 "$ip" > /dev/null 2>&1 || ping -c 2 -W 2 "$ip" > /dev/null 2>&1; then
        echo "   $ip OK"
    else
        echo "   $ip NO RESPONDE"
        falta=1
    fi
done
if [ "$falta" = 1 ]; then
    echo "NO ARRANCO: alguna placa no contesta. Enciende el hotspot y revisa el cable."
    exit 1
fi

if [ "$SALTAR_PREPARAR" = 1 ]; then
    echo "-- 2/5 provision SALTADA por VOLAR_SIN_PREPARAR=1"
else
    echo "-- 2/5 mandando el codigo y comprobando los modelos"
    for host in "${PIS[@]}"; do
        bash "$AQUI/preparar_placa.sh" "$host" 2>&1 | sed -n 's/^/   /p' | grep -aE "hay |FALTA|listo|NO IMPORTA" \
            || { echo "   FALLO preparando $host"; exit 1; }
    done
fi

PUERTO="${EST##*:}"
echo "-- 3/5 la estacion en $PUERTO"
if curl -s -m 3 -o /dev/null "http://127.0.0.1:$PUERTO/estado"; then
    echo "   ya hay una viva, se usa esa"
else
    NODOS=""
    for i in "${!PIS[@]}"; do
        [ -n "$NODOS" ] && NODOS="$NODOS,"
        NODOS="$NODOS$((i + 1))=${PIS[$i]#*@}:8200"
    done
    ( cd "$RAIZ" && nohup python scripts/banco_embedded/gs_mapa.py \
        --puerto "$PUERTO" --origen="${ORIGEN_GPS:--22.978029946,-43.23214256266666}" \
        --nodos "$NODOS" ${ESTACION_FLAGS:---segunda-opinion} > /tmp/estacion.log 2>&1 & )
    for i in $(seq 1 40); do
        curl -s -m 2 -o /dev/null "http://127.0.0.1:$PUERTO/estado" && break
        sleep 1
    done
    echo "   responde tras ${i}s   (--nodos $NODOS)"
fi

DIRS=()
for host in "${PIS[@]}"; do DIRS+=("${host#*@}:8200"); done
DIRS+=("$EST")

echo "-- 4/5 cargando la mision en cada aeronave"
AHORA="$(date -u '+%Y-%m-%d %H:%M:%S')"
for i in "${!PIS[@]}"; do
    $SSH -n "${PIS[$i]}" "sudo date -u -s '$AHORA' >/dev/null" 2>/dev/null || true
    $SSH "${PIS[$i]}" 'bash -s' -- "$((i + 1))" 0 "$MISION" "-" "${DIRS[@]}" \
        < "$AQUI/lanzar_banco_nodo.sh" 2>&1 | sed -n 's/^/   /p' | grep -aE "setup:|FALLO|entorno" \
        || { echo "   FALLO cargando en ${PIS[$i]}"; exit 1; }
done

echo "   arrancando las ${#PIS[@]} juntas"
for host in "${PIS[@]}"; do
    $SSH -n "$host" "curl -s -m 10 -X POST localhost:8100/mission/start > /dev/null" &
done
wait

echo "-- 5/5 comprobando que cada aeronave reporta"
listo=0
for intento in $(seq 1 24); do
    sleep 5
    if curl -s -m 5 "http://127.0.0.1:$PUERTO/estado" | python -c "
import json, sys
d = json.load(sys.stdin)
drones, ahora = d.get('drones') or {}, d.get('ahora') or 0
esperados, fresco = int(sys.argv[1]), float(sys.argv[2])
vivos = [k for k, v in drones.items() if ahora - (v.get('t') or 0) <= fresco]
for k in sorted(drones):
    print('   dron %s: hablo hace %.0f s' % (k, ahora - (drones[k].get('t') or 0)))
sys.exit(0 if len(vivos) >= esperados else 1)
" "${#PIS[@]}" "$FRESCO_S"; then
        listo=1
        break
    fi
done

echo "=============================================================="
if [ "$listo" = 1 ]; then
    echo "LISTO PARA VOLAR. Las ${#PIS[@]} aeronaves reportan."
    echo
    echo "  1. Abrir   http://127.0.0.1:$PUERTO     (con los numeros, no localhost)"
    echo "  2. Clic en 'person' en la fila SEARCHING, ANTES de despegar: sin esa orden las"
    echo "     tarjetas pueden mostrar un rechazo viejo de otra corrida."
    echo "  3. Volar."
    echo
    echo "  Al terminar, para las misiones y apaga las placas:"
    echo "    bash $AQUI/apagar.sh ${PIS[*]}"
else
    echo "NO ESTA LISTO: alguna aeronave no reporta en ${FRESCO_S} s. NO DESPEGAR."
    echo "  Mirar el log de la que falte:  ssh <placa> 'tail -20 ~/banco/runner.log'"
    echo "  Y recordar que el log de la estacion NO dice nada de salud: solo imprime hallazgos."
    exit 1
fi
echo "=============================================================="
