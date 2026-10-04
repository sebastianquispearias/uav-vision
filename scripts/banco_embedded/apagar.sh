#!/usr/bin/env bash
# Stops the missions and powers the boards down, in that order.
#
#   bash scripts/banco_embedded/apagar.sh pi@192.168.1.125 pi@192.168.1.126
#
# The order matters: a board cut mid-report leaves a half-written card on the station, and the
# sync before the halt keeps the flight's logs on the card instead of in a buffer.
#
# It halts, it does not reboot. Afterwards the boards cannot be reached over the network at all:
# using them again means unplugging and plugging them back in by hand.
#
# Each board is shut down in two steps, and the split is what makes the report honest. The halt
# cuts the connection, so ssh returns an error even when it worked and its status cannot say
# whether the order arrived; the first step, which stops the mission and syncs, returns normally
# and its status can. Boards that got that far are remembered, because a board whose ssh failed
# and which also does not answer a ping used to be reported as "APAGADA, ya se puede desenchufar"
# while it was still running and writing, and pulling the power from a Raspberry in that state is
# the easiest way to destroy a healthy SD card.
#
# The ping is read by its TEXT and not its exit status: on Windows ping succeeds when the reply is
# "Destination host unreachable", which the laptop itself sends.
set -u
SSH="${SSH:-ssh}"
[ "$#" -ge 1 ] || { echo "uso: bash $0 <usuario@placa> [usuario@placa ...]"; exit 2; }

ORDENADAS=""
for host in "$@"; do
    printf '%s: ' "$host"
    if salida="$($SSH -n "$host" 'curl -s -m 15 -X POST localhost:8100/mission/stop | cut -c1-48; sleep 2; sudo sync' 2>&1)"; then
        echo "$salida" | head -1
        ORDENADAS="$ORDENADAS $host"
        $SSH -n "$host" 'sudo shutdown -h now' > /dev/null 2>&1 || true
    else
        echo "$(echo "$salida" | head -1)"
        echo "   NO SE LE PUDO DAR LA ORDEN. No la desenchufes sin mirar su LED verde."
    fi
done

echo "-- esperando a que dejen de responder"
for i in $(seq 1 12); do
    sleep 5
    quedan=0
    for host in "$@"; do
        ip="${host#*@}"
        ping -n 1 -w 1500 "$ip" 2>&1 | grep -qiE "bytes=|TTL=" && quedan=$((quedan + 1))
    done
    [ "$quedan" = 0 ] && break
done
for host in "$@"; do
    ip="${host#*@}"
    printf '  %s: ' "$ip"
    if ping -n 1 -w 1500 "$ip" 2>&1 | grep -qiE "bytes=|TTL="; then
        echo "TODAVIA responde, esperar unos segundos mas antes de desenchufar"
    elif echo "$ORDENADAS" | grep -q -- "$host"; then
        echo "APAGADA limpiamente, ya se puede desenchufar"
    else
        echo "sin red Y SIN ORDEN DE APAGADO: puede estar viva y colgada."
        echo "        Mira su LED verde: si parpadea, sigue escribiendo en la SD. No la saques."
    fi
done
