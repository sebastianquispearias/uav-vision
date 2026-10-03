#!/usr/bin/env bash
# Stops the missions and powers the boards down, in that order.
#
#   bash scripts/banco_embedded/apagar.sh pi@192.168.1.125 pi@192.168.1.126
#
# The order matters: a board cut mid-report leaves a half-written card on the station, and the
# sync before the halt is what keeps the flight's logs on the card instead of in a buffer.
#
# AND IT HALTS, it does not reboot. After this the boards cannot be reached over the network at
# all: to use them again they have to be UNPLUGGED AND PLUGGED BACK IN by hand. Said here because
# on 2026-10-03 this command was run by mistake in the middle of a session, numbered right under
# the one that starts things, and the way back was a walk to the bench.
set -u
SSH="${SSH:-ssh}"
[ "$#" -ge 1 ] || { echo "uso: bash $0 <usuario@placa> [usuario@placa ...]"; exit 2; }

for host in "$@"; do
    printf '%s: ' "$host"
    $SSH -n "$host" 'curl -s -m 15 -X POST localhost:8100/mission/stop | cut -c1-48; echo; \
        sleep 2; sudo sync; sudo shutdown -h now' 2>&1 | head -1
done

echo "-- esperando a que dejen de responder"
for i in $(seq 1 12); do
    sleep 5
    quedan=0
    for host in "$@"; do
        ip="${host#*@}"
        ping -n 1 -w 1500 "$ip" > /dev/null 2>&1 || ping -c 1 -W 2 "$ip" > /dev/null 2>&1 \
            && quedan=$((quedan + 1))
    done
    [ "$quedan" = 0 ] && break
done
for host in "$@"; do
    ip="${host#*@}"
    printf '  %s: ' "$ip"
    if ping -n 1 -w 1500 "$ip" > /dev/null 2>&1 || ping -c 1 -W 2 "$ip" > /dev/null 2>&1; then
        echo "TODAVIA responde, esperar unos segundos mas antes de desenchufar"
    else
        echo "APAGADA, ya se puede desenchufar"
    fi
done
