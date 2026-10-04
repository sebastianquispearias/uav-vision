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

# En DOS pasos, y no por prolijidad. El apagado corta la conexion, asi que ssh devuelve error
# aunque haya funcionado: su estado no sirve para saber si la orden llego. El que si sirve es el
# del primer paso, que para la mision y vuelve normalmente.
# Y se recuerda a cuales llego, porque sin eso una placa cuyo ssh fallo y que ademas no contesta
# al ping salia informada como "APAGADA, ya se puede desenchufar" y no lo estaba: podia estar
# colgada, con el sistema vivo y escribiendo. Desenchufar una Raspberry asi es la forma mas facil
# de romper una SD sana, y el 3oct esa era justo la tarjeta cuya salud acabábamos de probar.
ORDENADAS=""
for host in "$@"; do
    printf '%s: ' "$host"
    if salida="$($SSH -n "$host" 'curl -s -m 15 -X POST localhost:8100/mission/stop | cut -c1-48; sleep 2; sudo sync' 2>&1)"; then
        echo "$salida" | head -1
        ORDENADAS="$ORDENADAS $host"
        # Y ahora si el apagado, cuyo estado se ignora a proposito.
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
        ping -n 1 -w 1500 "$ip" > /dev/null 2>&1 || ping -c 1 -W 2 "$ip" > /dev/null 2>&1 \
            && quedan=$((quedan + 1))
    done
    [ "$quedan" = 0 ] && break
done
for host in "$@"; do
    ip="${host#*@}"
    printf '  %s: ' "$ip"
    # El TEXTO del ping y no su codigo de salida: en Windows ping devuelve EXITO cuando la
    # respuesta es "Destination host unreachable", que la manda la propia laptop y no el destino.
    if ping -n 1 -w 1500 "$ip" 2>&1 | grep -qiE "bytes=|TTL="; then
        echo "TODAVIA responde, esperar unos segundos mas antes de desenchufar"
    elif echo "$ORDENADAS" | grep -q -- "$host"; then
        echo "APAGADA limpiamente, ya se puede desenchufar"
    else
        echo "sin red Y SIN ORDEN DE APAGADO: puede estar viva y colgada."
        echo "        Mira su LED verde: si parpadea, sigue escribiendo en la SD. No la saques."
    fi
done
