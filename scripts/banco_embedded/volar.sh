#!/usr/bin/env bash
# One command for a flight day: provision the boards, bring up the station, start the mission on
# every aircraft, and REFUSE TO SAY IT IS READY until it has checked that each drone is reporting.
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
set -u
SSH="${SSH:-ssh}"
MISION="${VOLAR_MISION:-mision_barrido:ProtocoloBarridoLAC}"
# Seconds a drone may go without speaking before this refuses to call it ready. Its report period
# is 2 s, so ten is four misses: late enough not to trip on one lost packet, early enough that
# nobody walks to the field with a dead board.
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

# Nothing starts until every argument looks like a board. A line pasted twice arrives as six
# boards, and the old launcher gave each of them a node id and started the real ones twice.
for host in "${PIS[@]}"; do
    case "${host#*@}" in
        *[!0-9.]*|*..*|"") echo "NO ARRANCO: '$host' no parece una placa."
                           echo "Se espera  usuario@A.B.C.D  por cada placa, y nada mas."
                           exit 1 ;;
    esac
done
if [ "$(printf '%s\n' "${PIS[@]}" | sort | uniq -d | wc -l)" -gt 0 ]; then
    echo "NO ARRANCO: hay una placa repetida."
    exit 1
fi

echo "=============================================================="
echo "VOLAR   ${#PIS[@]} aeronaves   estacion $EST"
echo "  mision: $MISION"
echo "=============================================================="

# 1. Que esten vivas, antes de gastar cuatro minutos en descubrir que una no lo esta.
# Costura para la prueba, igual que SSH: el gate corre este guion entero contra direcciones que
# no existen, y un ping de verdad tardaria y fallaria por motivos que no se estan probando.
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

# 2. El codigo. Cada vez, porque no viaja solo y porque una placa con el archivo nuevo y el
#    modulo viejo en memoria es el fallo mas caro de diagnosticar que dio este banco.
if [ "$SALTAR_PREPARAR" = 1 ]; then
    echo "-- 2/5 provision SALTADA por VOLAR_SIN_PREPARAR=1"
else
    echo "-- 2/5 mandando el codigo y comprobando los modelos"
    for host in "${PIS[@]}"; do
        bash "$AQUI/preparar_placa.sh" "$host" 2>&1 | sed -n 's/^/   /p' | grep -aE "hay |FALTA|listo|NO IMPORTA" \
            || { echo "   FALLO preparando $host"; exit 1; }
    done
fi

# 3. La estacion. Se reusa la que haya viva: dos pueden tomar el mismo puerto sin quejarse, y la
#    que contesta no tiene por que ser la recien abierta.
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

# 4. Las aeronaves. Preparadas todas primero y arrancadas juntas, que es lo unico que hace que
#    sus relojes y sus reproducciones hablen del mismo instante.
DIRS=()
for host in "${PIS[@]}"; do DIRS+=("${host#*@}:8200"); done
DIRS+=("$EST")

echo "-- 4/5 cargando la mision en cada aeronave"
AHORA="$(date -u '+%Y-%m-%d %H:%M:%S')"
for i in "${!PIS[@]}"; do
    $SSH -n "${PIS[$i]}" "sudo date -u -s '$AHORA' >/dev/null" 2>/dev/null || true
    # El centinela "-" y no una cadena vacia: un argumento vacio NO sobrevive a ssh, que une la
    # orden y deja que la placa la vuelva a parsear. Lo mismo con los espacios.
    $SSH "${PIS[$i]}" 'bash -s' -- "$((i + 1))" 0 "$MISION" "-" "${DIRS[@]}" \
        < "$AQUI/lanzar_banco_nodo.sh" 2>&1 | sed -n 's/^/   /p' | grep -aE "setup:|FALLO|entorno" \
        || { echo "   FALLO cargando en ${PIS[$i]}"; exit 1; }
done

echo "   arrancando las ${#PIS[@]} juntas"
for host in "${PIS[@]}"; do
    $SSH -n "$host" "curl -s -m 10 -X POST localhost:8100/mission/start > /dev/null" &
done
wait

# 5. Y la unica pregunta que importa: reportan? El log de la estacion NO sirve para esto: solo
#    imprime los reportes que traen un hallazgo, asi que un dron sano que todavia no vio a nadie
#    se lee como un dron muerto. /estado dice hace cuanto hablo cada uno, que es la salud.
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
