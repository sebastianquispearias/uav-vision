#!/usr/bin/env bash
# Puts everything one board of the bench needs on it, from the laptop, over the cable.
#
# Written because provisioning by hand is how a board ends up running code from three weeks ago
# while the other runs today's, and the only symptom is that the two disagree. Run it on every
# board before a bench session and they are identical by construction.
#
#   bash scripts/banco_embedded/preparar_placa.sh pi@192.168.1.126
#
# What it sends, and where each piece comes from:
#   ~/banco/uav_vision      the package, from this working tree, which is the newest there is
#   ~/banco/datos           the four files the recorded mission reads
#   ~/gradys-embedded       the GrADyS runtime, from the repo next door
#   ~/gradys_protocols      the mission module
#   ~/uav_api_stub.py       the fake autopilot THAT SIMULATES MOVEMENT. uav_api_falso.py may not,
#                           and then a drone sent to look from another side never arrives.
set -eu
PI="$1"
AQUI="$(cd "$(dirname "$0")" && pwd)"
RAIZ="$(cd "$AQUI/../.." && pwd)"
LAC="$(cd "$RAIZ/.." && pwd)"
ENTREN="$LAC/drone-geolocation/entrenamiento"

echo "== preparando $PI desde $RAIZ"
ssh -n "$PI" 'mkdir -p ~/banco/datos ~/gradys-embedded ~/gradys_protocols'

echo "-- el paquete uav_vision"
tar -czf - -C "$RAIZ" --exclude='__pycache__' uav_vision \
    | ssh "$PI" 'tar -xzf - -C ~/banco'

echo "-- los datos de la mision grabada"
for f in "$RAIZ/demo/data/examen_v3_datos.npz" "$RAIZ/demo/data/embs_osnet.npy" \
         "$RAIZ/demo/data/frames.csv" "$ENTREN/pistas_sustituto_02ago.npz"; do
    [ -f "$f" ] || { echo "   FALTA $f"; exit 1; }
    scp -q "$f" "$PI:~/banco/datos/"
    echo "   $(basename "$f")"
done

echo "-- el runtime de GrADyS"
tar -czf - -C "$LAC/gradys-embedded" --exclude='__pycache__' gradys_embedded \
    | ssh "$PI" 'tar -xzf - -C ~/gradys-embedded'

echo "-- la mision y el piloto automatico falso"
scp -q "$AQUI/mision_banco_dos_drones.py" "$PI:~/gradys_protocols/"
scp -q "$AQUI/uav_api_stub.py" "$PI:~/uav_api_stub.py"
scp -q "$AQUI/uav_api_proxy.py" "$PI:~/" 2>/dev/null || true

echo "-- comprobando que la placa puede importar lo que le mandamos"
ssh -n "$PI" 'cd ~/banco && PYTHONPATH="$HOME/banco:$HOME/gradys-embedded:$HOME/gradys_protocols" \
    python3 -c "
import uav_vision.vision_protocol as v
import gradys_embedded.runner.cli
print(\"  uav_vision y gradys_embedded importan\")
print(\"  rodeo: tolerancia\", v.RODEO_TOLERANCIA_M, \"m, plazo\", v.RODEO_PLAZO_S, \"s\")
"' || { echo "   LA PLACA NO PUEDE IMPORTAR: ver el error de arriba"; exit 1; }
echo "== $PI listo"
