#!/usr/bin/env bash
# Puts everything one board of the bench needs on it, from the laptop, in ONE ssh connection.
#
# Written because provisioning by hand is how a board ends up running code from three weeks ago
# while the other runs today's, and the only symptom is that the two disagree.
#
#   bash scripts/banco_embedded/preparar_placa.sh pi@192.168.1.126
#
# One connection for the transfer and one for the check, and that is not a style choice: the Pi 5's
# sshd gives up after three or four connections with 'Connection timed out during banner exchange'
# and then needs two minutes to let go of the half-open ones. A script that opened eight could not
# be run on it at all.
#
# What it sends:
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
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

echo "== preparando $PI desde $RAIZ"
mkdir -p "$STAGE/banco/datos" "$STAGE/gradys-embedded" "$STAGE/gradys_protocols"
cp -r "$RAIZ/uav_vision" "$STAGE/banco/"
cp -r "$LAC/gradys-embedded/gradys_embedded" "$STAGE/gradys-embedded/"
# Both missions: the one that replays the recording and the one that uses the real camera. Which
# of the two runs is chosen when the mission is loaded, not when the board is provisioned, and a
# board that only has one of them cannot be switched without another transfer.
for m in mision_banco_dos_drones.py mision_vision.py mision_vision_2fps.py mision_banco_lab.py; do
    [ -f "$AQUI/$m" ] && cp "$AQUI/$m" "$STAGE/gradys_protocols/"
done
cp "$AQUI/uav_api_stub.py" "$STAGE/uav_api_stub.py"
for f in "$RAIZ/demo/data/examen_v3_datos.npz" "$RAIZ/demo/data/embs_osnet.npy" \
         "$RAIZ/demo/data/frames.csv" "$ENTREN/pistas_sustituto_02ago.npz"; do
    [ -f "$f" ] || { echo "   FALTA $f"; exit 1; }
    cp "$f" "$STAGE/banco/datos/"
done
find "$STAGE" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
echo "-- $(du -sh "$STAGE" | cut -f1) a enviar en una sola conexion"

tar -czf - -C "$STAGE" . | ssh "$PI" 'tar -xzf - -C ~ && echo "   recibido"'

# Los modelos NO viajan aqui: pesan y viven en la placa. Pero su ausencia no se nota al
# provisionar, se nota cuando la mision no arranca, y el error de entonces no nombra el archivo.
# El 3oct la Pi 4 no tenia el de apariencia y hubo que copiarlo a mano desde la Pi 5.
echo "-- comprobando los modelos que la placa ya tiene que tener"
ssh -n "$PI" 'falta=0
for m in /home/pi/yolov8n_ncnn_model /home/pi/modelos_visdrone/osnet_x0_25_msmt17.pt; do
    if [ -e "$m" ]; then echo "   hay  $m"; else echo "   FALTA $m"; falta=1; fi
done
[ "$falta" = 0 ] || echo "   una mision con recortes pide el de apariencia y NO ARRANCA sin el.
   Se copia desde la otra placa:  scp pi@<otra>:/home/pi/modelos_visdrone/osnet_x0_25_msmt17.pt ."
exit 0'

echo "-- comprobando que la placa importa lo que se le mando"
ssh -n "$PI" 'cd ~/banco && PYTHONPATH="$HOME/banco:$HOME/gradys-embedded:$HOME/gradys_protocols" \
    python3 -c "
import numpy, uav_vision.vision_protocol as v, gradys_embedded.runner.cli
import mision_banco_dos_drones
print(\"   numpy\", numpy.__version__)
print(\"   rodeo: tolerancia\", v.RODEO_TOLERANCIA_M, \"m, plazo\", v.RODEO_PLAZO_S, \"s\")
print(\"   maniobra\", hasattr(v.VisionProtocol, \"rodear\"), \"descarte\", hasattr(v.VisionProtocol, \"descartar\"))
"' || { echo "   LA PLACA NO IMPORTA: ver el error de arriba"; exit 1; }
echo "== $PI listo"
