#!/usr/bin/env bash
# Installs reloj_lan.sh on a board as a systemd service, so the clock fixes itself at every
# boot without anyone typing anything.
#
#     bash scripts/banco_embedded/instalar_reloj.sh pi@192.168.1.125 http://192.168.1.121:8300/
#
# Run once per board. Afterwards the board sets its own clock from the laptop whenever the
# laptop is reachable, and keeps retrying for ten minutes after boot in case the ground station
# is started later.
#
# WHY A SERVICE AND NOT A LINE IN THE BENCH SCRIPTS. The bench scripts already set the clock,
# all three of them, so the board is right whenever it is driven from here. The gap this closes
# is the board powered up on its own: nothing on it would ever notice, and the first thing that
# does notice is a ground station showing an empty screen.
#
# WHY After=network-online.target AND a retry loop, which looks like belt and braces but is not:
# the target only promises an address, not that the laptop at the other end is up. The loop is
# what covers the ordinary case of switching the boards on before the ground station.
#
# To see what it did on the board, after a boot:
#     ssh pi@<placa> 'systemctl status reloj-lan --no-pager; journalctl -u reloj-lan -b 0'
# To take it off again:
#     ssh pi@<placa> 'sudo systemctl disable --now reloj-lan; sudo rm /etc/systemd/system/reloj-lan.service /usr/local/bin/reloj_lan.sh'

set -eu

PI="${1:-}"
shift || true
URLS="$*"
AQUI="$(cd "$(dirname "$0")" && pwd)"

if [ -z "$PI" ] || [ -z "$URLS" ]; then
    echo "uso: bash $0 <usuario@placa> <url> [url ...]" >&2
    echo "  p.ej: bash $0 pi@192.168.1.125 http://192.168.1.121:8300/" >&2
    exit 2
fi

echo "-- copiando reloj_lan.sh a $PI"
scp -o ConnectTimeout=10 "$AQUI/reloj_lan.sh" "$PI:/tmp/reloj_lan.sh"
ssh -n -o ConnectTimeout=10 "$PI" \
    'sudo install -m 755 /tmp/reloj_lan.sh /usr/local/bin/reloj_lan.sh && rm -f /tmp/reloj_lan.sh'

echo "-- escribiendo la unidad de systemd"
ssh -o ConnectTimeout=10 "$PI" "sudo tee /etc/systemd/system/reloj-lan.service >/dev/null" <<UNIDAD
[Unit]
Description=Pone el reloj desde un servidor HTTP de la red local
# Not Before=time-sync.target: on this network nothing ever reaches that target, and ordering
# against it would hold the boot.
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/local/bin/reloj_lan.sh $URLS
# It is allowed to fail. A board whose clock could not be set still has to boot and fly; the
# failure is in the journal and the bench scripts set the clock again anyway.
SuccessExitStatus=0 1
TimeoutStartSec=700

[Install]
WantedBy=multi-user.target
UNIDAD

echo "-- habilitando"
ssh -n -o ConnectTimeout=10 "$PI" 'sudo systemctl daemon-reload && sudo systemctl enable reloj-lan'

echo "-- probandolo AHORA, que es la unica forma de saber que anda"
ssh -n -o ConnectTimeout=10 "$PI" 'sudo systemctl start reloj-lan; systemctl is-enabled reloj-lan; journalctl -u reloj-lan -b 0 --no-pager | tail -3'
echo "== $PI: el reloj se arregla solo en cada arranque"
