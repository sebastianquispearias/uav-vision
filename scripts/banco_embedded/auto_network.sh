#!/usr/bin/env bash
#
# Configure a drone companion computer's network from the flight controller's
# SYSID, so that flashing a card and plugging in the flight controller is all it
# takes to put a Raspberry Pi (Rpanion image) on the fleet network.
#
# What it does, on every run:
#   1. writes the drone1..drone10 peer table into /etc/hosts
#   2. reads the SYSID from the flight controller: the sysid byte of the
#      HEARTBEAT that mavlink-router forwards to UDP 127.0.0.1:17171 (this is
#      SYSID_THISMAV; the FC stamps it on every message it sends)
#   3. sets the hostname to drone<SYSID>
#   4. joins the "skynet" Wi-Fi network as 192.168.1.(100+SYSID)/24
#   5. if there is no SYSID, or skynet cannot be joined, disables Rpanion's
#      own hotspot ("WiFiAP", SSID "rpanion") and brings up the
#      "drone_undefined" access point instead so the Pi stays reachable
#
# Everything goes through NetworkManager (nmcli), which is what the Rpanion
# image uses and what Rpanion's own web UI drives.
#
# It deliberately does NOT: delete any NetworkManager profile (Rpanion's hotspot
# is only switched to autoconnect=no and brought down), set SYSID_THISMAV on the flight
# controller, configure mavlink-router (Rpanion owns it; it must already output
# to 127.0.0.1:17171), or open the serial port.
#
# Usage:
#   sudo ./auto_network.sh                 # run once now
#   sudo ./auto_network.sh --install       # install as a boot-time systemd unit
#   ./auto_network.sh --dry-run --sysid 3  # show what would be done
set -euo pipefail

# --- defaults ---------------------------------------------------------------
WIFI_SSID="${WIFI_SSID:-skynet}"
WIFI_PSK="${WIFI_PSK:-gradysgradys}"
WIFI_IFACE="${WIFI_IFACE:-wlan0}"
WIFI_CON="${WIFI_CON:-skynet}"
IP_BASE="${IP_BASE:-192.168.1.100}"      # SYSID is added to the last octet
IP_PREFIX="${IP_PREFIX:-24}"
IP_GATEWAY="${IP_GATEWAY:-192.168.1.1}"
IP_DNS="${IP_DNS:-1.1.1.1,8.8.8.8}"
AP_SSID="${AP_SSID:-drone_undefined}"
AP_PSK="${AP_PSK:-rpanion123}"
AP_CON="${AP_CON:-drone_undefined}"
AP_ADDR="${AP_ADDR:-10.0.2.100/24}"      # same address Rpanion's own hotspot used
RPANION_AP_CON="${RPANION_AP_CON:-WiFiAP}" # Rpanion's hotspot profile (deploy/wifi_access_point.sh)
MAVLINK_PORT="${MAVLINK_PORT:-17171}"
MAVLINK_TIMEOUT="${MAVLINK_TIMEOUT:-90}" # seconds to wait for a FC heartbeat
HOSTS_COUNT="${HOSTS_COUNT:-10}"
HOSTS_PREFIX="${HOSTS_PREFIX:-drone}"
CONNECT_ATTEMPTS=3
CONNECT_WAIT=45                          # seconds nmcli waits per attempt
SYSID_MAX=154                            # 100 + 154 = 254, last usable octet

HOSTS_FILE="${HOSTS_FILE:-/etc/hosts}"
HOSTS_BEGIN="# BEGIN auto-network drones"
HOSTS_END="# END auto-network drones"
INSTALL_BIN=/usr/local/sbin/auto-network
INSTALL_UNIT=/etc/systemd/system/auto-network.service

SYSID_OVERRIDE=""
DRY_RUN=0
MODE=run

# Exit codes
EXIT_OK=0        # joined skynet as drone<SYSID>
EXIT_ERROR=1     # preflight or hard failure
EXIT_FALLBACK=3  # running the drone_undefined access point

usage() {
  cat <<EOF
Usage: sudo $0 [options]
       sudo $0 --install | --uninstall

Options:
  --sysid N          Use this SYSID instead of reading it from the flight
                     controller. 0 or out of range (1..${SYSID_MAX}) means
                     "undefined" and takes the access-point fallback.
  --timeout SECONDS  How long to wait for a heartbeat on UDP port
                     ${MAVLINK_PORT} (default: ${MAVLINK_TIMEOUT}).
  --port PORT        UDP port mavlink-router sends to (default: ${MAVLINK_PORT}).
  --iface DEV        Wi-Fi interface (default: ${WIFI_IFACE}).
  --dry-run          Print the commands instead of running them. The SYSID
                     is still read from UDP unless --sysid is given.
  --install          Copy this script to ${INSTALL_BIN} and enable
                     auto-network.service (unit file must sit next to this
                     script). Does not run it.
  --uninstall        Disable the unit and remove both files. Leaves the
                     NetworkManager profiles alone.
  -h, --help         This message.

Result:      hostname drone<SYSID>, Wi-Fi "${WIFI_SSID}" as ${IP_BASE%.*}.(${IP_BASE##*.}+SYSID)/${IP_PREFIX}
Fallback:    Rpanion's hotspot "${RPANION_AP_CON}" disabled, access point "${AP_SSID}"
             (password "${AP_PSK}") brought up on ${AP_ADDR}
Exit codes:  ${EXIT_OK} on ${WIFI_SSID}, ${EXIT_FALLBACK} on the fallback AP, ${EXIT_ERROR} on error.
EOF
}

die()  { echo "error: $*" >&2; exit "$EXIT_ERROR"; }
say()  { echo ">> $*"; }
warn() { echo "warning: $*" >&2; }

# run CMD...: execute, or print when --dry-run.
run() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '+'; printf ' %q' "$@"; printf '\n'
  else
    "$@"
  fi
}

# --- argument parsing -------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sysid)     SYSID_OVERRIDE="${2:?--sysid needs a value}"; shift 2 ;;
    --timeout)   MAVLINK_TIMEOUT="${2:?--timeout needs a value}"; shift 2 ;;
    --port)      MAVLINK_PORT="${2:?--port needs a value}"; shift 2 ;;
    --iface)     WIFI_IFACE="${2:?--iface needs a value}"; shift 2 ;;
    --dry-run)   DRY_RUN=1; shift ;;
    --install)   MODE=install; shift ;;
    --uninstall) MODE=uninstall; shift ;;
    -h|--help)   usage; exit 0 ;;
    -*)          usage >&2; die "unknown option: $1" ;;
    *)           usage >&2; die "unexpected argument '$1'. This script takes flags only." ;;
  esac
done

# --- helpers ----------------------------------------------------------------
# ip_for N: the address for SYSID N, i.e. IP_BASE with N added to the last octet.
ip_for() {
  local last=$(( ${IP_BASE##*.} + $1 ))
  (( last <= 254 )) || die "SYSID $1 puts the address past .254 (${IP_BASE%.*}.${last})"
  echo "${IP_BASE%.*}.${last}"
}

con_exists() { nmcli -t -f NAME con show 2>/dev/null | grep -Fxq -- "$1"; }

# ensure_wifi_profile NAME IFACE SSID PROP VALUE...: create the profile or bring
# an existing one in line, without dropping it if it is currently active.
ensure_wifi_profile() {
  local name="$1" iface="$2" ssid="$3"; shift 3
  if con_exists "$name"; then
    run nmcli con modify "$name" connection.interface-name "$iface" 802-11-wireless.ssid "$ssid" "$@"
  else
    run nmcli con add type wifi ifname "$iface" con-name "$name" ssid "$ssid" "$@"
  fi
}

# --- install / uninstall ----------------------------------------------------
install_service() {
  local here unit
  here="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
  unit="${here}/auto-network.service"
  [[ -f "$unit" ]] || die "unit file not found next to this script: $unit"
  [[ "$DRY_RUN" -eq 1 || "$EUID" -eq 0 ]] || die "must run as root: sudo $0 --install"
  say "[1/3] installing script to ${INSTALL_BIN}"
  run install -m 0755 "$(readlink -f "$0")" "$INSTALL_BIN"
  say "[2/3] installing unit to ${INSTALL_UNIT}"
  run install -m 0644 "$unit" "$INSTALL_UNIT"
  say "[3/3] enabling auto-network.service"
  run systemctl daemon-reload
  run systemctl enable auto-network.service
  cat <<EOF

Installed. It runs at every boot; to run it now and watch:
  sudo systemctl start auto-network
  journalctl -u auto-network -b
EOF
}

uninstall_service() {
  [[ "$DRY_RUN" -eq 1 || "$EUID" -eq 0 ]] || die "must run as root: sudo $0 --uninstall"
  say "disabling and removing auto-network.service"
  run systemctl disable auto-network.service || true
  run rm -f "$INSTALL_UNIT" "$INSTALL_BIN"
  run systemctl daemon-reload
  say "done. NetworkManager profiles '${WIFI_CON}' and '${AP_CON}' were left in place."
}

case "$MODE" in
  install)   install_service; exit 0 ;;
  uninstall) uninstall_service; exit 0 ;;
esac

# --- [1/5] preflight --------------------------------------------------------
say "[1/5] preflight"
[[ "$DRY_RUN" -eq 1 || "$EUID" -eq 0 ]] || die "must run as root: sudo $0 ..."
for tool in nmcli hostnamectl python3; do
  command -v "$tool" >/dev/null 2>&1 || die "missing required tool: $tool"
done
if ! nmcli -t -f DEVICE dev status 2>/dev/null | grep -Fxq -- "$WIFI_IFACE"; then
  if [[ "$DRY_RUN" -eq 1 ]]; then
    warn "NetworkManager does not know interface '${WIFI_IFACE}' (ignored in dry-run)"
  else
    die "NetworkManager does not know interface '${WIFI_IFACE}'"
  fi
fi
[[ "$HOSTS_COUNT" =~ ^[0-9]+$ ]] || die "HOSTS_COUNT must be a number"
[[ "$MAVLINK_TIMEOUT" =~ ^[0-9]+$ ]] || die "--timeout must be a whole number of seconds"

# --- [2/5] /etc/hosts peer table --------------------------------------------
# Independent of this drone's own SYSID, so it is written on every path.
write_hosts_block() {
  local block i
  block="$(
    echo "$HOSTS_BEGIN"
    for ((i = 1; i <= HOSTS_COUNT; i++)); do
      printf '%s\t%s%d\n' "$(ip_for "$i")" "$HOSTS_PREFIX" "$i"
    done
    echo "$HOSTS_END"
  )"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "+ would replace the managed block in ${HOSTS_FILE} with:"
    sed 's/^/    /' <<<"$block"
    return
  fi
  local tmp
  tmp="$(mktemp)"
  awk -v b="$HOSTS_BEGIN" -v e="$HOSTS_END" \
    '$0 == b { skip = 1; next } $0 == e { skip = 0; next } !skip' "$HOSTS_FILE" >"$tmp"
  printf '%s\n' "$block" >>"$tmp"
  install -m 0644 "$tmp" "$HOSTS_FILE"
  rm -f "$tmp"
}
say "[2/5] writing ${HOSTS_PREFIX}1..${HOSTS_PREFIX}${HOSTS_COUNT} into ${HOSTS_FILE}"
write_hosts_block

# --- [3/5] read the SYSID ---------------------------------------------------
# Prints the sysid of the first HEARTBEAT (msgid 0) that comes from an
# autopilot (compid 1, not a GCS, autopilot != INVALID) with a valid checksum.
# Exit 1 on timeout, 2 if the port is already taken.
get_sysid() {
  python3 - "$MAVLINK_PORT" "$MAVLINK_TIMEOUT" <<'PY'
import socket, sys, time

port, timeout = int(sys.argv[1]), float(sys.argv[2])
HEARTBEAT_CRC_EXTRA = 50
MAV_COMP_ID_AUTOPILOT1 = 1
MAV_TYPE_GCS = 6
MAV_AUTOPILOT_INVALID = 8


def x25(data, crc=0xFFFF):
    for b in data:
        tmp = (b ^ (crc & 0xFF)) & 0xFF
        tmp = (tmp ^ (tmp << 4)) & 0xFF
        crc = ((crc >> 8) ^ (tmp << 8) ^ (tmp << 3) ^ (tmp >> 4)) & 0xFFFF
    return crc


def frame_at(buf, i):
    """Decode the MAVLink frame starting at buf[i]. Returns
    (sysid, compid, msgid, payload) or None if it is not a valid frame."""
    magic = buf[i]
    if magic == 0xFD:                       # MAVLink 2
        if i + 10 > len(buf):
            return None
        plen = buf[i + 1]
        sysid, compid = buf[i + 5], buf[i + 6]
        msgid = buf[i + 7] | (buf[i + 8] << 8) | (buf[i + 9] << 16)
        hdr, payload, crc_at = buf[i + 1:i + 10], buf[i + 10:i + 10 + plen], i + 10 + plen
    elif magic == 0xFE:                     # MAVLink 1
        if i + 6 > len(buf):
            return None
        plen = buf[i + 1]
        sysid, compid, msgid = buf[i + 3], buf[i + 4], buf[i + 5]
        hdr, payload, crc_at = buf[i + 1:i + 6], buf[i + 6:i + 6 + plen], i + 6 + plen
    else:
        return None
    if msgid != 0 or len(payload) != plen or crc_at + 2 > len(buf):
        return None
    crc = x25(bytes([HEARTBEAT_CRC_EXTRA]), x25(payload, x25(hdr)))
    if crc != (buf[crc_at] | (buf[crc_at + 1] << 8)):
        return None
    return sysid, compid, msgid, payload


def autopilot_sysid(buf):
    for i in range(len(buf)):
        f = frame_at(buf, i)
        if f is None:
            continue
        sysid, compid, _, p = f
        # MAVLink 2 trims trailing zero bytes; missing fields read as 0.
        mtype = p[4] if len(p) > 4 else 0
        autopilot = p[5] if len(p) > 5 else 0
        if (sysid != 0 and compid == MAV_COMP_ID_AUTOPILOT1
                and mtype != MAV_TYPE_GCS and autopilot != MAV_AUTOPILOT_INVALID):
            return sysid
    return None


try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", port))
except OSError as exc:
    print(f"cannot bind UDP port {port}: {exc} (is vehicle-api or another "
          f"listener already using it?)", file=sys.stderr)
    sys.exit(2)

deadline = time.monotonic() + timeout
while True:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        print(f"no autopilot heartbeat on UDP port {port} within {timeout:.0f}s",
              file=sys.stderr)
        sys.exit(1)
    sock.settimeout(remaining)
    try:
        data, _ = sock.recvfrom(65535)
    except socket.timeout:
        continue
    sysid = autopilot_sysid(data)
    if sysid is not None:
        print(sysid)
        sys.exit(0)
PY
}

SYSID=""
if [[ -n "$SYSID_OVERRIDE" ]]; then
  say "[3/5] using SYSID ${SYSID_OVERRIDE} from --sysid"
  SYSID="$SYSID_OVERRIDE"
else
  say "[3/5] waiting up to ${MAVLINK_TIMEOUT}s for a flight controller heartbeat on UDP ${MAVLINK_PORT}"
  if ! SYSID="$(get_sysid)"; then
    warn "could not read the SYSID from the flight controller"
    SYSID=""
  fi
fi

if [[ -n "$SYSID" ]] && ! { [[ "$SYSID" =~ ^[0-9]+$ ]] && (( SYSID >= 1 && SYSID <= SYSID_MAX )); }; then
  warn "SYSID '${SYSID}' is outside 1..${SYSID_MAX}; treating it as undefined"
  SYSID=""
fi

# --- [4/5] hostname + skynet -------------------------------------------------
set_hostname() {
  local name="$1"
  run hostnamectl set-hostname "$name"
  # Debian keeps the local hostname on the 127.0.1.1 line; without it sudo
  # complains "unable to resolve host" after every rename.
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "+ would point 127.0.1.1 at ${name} in ${HOSTS_FILE}"
  elif grep -Eq '^127\.0\.1\.1[[:space:]]' "$HOSTS_FILE"; then
    sed -i -E "s/^127\.0\.1\.1[[:space:]].*/127.0.1.1\t${name}/" "$HOSTS_FILE"
  else
    printf '127.0.1.1\t%s\n' "$name" >>"$HOSTS_FILE"
  fi
}

# connect_wifi ADDRESS: (re)build the skynet profile and activate it.
connect_wifi() {
  local addr="$1" attempt state
  ensure_wifi_profile "$WIFI_CON" "$WIFI_IFACE" "$WIFI_SSID" \
    802-11-wireless.mode infrastructure \
    wifi-sec.key-mgmt wpa-psk wifi-sec.psk "$WIFI_PSK" \
    ipv4.method manual ipv4.addresses "${addr}/${IP_PREFIX}" \
    ipv4.gateway "$IP_GATEWAY" ipv4.dns "$IP_DNS" \
    ipv6.method disabled \
    connection.autoconnect yes connection.autoconnect-priority 10
  for ((attempt = 1; attempt <= CONNECT_ATTEMPTS; attempt++)); do
    say "    attempt ${attempt}/${CONNECT_ATTEMPTS}: nmcli con up ${WIFI_CON}"
    run nmcli dev wifi rescan ifname "$WIFI_IFACE" 2>/dev/null || true
    if run nmcli --wait "$CONNECT_WAIT" con up "$WIFI_CON"; then
      [[ "$DRY_RUN" -eq 1 ]] && return 0
      state="$(nmcli -g GENERAL.STATE con show "$WIFI_CON" 2>/dev/null || true)"
      if [[ "$state" == "activated" ]]; then
        if ping -c 1 -W 2 "$IP_GATEWAY" >/dev/null 2>&1; then
          say "    gateway ${IP_GATEWAY} answers ping"
        else
          warn "gateway ${IP_GATEWAY} does not answer ping (associated anyway)"
        fi
        return 0
      fi
      warn "connection '${WIFI_CON}' is '${state:-unknown}', not activated"
    fi
    (( attempt < CONNECT_ATTEMPTS )) && sleep 5
  done
  return 1
}

# disable_other_hotspots: Rpanion's own hotspot (and any other AP-mode profile
# that is not ours) must not come back on wlan0 and fight drone_undefined for
# the interface. Profiles are kept, only switched to autoconnect=no and
# brought down, so Rpanion's web UI can still re-enable them.
disable_other_hotspots() {
  local con mode active
  while IFS= read -r con; do
    [[ -n "$con" && "$con" != "$AP_CON" ]] || continue
    mode="$(nmcli -g 802-11-wireless.mode con show "$con" 2>/dev/null || true)"
    [[ "$con" == "$RPANION_AP_CON" || "$mode" == "ap" ]] || continue
    say "    disabling hotspot profile '${con}'"
    run nmcli con modify "$con" connection.autoconnect no
    active="$(nmcli -g GENERAL.STATE con show "$con" 2>/dev/null || true)"
    if [[ "$active" == "activated" || "$active" == "activating" ]]; then
      run nmcli con down "$con" || warn "could not bring down '${con}'"
    fi
  done < <(nmcli -g NAME con show 2>/dev/null || true)
}

start_ap() {
  disable_other_hotspots
  ensure_wifi_profile "$AP_CON" "$WIFI_IFACE" "$AP_SSID" \
    802-11-wireless.mode ap 802-11-wireless.band bg \
    wifi-sec.key-mgmt wpa-psk wifi-sec.psk "$AP_PSK" \
    ipv4.method shared ipv4.addresses "$AP_ADDR" \
    ipv6.method disabled \
    connection.autoconnect yes connection.autoconnect-priority -10
  run nmcli --wait "$CONNECT_WAIT" con up "$AP_CON"
}

RESULT="$EXIT_FALLBACK"
if [[ -n "$SYSID" ]]; then
  HOSTNAME_NEW="${HOSTS_PREFIX}${SYSID}"
  ADDR="$(ip_for "$SYSID")"
  say "[4/5] SYSID ${SYSID}: hostname ${HOSTNAME_NEW}, ${WIFI_SSID} as ${ADDR}/${IP_PREFIX}"
  set_hostname "$HOSTNAME_NEW"
  if connect_wifi "$ADDR"; then
    RESULT="$EXIT_OK"
  else
    warn "could not join '${WIFI_SSID}' after ${CONNECT_ATTEMPTS} attempts"
  fi
else
  say "[4/5] no SYSID: hostname left as '$(hostname)'"
fi

# --- [5/5] fallback access point ----------------------------------------------
if [[ "$RESULT" -ne "$EXIT_OK" ]]; then
  say "[5/5] falling back to access point '${AP_SSID}' on ${AP_ADDR}"
  if ! start_ap; then
    die "could not bring up the fallback access point '${AP_CON}'"
  fi
else
  say "[5/5] skipping the fallback access point"
fi

# --- summary ------------------------------------------------------------------
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo
  echo "dry-run: nothing was changed."
  exit "$RESULT"
fi
cat <<EOF

Summary
  SYSID:       ${SYSID:-undefined}
  hostname:    $(hostname)
  active:      $(nmcli -t -f NAME,DEVICE con show --active 2>/dev/null | grep -F ":${WIFI_IFACE}" | cut -d: -f1 || true)
  address:     $(ip -4 -brief addr show "$WIFI_IFACE" 2>/dev/null | awk '{print $3}')
  result:      $([[ "$RESULT" -eq "$EXIT_OK" ]] && echo "on ${WIFI_SSID}" || echo "fallback access point ${AP_SSID} (exit ${EXIT_FALLBACK})")
EOF
exit "$RESULT"
