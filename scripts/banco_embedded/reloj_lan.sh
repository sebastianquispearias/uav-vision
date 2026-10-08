#!/usr/bin/env bash
# Sets this board's clock from any HTTP server on the LAN, with no internet and no NTP.
#
# WHY THIS HAS TO EXIST. A Raspberry has no battery-backed clock. With no route to the internet
# systemd-timesyncd never synchronises, so fake-hwclock restores the time of the last shutdown
# and the board stays there. Measured on the Pi 5 on 2026-10-08: it came up believing it was
# 2026-10-06 03:17, forty-seven hours behind, which was the moment it had been switched off.
#
# THE SYMPTOM IS NOT AN ERROR, which is why it kept costing time. tar warns that every file it
# unpacks is "in the future" and carries on. The ground station then reads reports stamped two
# days ago, decides every contact is stale, and shows an EMPTY SCREEN while the drone is
# reporting perfectly. Nothing anywhere says "the clock is wrong".
#
# WHERE THE TIME COMES FROM. Every HTTP response carries a Date header, in GMT, to the second.
# That is enough: what this clock is for is crossing the board's logs with the laptop's, not
# timing anything. So this asks whatever is already serving HTTP on the network.
#
# WHAT WAS TRIED AND DOES NOT WORK HERE, so nobody spends an evening on it again:
#   - systemd-timesyncd against the internet: this LAN has no route out. Verified by ping.
#   - systemd-timesyncd against the router at 192.168.1.1: the router answers ICMP but does NOT
#     serve NTP. "Timed out waiting for reply from 192.168.1.1:123".
#   - the router's Date header: it serves no HTTP on 80 or 443 either.
# What is left on this network is the laptop, and the ground station is an HTTP server that is
# up exactly when the time matters.
#
# THE REAL FIX IS A COIN CELL, and this does not replace it. The Pi 5 already has the RTC
# (/dev/rtc0 exists); it has no battery, so it reads 1970 at every boot. A CR2032 on the J5
# header keeps the time across power cycles with NO network at all, which is the only thing
# that works in a field on a flight day. This script is what covers the board until then, and
# it covers it well whenever the laptop is reachable.
#
# Usage, normally through the systemd unit that instalar_reloj.sh writes:
#     bash reloj_lan.sh http://192.168.1.121:8300/ http://192.168.1.121:8000/
# It tries each URL in turn, keeps retrying while REINTENTOS_S has not elapsed, and exits 0 the
# moment one of them answers.

set -u

INTERVALO_S="${RELOJ_INTERVALO_S:-10}"
REINTENTOS_S="${RELOJ_REINTENTOS_S:-600}"
# Nothing before this is a believable answer. A server with its own broken clock, or a proxy
# serving a cached header, would otherwise be able to drag this board into the past, and the
# whole point is that a wrong clock is invisible downstream.
PISO="2026-01-01 00:00:00"

if [ "$#" -eq 0 ]; then
    echo "uso: $0 <url> [url ...]" >&2
    exit 2
fi

piso_s="$(date -u -d "$PISO" +%s)"

fecha_de() {
    # The Date header of one HTTP response, or nothing. --head so no body is fetched, and a
    # short timeout because this runs at boot and must not hold it.
    curl -sS -I -m 5 "$1" 2>/dev/null \
        | tr -d '\r' \
        | sed -n 's/^[Dd][Aa][Tt][Ee]: *//p' \
        | head -1
}

fin=$(( $(date -u +%s) + REINTENTOS_S ))
while :; do
    for url in "$@"; do
        cabecera="$(fecha_de "$url")"
        [ -n "$cabecera" ] || continue

        nuevo_s="$(date -u -d "$cabecera" +%s 2>/dev/null)" || continue
        [ -n "$nuevo_s" ] || continue
        if [ "$nuevo_s" -lt "$piso_s" ]; then
            echo "reloj_lan: $url dice '$cabecera', anterior a $PISO. No lo creo." >&2
            continue
        fi

        antes="$(date -u '+%Y-%m-%d %H:%M:%S')"
        date -u -s "@$nuevo_s" > /dev/null || {
            echo "reloj_lan: no pude poner el reloj. Hace falta root." >&2
            exit 1
        }
        # So the NEXT boot starts from this instead of from the last shutdown. It does not make
        # the clock right on its own, but it stops the board from travelling backwards.
        command -v fake-hwclock > /dev/null && fake-hwclock save
        echo "reloj_lan: $antes UTC -> $(date -u '+%Y-%m-%d %H:%M:%S') UTC, segun $url"
        exit 0
    done

    if [ "$(date -u +%s)" -ge "$fin" ]; then
        echo "reloj_lan: ninguna de las $# fuentes contesto en ${REINTENTOS_S}s." >&2
        echo "reloj_lan: el reloj queda en $(date -u '+%Y-%m-%d %H:%M:%S') UTC, que puede estar mal." >&2
        exit 1
    fi
    sleep "$INTERVALO_S"
done
