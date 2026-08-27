# -*- coding: utf-8 -*-
"""Invariants for derived quantities, enforced in code rather than documented as warnings.

The failure mode these guard against is a specific one: a derived number that is wrong but
plausible. Nothing raises, the value prints, and the error surfaces later as a conclusion that
is quietly off in a consistent direction -- a rate, a maturity threshold, a cached column.

Enforcing them as functions rather than as conventions is deliberate. A convention has to be
remembered at the moment of writing; a function only has to be the easiest thing to call.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Sequence

import numpy as np


class CorduraError(Exception):
    """A derived quantity is not what it was declared to be."""


def cadencia_instantanea(tiempos: Sequence[float], avisar: bool = True) -> float:
    """
    The rate a stream of timestamped samples actually arrives at, in Hz.

    The only correct way to derive a rate here. Counting samples over the wall-clock span
    understates it whenever the recording has gaps:

        flight 3: 2983 frames over a 909 s span says 3.28 Hz.
        The same flight's median interval says 9.35 Hz.

    Both describe the same recording. The span includes 573 s of the drone sitting on the
    ground between passes, and dividing by it treats that as if the camera were running. The
    consequences are never a crash: a tracker buffer three times too small, a duty-cycle floor
    three times too loose, maturity thresholds that make "36 s" mean something else.

    The median of the intervals is used rather than the mean because a handful of long gaps --
    landings, dropped frames, a pass that ended -- must not move it, and that is exactly the
    shape of the data that caused the trouble.

    Args:
        tiempos: sample times in seconds, ascending. Needs at least two.
        avisar: print a note when the span-based figure disagrees badly, which is the moment
            the next person is about to reach for it.
    """
    t = np.asarray(tiempos, dtype="float64")
    if t.size < 2:
        raise CorduraError("hacen falta al menos dos tiempos para una cadencia; hay %d" % t.size)
    dt = np.diff(t)
    positivos = dt[dt > 0]
    if positivos.size == 0:
        raise CorduraError("ningun intervalo positivo: los tiempos no estan ordenados")
    cadencia = float(1.0 / np.median(positivos))

    if avisar:
        span = float(t[-1] - t[0])
        por_span = (t.size / span) if span > 0 else float("inf")
        if por_span < 0.8 * cadencia:
            print("[cordura] la cadencia por span daria %.2f Hz y la real es %.2f Hz: "
                  "el registro tiene huecos (%.0f s de %.0f). No dividas por el span."
                  % (por_span, cadencia, span - t.size / cadencia, span))
    return cadencia


def verificar_tasa(pedida: float, lograda: float, tolerancia: float = 0.10,
                   que: str = "la tasa") -> None:
    """
    Raises unless the rate delivered is the rate asked for.

    A configured rate is a claim about the world, and worth checking against what arrives.
    Measured cases: 3.00 declared against 2.31 delivered on the board, and 3.00 asked of a
    subsampler that returned 2.55. Neither discrepancy showed in any output being read.
    """
    if pedida <= 0:
        raise CorduraError("%s pedida no puede ser %.3f" % (que, pedida))
    desvio = abs(lograda - pedida) / pedida
    if desvio > tolerancia:
        raise CorduraError(
            "%s no es la que se pidio: %.2f pedida, %.2f lograda (%.0f%% de desvio, "
            "el limite es %.0f%%)" % (que, pedida, lograda, 100 * desvio, 100 * tolerancia))


def _firma(params: Dict[str, Any]) -> str:
    """A stable text form of the parameters that produced a cache."""
    return json.dumps(params, sort_keys=True, default=str)


def guardar_cache(ruta: str, params: Dict[str, Any], **arreglos) -> None:
    """Stores arrays together with the parameters that produced them."""
    np.savez(ruta, _firma=np.array(_firma(params)), **arreglos)


def cargar_cache(ruta: str, params: Dict[str, Any],
                 callado: bool = False) -> Optional[Dict[str, np.ndarray]]:
    """
    Returns the cached arrays, or None if this cache was not built from these parameters.

    A cache without a stamp is the most expensive kind of stale data, because nothing about it
    looks wrong: a file that outlives the parameters that built it will go on supplying the
    reference column of a comparison. Refusing to load an unstamped or mismatched cache costs
    one rebuild; trusting one costs the conclusion.
    """
    if not os.path.exists(ruta):
        return None
    try:
        d = np.load(ruta, allow_pickle=False)
    except Exception as exc:
        if not callado:
            print("[cordura] %s no se pudo leer (%s): se reconstruye" % (ruta, exc))
        return None
    guardada = str(d["_firma"]) if "_firma" in d.files else None
    quiere = _firma(params)
    if guardada != quiere:
        if not callado:
            motivo = "sin firma (de antes de este mecanismo)" if guardada is None \
                else "firma distinta"
            print("[cordura] %s se ignora: %s.\n           pedido: %s\n           guardado: %s"
                  % (os.path.basename(ruta), motivo, quiere, guardada))
        return None
    return {k: d[k] for k in d.files if k != "_firma"}
