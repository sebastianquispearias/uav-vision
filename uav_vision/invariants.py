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


class InvariantError(Exception):
    """A derived quantity is not what it was declared to be."""


def instantaneous_rate(times: Sequence[float], warn: bool = True) -> float:
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
        times: sample times in seconds, ascending. Needs at least two.
        warn: print a note when the span-based figure disagrees badly, which is the moment
            the next person is about to reach for it.
    """
    t = np.asarray(times, dtype="float64")
    if t.size < 2:
        raise InvariantError("a rate needs at least two timestamps; got %d" % t.size)
    dt = np.diff(t)
    positive = dt[dt > 0]
    if positive.size == 0:
        raise InvariantError("no positive interval: the timestamps are not in order")
    rate = float(1.0 / np.median(positive))

    if warn:
        span = float(t[-1] - t[0])
        by_span = (t.size / span) if span > 0 else float("inf")
        if by_span < 0.8 * rate:
            print("[invariants] the span-based rate would say %.2f Hz and the real one is "
                  "%.2f Hz: the recording has gaps (%.0f s of %.0f). Do not divide by span."
                  % (by_span, rate, span - t.size / rate, span))
    return rate


def check_rate(requested: float, achieved: float, tolerance: float = 0.10,
               what: str = "the rate") -> None:
    """
    Raises unless the rate delivered is the rate asked for.

    A configured rate is a claim about the world, and worth checking against what arrives.
    Measured cases: 3.00 declared against 2.31 delivered on the board, and 3.00 asked of a
    subsampler that returned 2.55. Neither discrepancy showed in any output being read.
    """
    if requested <= 0:
        raise InvariantError("%s requested cannot be %.3f" % (what, requested))
    deviation = abs(achieved - requested) / requested
    if deviation > tolerance:
        raise InvariantError(
            "%s is not what was asked for: %.2f requested, %.2f achieved (%.0f%% off, "
            "the limit is %.0f%%)" % (what, requested, achieved, 100 * deviation,
                                      100 * tolerance))


def _signature(params: Dict[str, Any]) -> str:
    """A stable text form of the parameters that produced a cache."""
    return json.dumps(params, sort_keys=True, default=str)


def save_cache(path: str, params: Dict[str, Any], **arrays) -> None:
    """Stores arrays together with the parameters that produced them."""
    np.savez(path, _signature=np.array(_signature(params)), **arrays)


def load_cache(path: str, params: Dict[str, Any],
               quiet: bool = False) -> Optional[Dict[str, np.ndarray]]:
    """
    Returns the cached arrays, or None if this cache was not built from these parameters.

    A cache without a stamp is the most expensive kind of stale data, because nothing about it
    looks wrong: a file that outlives the parameters that built it will go on supplying the
    reference column of a comparison. Refusing to load an unstamped or mismatched cache costs
    one rebuild; trusting one costs the conclusion.
    """
    if not os.path.exists(path):
        return None
    try:
        d = np.load(path, allow_pickle=False)
    except Exception as exc:
        if not quiet:
            print("[invariants] %s could not be read (%s): rebuilding" % (path, exc))
        return None
    stored = str(d["_signature"]) if "_signature" in d.files else None
    wanted = _signature(params)
    if stored != wanted:
        if not quiet:
            reason = "no signature (predates this mechanism)" if stored is None \
                else "different signature"
            print("[invariants] %s ignored: %s.\n           wanted: %s\n           stored: %s"
                  % (os.path.basename(path), reason, wanted, stored))
        return None
    return {k: d[k] for k in d.files if k != "_signature"}
