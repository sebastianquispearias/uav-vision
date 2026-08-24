"""Incremental identity: from per-frame detections to named POIs, online.

This is the live version of the logic validated offline on flight 3
(drone-geolocation/entrenamiento/correr_botsort.py): image-plane tracks
first (strong signal), ground projection only to place tracks that
already exist (weak signal, 2-3 m of telemetry noise). The offline run
separated the operator (674 obs, 2.49 m) from the equipment box and
found 4 walkers with trajectories; these are the same association rules
made incremental.

Division of labour
------------------
- The CAMERA owns image-plane tracking (it has the image and the boxes;
  on the drone that is the official BoT-SORT). Each detection arrives
  here already carrying a track_id.
- This module owns everything after: accumulate ground impacts per
  track, classify tracks static vs MOBILE, and merge tracks that are
  the same physical thing (position + appearance, with co-occurrence
  veto) into reportable candidates.

Why the thresholds are parameters, not constants
------------------------------------------------
The offline constants were tuned for one flight and silently encode the
scene: POS_M=3.5 assumes the 02ago telemetry noise, the n>=6/20/25
observation counts assume its ~0.7 FPS capture rate. Copied as-is they
would look general and fail quietly at another altitude or frame rate.
So here:

- radio_fusion_m has NO default. It is roughly the expected ground
  noise: gps_sigma + slant_range * yaw_sigma_rad (3.5 m fits the 02ago
  scene at 35 m altitude). The caller states it, like pitch_deg in the
  camera.
- observation counts are DURATIONS (seconds) times the fps the caller
  declares. The defaults (8.6 s / 29 s / 36 s) reproduce exactly the
  validated offline counts (6 / 20 / 25) at that flight's 0.7 FPS, and
  scale correctly at the 5 Hz live rate.
- emb_dist_max comes from a measurement, not taste: on flight 02ago,
  same-identity crops score cosine 0.63-0.87 and different identities
  0.33-0.46 (no overlap). The midpoint of the gap, cosine 0.545, is
  L2 distance sqrt(2 - 2*0.545) = 0.95 on normalised vectors.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# Measured on flight 02ago (see module docstring). Midpoint of the gap
# between same-identity and different-identity cosine similarity.
EMB_DIST_MAX_MEDIDO = 0.95

# Two tracks seen together in this many frames are two different
# physical things, whatever position and appearance say.
COOCURRENCIA_MIN = 3

# ...unless the evidence screams "same person": the detector sometimes
# emits duplicate boxes for one person, the tracker turns them into two
# co-occurring tracks, and the veto then forbids the obvious merge (the
# residual-double problem seen offline: the operator split #4/#6).
# Override thresholds, from the measured embedding separation: same
# identity scores distance <= 0.86, different identities >= 1.04, so
# 0.70 is deep inside the same-identity zone; and the position gate is
# far tighter than the fusion radius because a duplicate box lands on
# the SAME spot, not merely nearby.
EMB_DIST_GEMELO = 0.70
POS_FRAC_GEMELO = 0.4


def _posicion_actual(ii: np.ndarray) -> np.ndarray:
    """Where the track is NOW, from its recent impacts.

    The naive answer -- median of the last quarter -- systematically
    LAGS a walker: it is the centre of the recent past, so at 1 m/s and
    a 30-observation window it points ~half a window behind the person.
    A straight-line fit over the same window, evaluated at the last
    sample, removes that lag while still averaging the projection
    noise. (Observation index stands in for time: the camera timer
    fires at a fixed rate, so samples are near-uniform.)
    """
    q = max(2, len(ii) // 4)
    v = ii[-q:]
    if len(v) < 3:
        return np.median(v, axis=0)
    idx = np.arange(len(v), dtype=float)
    ajuste = np.polynomial.polynomial.polyfit(idx, v, 1)   # (2, 2): b, m
    return np.asarray(ajuste[0] + ajuste[1] * idx[-1])


class IdentidadIncremental:
    """Accumulates tracked detections; produces candidates on demand.

    observar() is O(1) per detection. candidatos() re-associates all
    track summaries from scratch; tracks number in the tens, so doing
    it at every report tick (2 s) is negligible next to the detector.
    Re-deriving from summaries instead of patching a live clustering
    keeps the semantics identical to the validated offline run.
    """

    def __init__(
        self,
        radio_fusion_m: float,
        fps: float,
        emb_dist_max: float = EMB_DIST_MAX_MEDIDO,
        dur_pista_s: float = 8.6,
        dur_movil_s: float = 29.0,
        dur_reporte_s: float = 36.0,
        desplaz_movil_m: Optional[float] = None,
    ) -> None:
        self.radio_fusion_m = radio_fusion_m
        self.emb_dist_max = emb_dist_max
        # Static tracks wander by projection noise only, so anything
        # moving farther than the fusion radius (plus margin) walked.
        # 1.15 * 3.5 = 4.0, the value validated offline.
        self.desplaz_movil_m = (desplaz_movil_m if desplaz_movil_m is not None
                                else 1.15 * radio_fusion_m)
        self.n_pista = max(3, round(dur_pista_s * fps))
        self.n_movil = max(6, round(dur_movil_s * fps))
        self.n_reporte = max(8, round(dur_reporte_s * fps))

        self._tracks: Dict[int, dict] = {}

    # -- ingest ------------------------------------------------------------

    def observar(
        self,
        frame: int,
        track_id: int,
        impacto_xy: Tuple[float, float],
        conf: float,
        emb: Optional[np.ndarray] = None,
    ) -> None:
        """One tracked detection, already projected to the ground."""
        t = self._tracks.get(track_id)
        if t is None:
            t = {"imps": [], "conf_sum": 0.0,
                 "emb_sum": None, "n_emb": 0, "frames": set()}
            self._tracks[track_id] = t
        t["imps"].append((float(impacto_xy[0]), float(impacto_xy[1])))
        t["conf_sum"] += float(conf)
        t["frames"].add(int(frame))
        if emb is not None:
            v = np.asarray(emb, dtype=np.float32)
            t["emb_sum"] = v.copy() if t["emb_sum"] is None else t["emb_sum"] + v
            t["n_emb"] += 1

    # -- association (the offline rules, on current summaries) ------------

    def _resumen_pistas(self) -> List[dict]:
        pistas = []
        for tid, t in self._tracks.items():
            n = len(t["imps"])
            if n < self.n_pista:
                continue
            ii = np.asarray(t["imps"])
            q = max(1, n // 4)
            desplaz = float(np.linalg.norm(
                np.median(ii[:q], axis=0) - np.median(ii[-q:], axis=0)))
            emb = None
            if t["n_emb"] > 0:
                emb = t["emb_sum"] / (np.linalg.norm(t["emb_sum"]) + 1e-9)
            pistas.append({
                "tid": tid, "n": n,
                "pos": np.median(ii, axis=0),        # robust centre
                "pos_actual": _posicion_actual(ii),
                "desplaz": desplaz,
                "conf": t["conf_sum"] / n,
                "emb": emb,
                "frames": t["frames"],
            })
        return pistas

    def candidatos(self) -> List[dict]:
        """Current candidate list, biggest evidence first, mobiles first.

        Each candidate: {x, y, n_obs, conf, movil}. For a MOBILE
        candidate (x, y) is its CURRENT position (median of the last
        quarter of its impacts), not its lifetime median -- a walker's
        lifetime median points at the middle of the path, which is
        nowhere. Static candidates keep the lifetime median, which is
        the whole point of accumulating views.
        """
        cands: List[dict] = []
        for tk in sorted(self._resumen_pistas(), key=lambda p: -p["n"]):
            if tk["desplaz"] > self.desplaz_movil_m and tk["n"] >= self.n_movil:
                cands.append({"movil": True, "pos": tk["pos_actual"].copy(),
                              "emb": tk["emb"], "conf": tk["conf"],
                              "n": tk["n"], "frames": set(tk["frames"])})
                continue
            mejor, smin = None, math.inf
            for k, c in enumerate(cands):
                if c["movil"]:
                    continue
                dp = float(np.linalg.norm(tk["pos"] - c["pos"]))
                if len(tk["frames"] & c["frames"]) >= COOCURRENCIA_MIN:
                    # seen together: two different things -- except the
                    # duplicate-box case (same spot + same appearance)
                    es_gemelo = (
                        dp < POS_FRAC_GEMELO * self.radio_fusion_m
                        and tk["emb"] is not None and c["emb"] is not None
                        and float(np.linalg.norm(tk["emb"] - c["emb"]))
                        < EMB_DIST_GEMELO)
                    if not es_gemelo:
                        continue
                if dp >= self.radio_fusion_m:
                    continue
                if tk["emb"] is not None and c["emb"] is not None:
                    de = float(np.linalg.norm(tk["emb"] - c["emb"]))
                    if de >= self.emb_dist_max:
                        continue
                    s = dp / self.radio_fusion_m + 0.5 * de / self.emb_dist_max
                else:
                    s = dp / self.radio_fusion_m
                if s < smin:
                    mejor, smin = k, s
            if mejor is None:
                cands.append({"movil": False, "pos": tk["pos"].copy(),
                              "emb": tk["emb"], "conf": tk["conf"],
                              "n": tk["n"], "frames": set(tk["frames"])})
            else:
                c = cands[mejor]
                w = c["n"] / (c["n"] + tk["n"])
                c["pos"] = w * c["pos"] + (1 - w) * tk["pos"]
                if c["emb"] is not None and tk["emb"] is not None:
                    e = w * c["emb"] + (1 - w) * tk["emb"]
                    c["emb"] = e / (np.linalg.norm(e) + 1e-9)
                c["conf"] = w * c["conf"] + (1 - w) * tk["conf"]
                c["n"] += tk["n"]
                c["frames"] |= tk["frames"]

        listos = [c for c in cands if c["n"] >= self.n_reporte]
        listos.sort(key=lambda c: (not c["movil"], -c["n"]))
        return [{
            "x": round(float(c["pos"][0]), 2),
            "y": round(float(c["pos"][1]), 2),
            "n_obs": int(c["n"]),
            "conf": round(float(c["conf"]), 3),
            "movil": bool(c["movil"]),
        } for c in listos]
