"""
Incremental identity: turns tracked detections into named points of interest, online.

Division of labour:
    - The camera owns image-plane tracking (it has the image and the boxes). Every detection
      arrives here already carrying a track_id.
    - This module owns everything after: it accumulates ground impacts per track, classifies
      tracks as static or mobile, and merges tracks that correspond to the same physical thing
      (by position and appearance, with a co-occurrence veto) into reportable candidates.

Image-plane tracking runs first because it is the strong signal: two people two meters apart are
clearly separate in the image, while their ground projections overlap under telemetry noise.
Ground coordinates are used only to place tracks that already exist.

Thresholds are parameters rather than constants because they encode scene properties (ground
noise) and data rate. The provenance of every default is documented in NOTES.md.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

# Fusion threshold for appearance embeddings (L2 distance between unit vectors). Chosen at the
# midpoint of the measured gap between same-identity and different-identity scores; see NOTES.md.
EMB_DIST_MAX_MEDIDO = 0.95

# Two tracks seen together in this many frames are two different physical things, whatever
# position and appearance say: nothing appears twice in the same photo.
COOCURRENCIA_MIN = 3

# Minimum fraction of the frames a track spans in which it must actually have been detected.
# Below it the track is a handful of sightings spread thin, and its frame span overstates the
# evidence behind it.
DUTY_MIN = 0.10

# Exception to the veto: detectors sometimes emit duplicate boxes for one person, which the
# tracker turns into two co-occurring tracks. The veto is lifted only when the evidence says
# "same person" — nearly identical position AND an embedding distance deep inside the measured
# same-identity zone.
EMB_DIST_GEMELO = 0.70
POS_FRAC_GEMELO = 0.4


def _posicion_actual(ii: np.ndarray) -> np.ndarray:
    """
    Estimates where a track is NOW from its recent impacts.

    The median of the recent window systematically lags a moving target, since it is the center
    of the recent past. A straight-line fit over the same window, evaluated at the last sample,
    removes that lag while still averaging out projection noise. The observation index stands in
    for time: samples arrive at a near-uniform rate.
    """
    q = max(2, len(ii) // 4)
    v = ii[-q:]
    if len(v) < 3:
        return np.median(v, axis=0)
    idx = np.arange(len(v), dtype=float)
    ajuste = np.polynomial.polynomial.polyfit(idx, v, 1)  # rows: intercept, slope
    return np.asarray(ajuste[0] + ajuste[1] * idx[-1])


class IdentidadIncremental:
    """
    Accumulates tracked detections and produces candidate POIs on demand.

    observar() is O(1) per detection. candidatos() re-associates all track summaries from
    scratch on each call; tracks number in the tens, so running it at every report tick is
    negligible next to the detector. Re-deriving candidates from summaries, instead of patching
    a live clustering, keeps the association rules simple and order-independent.

    Args:
        radio_fusion_m: expected ground-projection noise of the scene, in meters (roughly
            gps_sigma + slant_range * yaw_sigma). Two static tracks closer than this may be the
            same thing. No default: it is a property of the deployment, not of the algorithm.
        fps: the rate at which FRAMES are offered to the detector. A FALLBACK now, and only
            for callers with no clock to offer: pass `t` to observar() and maturity is measured
            in seconds, leaving this number used for nothing but the duty-cycle floor.

            The history is worth keeping, because this parameter has now been wrong three
            ways. It first meant the detection rate, which nobody can know in advance since it
            depends on how intermittent the scene is. On 25ago it was redefined as the FRAME
            rate, on the grounds that the caller sets the vision timer and therefore knows it
            exactly. Measured the same evening, that was false too: the loop rescheduled
            itself as `now + period`, delivering 2.31 frames per second against 3.00
            configured, and every threshold derived from the declared rate stretched by a
            third. Both loop and callers are fixed -- but a number wrong three ways is a
            number to stop depending on. A clock cannot be misconfigured.
        emb_dist_max: appearance distance above which two tracks are never merged.
        dur_pista_s: minimum accumulated observation time for a track to be considered.
        dur_movil_s: minimum accumulated observation time to classify a track as mobile.
        dur_reporte_s: minimum accumulated observation time for a candidate to be reported.
        desplaz_movil_m: net displacement above which a track counts as moving. Defaults to
            slightly above the fusion radius: a static track wanders by projection noise only.
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
        self.desplaz_movil_m = (desplaz_movil_m if desplaz_movil_m is not None
                                else 1.15 * radio_fusion_m)
        # Maturity is measured as a span of frames, not as a count of detections. Both say
        # "enough evidence", but only the span says it in wall-clock terms: a target found in
        # every frame and one found in every third frame become reportable at the same moment,
        # which is what an operator waiting for an alert expects.
        self.dur_pista_s = dur_pista_s
        self.dur_movil_s = dur_movil_s
        self.dur_reporte_s = dur_reporte_s
        self.span_pista = max(3, round(dur_pista_s * fps))
        self.span_movil = max(6, round(dur_movil_s * fps))
        self.span_reporte = max(8, round(dur_reporte_s * fps))

        # A track detected in a tenth of the frames it spans is not being tracked, it is being
        # rediscovered; the span would flatter it. This floor keeps that out.
        self.n_pista = max(3, round(DUTY_MIN * self.span_pista))
        self.n_movil = max(4, round(DUTY_MIN * self.span_movil))
        self.n_reporte = max(5, round(DUTY_MIN * self.span_reporte))

        self._tracks: Dict[int, dict] = {}

    @staticmethod
    def _span(frames) -> int:
        """Frames covered by a track, from first sighting to last."""
        return (max(frames) - min(frames) + 1) if frames else 0

    @staticmethod
    def _alcanza(pista, dur_s: float, span_frames: int) -> bool:
        """
        Has this track covered enough time?

        Measured off the clock whenever the caller supplied one, which is the only version a
        loop running slower than configured cannot distort. The frame span stays as the
        fallback for callers replaying recorded data with no timestamps.
        """
        t0, t1 = pista.get("t0"), pista.get("t1")
        if t0 is not None and t1 is not None:
            return (t1 - t0) >= dur_s
        return IdentidadIncremental._span(pista["frames"]) >= span_frames

    # -- ingest ------------------------------------------------------------

    def observar(
        self,
        frame: int,
        track_id: int,
        impacto_xy: Tuple[float, float],
        conf: float,
        emb: Optional[np.ndarray] = None,
        recorte: Optional[bytes] = None,
        t: Optional[float] = None,
    ) -> None:
        """
        Records one tracked detection, already projected to the ground.

        The crop is optional and only one is kept per track: the one from the most confident
        sighting. A preliminary candidate is a request for verification, and what a verifier
        needs is the clearest look the drone ever got, not the latest -- the latest is often
        the target leaving the frame. Keeping one bounded the message at roughly 3 KB per
        candidate, which is what the whole architecture was sized around.
        """
        sello = t          # `t` below is the track record; keep the timestamp first
        t = self._tracks.get(track_id)
        if t is None:
            t = {"imps": [], "conf_sum": 0.0,
                 "emb_sum": None, "n_emb": 0, "frames": set(),
                 "recorte": None, "recorte_conf": -1.0,
                 "t0": None, "t1": None}
            self._tracks[track_id] = t
        t["imps"].append((float(impacto_xy[0]), float(impacto_xy[1])))
        t["conf_sum"] += float(conf)
        t["frames"].add(int(frame))
        if sello is not None:
            ts = float(sello)
            t["t0"] = ts if t["t0"] is None else min(t["t0"], ts)
            t["t1"] = ts if t["t1"] is None else max(t["t1"], ts)
        if recorte and float(conf) > t["recorte_conf"]:
            t["recorte"], t["recorte_conf"] = recorte, float(conf)
        if emb is not None:
            v = np.asarray(emb, dtype=np.float32)
            t["emb_sum"] = v.copy() if t["emb_sum"] is None else t["emb_sum"] + v
            t["n_emb"] += 1

    # -- association -------------------------------------------------------

    def _resumen_pistas(self) -> List[dict]:
        pistas = []
        for tid, t in self._tracks.items():
            n = len(t["imps"])
            if n < self.n_pista or not self._alcanza(t, self.dur_pista_s, self.span_pista):
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
                "pos": np.median(ii, axis=0),  # robust lifetime center
                "pos_actual": _posicion_actual(ii),
                "desplaz": desplaz,
                "conf": t["conf_sum"] / n,
                "emb": emb,
                "frames": t["frames"],
                "recorte": t.get("recorte"),
                "recorte_conf": t.get("recorte_conf", -1.0),
                "t0": t.get("t0"),
                "t1": t.get("t1"),
            })
        return pistas

    def candidatos(self, preliminares: bool = False) -> List[dict]:
        """
        Returns the current candidate list, mobiles first, then by descending evidence.

        Each candidate: {x, y, n_obs, conf, movil, maduro, recorte}. For a mobile candidate (x, y) is
        its CURRENT position (a mobile's lifetime median points at the middle of its path).
        Static candidates report the lifetime median, which is the point of accumulating views.

        Args:
            preliminares: also return candidates that have formed a track but not yet earned
                a report, marked maduro=False. They exist for the sweep case. Measured on
                flight 3, a pass of 30 s over a person NEVER produces a mature candidate and
                a pass of 60 s produces one 47% of the time: a search that crosses each point
                once and moves on would stay silent over a victim it saw perfectly well.
                Maturity is the right bar for a loitering drone, which can afford to wait and
                should not cry wolf; it is the wrong bar for a sweep, where the only chance to
                say anything is now. A preliminary candidate is not an alert -- it is a
                request for verification, to be sent with the detection crop so the ground
                station decides. Never present one to an operator as a confirmed find.
        """
        cands: List[dict] = []
        for tk in sorted(self._resumen_pistas(), key=lambda p: -p["n"]):
            if (tk["desplaz"] > self.desplaz_movil_m and tk["n"] >= self.n_movil
                    and self._alcanza(tk, self.dur_movil_s, self.span_movil)):
                cands.append({"movil": True, "pos": tk["pos_actual"].copy(),
                              "emb": tk["emb"], "conf": tk["conf"],
                              "n": tk["n"], "frames": set(tk["frames"]),
                              "recorte": tk["recorte"],
                              "recorte_conf": tk["recorte_conf"],
                              "t0": tk["t0"], "t1": tk["t1"]})
                continue
            mejor, smin = None, math.inf
            for k, c in enumerate(cands):
                if c["movil"]:
                    continue
                dp = float(np.linalg.norm(tk["pos"] - c["pos"]))
                if len(tk["frames"] & c["frames"]) >= COOCURRENCIA_MIN:
                    # Seen together: two different things — unless this is the duplicate-box
                    # case (same spot, same appearance).
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
                              "n": tk["n"], "frames": set(tk["frames"]),
                              "recorte": tk["recorte"],
                              "recorte_conf": tk["recorte_conf"],
                              "t0": tk["t0"], "t1": tk["t1"]})
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
                # Positions and appearances average; a photograph cannot. Keep the clearest
                # of the two, which is the one the verifier would have chosen.
                if tk["recorte"] and tk["recorte_conf"] > c["recorte_conf"]:
                    c["recorte"], c["recorte_conf"] = tk["recorte"], tk["recorte_conf"]
                # Two tracks of one target: the evidence spans the union of their intervals.
                if tk["t0"] is not None:
                    c["t0"] = tk["t0"] if c["t0"] is None else min(c["t0"], tk["t0"])
                    c["t1"] = tk["t1"] if c["t1"] is None else max(c["t1"], tk["t1"])

        def maduro(c):
            return (c["n"] >= self.n_reporte
                    and self._alcanza(c, self.dur_reporte_s, self.span_reporte))

        salida = [c for c in cands if maduro(c) or preliminares]
        # Mature first, then mobiles, then by evidence: whatever the caller truncates, it
        # truncates the least certain rows.
        salida.sort(key=lambda c: (not maduro(c), not c["movil"], -c["n"]))
        return [{
            "x": round(float(c["pos"][0]), 2),
            "y": round(float(c["pos"][1]), 2),
            "n_obs": int(c["n"]),
            "conf": round(float(c["conf"]), 3),
            "movil": bool(c["movil"]),
            "maduro": maduro(c),
            # Raw JPEG bytes, or None. Serialising it is the transport's problem, not this
            # layer's; the protocol base64-encodes it on the way out.
            "recorte": c.get("recorte"),
        } for c in salida]
