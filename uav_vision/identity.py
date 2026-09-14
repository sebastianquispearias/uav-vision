"""
Incremental identity: turns tracked detections into named points of interest, online.

Division of labour:
    - The camera owns image-plane tracking (it has the image and the boxes). Every detection
      arrives here already carrying a track_id.
    - This module owns everything after: it accumulates ground impacts per track, classifies
      tracks as static or mobile, and merges tracks that correspond to the same physical thing
      (by position and appearance, with a co-occurrence veto) into reportable candidates.
    - It also carries the detector's class name through to the report. A coordinate with no
      name is not actionable once more than one class is enabled: the ground station cannot
      tell a person from a car, and the two must never be merged into one another.

Image-plane tracking runs first because it is the strong signal: two people two meters apart are
clearly separate in the image, while their ground projections overlap under telemetry noise.
Ground coordinates are used only to place tracks that already exist.

Thresholds are parameters rather than constants because they encode scene properties (ground
noise) and data rate. The provenance of every default is documented in NOTES.md.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Tuple

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

# Radius of the circle holding 95 % of a two-dimensional isotropic Gaussian, in sigmas:
# sqrt(-2 ln 0.05). Used to turn a per-axis position uncertainty into something an operator can
# draw on a map and walk to.
RADIO_95_2D = 2.4477


# Smallest centre-to-centre distance at which two members of a class are still two things,
# in metres. This is parking geometry, not noise: a standard bay is 2.4-2.6 m wide, so two cars
# side by side are 2.5 m apart and a fusion radius of 3.5 m -- the value tuned for people --
# reports them as one car. Nothing downstream can undo that, because by then there is one
# candidate.
#
# PEOPLE ARE DELIBERATELY ABSENT, and for a reason worth stating: two people can also stand
# 0.6 m apart, but nothing places them there the way bays place cars. Applying a 0.6 m floor to
# people would shrink the radius every flight so far was measured with, to buy a separation
# that the scene does not actually impose. Vehicles are the case where the geometry is regular
# enough to encode.
#
# The trade this makes is real and goes one way on purpose. Below the noise radius, one car
# under projection noise can fragment into two candidates a couple of metres apart; above it,
# two cars merge into one. Fragmentation reports the same thing twice in nearly the same place,
# which an operator resolves at a glance. Conflation makes a vehicle disappear, and nothing on
# the screen says so.
SEPARACION_MINIMA_M: Dict[str, float] = {
    "bicycle": 0.8,
    "motor": 1.0,
    "tricycle": 1.2,
    "awning-tricycle": 1.2,
    "car": 2.5,
    "van": 2.7,
    "truck": 3.2,
    "bus": 3.4,
}


def radii_by_class(noise_radius_m: float,
                   separations: Optional[Mapping[str, float]] = None
                   ) -> Dict[str, float]:
    """
    The per-class fusion radii for a scene whose projection noise is noise_radius_m.

    Each class gets the smaller of the two constraints on it: how far noise can move one
    object, and how close two of them are ever placed. Ready to hand to
    IncrementalIdentity(fusion_radius_by_class=...); classes not in the table keep the scene
    radius, which is what people do.
    """
    sep = SEPARACION_MINIMA_M if separations is None else separations
    return {c: min(noise_radius_m, d) for c, d in sep.items()}


def dominant_class(votos: Optional[Mapping[str, int]]) -> Optional[str]:
    """
    The class an object is reported as: the one most of its detections carried.

    A vote rather than the latest label, because a detector flips class on the odd frame and
    one bad frame must not rename a track that fifty good ones agreed on. Ties break
    alphabetically, so the answer never depends on which frame happened to arrive first.

    Returns None when there are no votes at all, which is what a camera that does not report
    a class produces. None means "unknown", never "different from yours".
    """
    if not votos:
        return None
    return max(sorted(votos), key=lambda nombre: votos[nombre])


def _current_position(ii: np.ndarray) -> np.ndarray:
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


class IncrementalIdentity:
    """
    Accumulates tracked detections and produces candidate POIs on demand.

    observe() is O(1) per detection. candidates() re-associates all track summaries from
    scratch on each call; tracks number in the tens, so running it at every report tick is
    negligible next to the detector. Re-deriving candidates from summaries, instead of patching
    a live clustering, keeps the association rules simple and order-independent.

    Args:
        fusion_radius_m: expected ground-projection noise of the scene, in meters (roughly
            gps_sigma + slant_range * yaw_sigma). Two static tracks closer than this may be the
            same thing. No default: it is a property of the deployment, not of the algorithm.
        fps: the rate at which FRAMES are offered to the detector. A FALLBACK now, and only
            for callers with no clock to offer: pass `t` to observe() and maturity is measured
            in seconds, leaving this number used for nothing but the duty-cycle floor.

            The history is worth keeping, because this parameter has now been wrong three
            ways. It first meant the detection rate, which nobody can know in advance since it
            depends on how intermittent the scene is. On 2026-08-25 it was redefined as the FRAME
            rate, on the grounds that the caller sets the vision timer and therefore knows it
            exactly. Measured the same evening, that was false too: the loop rescheduled
            itself as `now + period`, delivering 2.31 frames per second against 3.00
            configured, and every threshold derived from the declared rate stretched by a
            third. Both loop and callers are fixed -- but a number wrong three ways is a
            number to stop depending on. A clock cannot be misconfigured.
        emb_dist_max: appearance distance above which two tracks are never merged.
        track_dur_s: minimum accumulated observation time for a track to be considered.
        mobile_dur_s: minimum accumulated observation time to classify a track as mobile.
        report_dur_s: minimum accumulated observation time for a candidate to be reported.
        mobile_disp_m: net displacement above which a track counts as moving. Defaults to
            slightly above the fusion radius: a static track wanders by projection noise only.
        fusion_radius_by_class: per-class override of fusion_radius_m, for classes whose
            members can legitimately stand closer together than the scene's noise. The scene
            radius answers "how far can noise move one thing?"; the radius also has to answer
            "how close can two of these ever be?", and for vehicles the second is the binding
            one. Two cars in adjacent parking bays are about 2.5 m apart centre to centre, so
            the 3.5 m radius tuned for people reports them as one car -- and nothing
            downstream can undo that, because by then there is one candidate. Set a class to
            min(noise radius, smallest plausible separation for that class).
        reinforce_with_fragments: let a track too short to stand on its own join a candidate
            that already exists, by the same rules that merge two full tracks. It never creates
            a candidate: track_dur_s is there so that a few seconds of noise cannot put a point
            on the map, and that stays true. What it removes is the other effect of the same
            gate -- a target the tracker keeps losing and re-finding under new ids contributes
            only its longest pieces to the candidate that is plainly the same thing. Off by
            default: the reported chain is the validated one.
        maturity: what "enough evidence" means. "span" (the default, the validated chain) is the
            time between the first and the last sighting: track_dur_s to open a track,
            report_dur_s to report. It measures the wrong thing. On flight 3 a static object
            the detector keeps confusing with a person is sighted 86 times over 758 s and
            matures, while a person seen continuously for 30 s on a sweep never does; and the
            operator, located within 3 m of the truth after 2 s, is reported only at 76 s.
            "looks" counts independent looks instead: the distinct look_s-long intervals in
            which the thing was detected at all. Consecutive frames of one second are one look,
            because they share the same pose error and the same background; a second sighting a
            minute later is another. A track opens with track_min_looks, a candidate is reported
            with report_min_looks, and every candidate carries its looks and radius_m.
            Neither mode separates a real target from a persistent false detection: measured,
            the chain's own signals do not (detection rate in view, confidence, apparent size,
            and an appearance classifier that does not transfer between flights). That
            decision belongs to whoever looks at the crop.
        track_min_looks, report_min_looks: the "looks" thresholds. They encode how costly a
            false report is against a late one, which changes per mission; they are decisions,
            not measurements.
        look_s: the length of one look, in seconds.
        bias_sigma_m: per-axis standard deviation of the error that more looks cannot average
            away -- GPS and heading bias, shared by every sighting of a flight. The default
            comes from the chain's measured median error on real flights, 2.4 m: for a
            two-dimensional Gaussian the median radial error is 1.1774 sigma.
    """

    def __init__(
        self,
        fusion_radius_m: float,
        fps: float,
        emb_dist_max: float = EMB_DIST_MAX_MEDIDO,
        track_dur_s: float = 8.6,
        mobile_dur_s: float = 29.0,
        report_dur_s: float = 36.0,
        mobile_disp_m: Optional[float] = None,
        fusion_radius_by_class: Optional[Mapping[str, float]] = None,
        reinforce_with_fragments: bool = False,
        maturity: str = "span",
        track_min_looks: int = 3,
        report_min_looks: int = 5,
        look_s: float = 1.0,
        bias_sigma_m: float = 2.4 / 1.1774,
    ) -> None:
        if maturity not in ("span", "looks"):
            raise ValueError("maturity must be 'span' or 'looks', got %r" % (maturity,))
        self.fusion_radius_m = fusion_radius_m
        self.reinforce_with_fragments = reinforce_with_fragments
        self.maturity = maturity
        self.track_min_looks = track_min_looks
        self.report_min_looks = report_min_looks
        self.look_s = look_s
        self.bias_sigma_m = bias_sigma_m
        self._fps = fps
        self.fusion_radius_by_class = dict(fusion_radius_by_class or {})
        self.emb_dist_max = emb_dist_max
        # Kept as given, so "derive it from the radius" stays a per-class answer while an
        # explicit value stays one number for the whole scene, as the caller asked.
        self._disp_dado = mobile_disp_m
        self.mobile_disp_m = (mobile_disp_m if mobile_disp_m is not None
                                else 1.15 * fusion_radius_m)
        # Maturity is measured as a span of frames, not as a count of detections. Both say
        # "enough evidence", but only the span says it in wall-clock terms: a target found in
        # every frame and one found in every third frame become reportable at the same moment,
        # which is what an operator waiting for an alert expects.
        self.track_dur_s = track_dur_s
        self.mobile_dur_s = mobile_dur_s
        self.report_dur_s = report_dur_s
        self.span_pista = max(3, round(track_dur_s * fps))
        self.span_movil = max(6, round(mobile_dur_s * fps))
        self.span_reporte = max(8, round(report_dur_s * fps))

        # A track detected in a tenth of the frames it spans is not being tracked, it is being
        # rediscovered; the span would flatter it. This floor keeps that out.
        self.n_pista = max(3, round(DUTY_MIN * self.span_pista))
        self.n_movil = max(4, round(DUTY_MIN * self.span_movil))
        self.n_reporte = max(5, round(DUTY_MIN * self.span_reporte))

        self._tracks: Dict[int, dict] = {}

    def _radio(self, cls: Optional[str]) -> float:
        """The fusion radius that applies to a class: its own if it declared one, else the
        scene's. An unknown class gets the scene radius, which is today's behaviour."""
        return self.fusion_radius_by_class.get(cls, self.fusion_radius_m)

    def _disp_movil(self, cls: Optional[str]) -> float:
        """Displacement above which a track of this class counts as moving."""
        if self._disp_dado is not None:
            return self._disp_dado
        return 1.15 * self._radio(cls)

    @staticmethod
    def _span(frames) -> int:
        """Frames covered by a track, from first sighting to last."""
        return (max(frames) - min(frames) + 1) if frames else 0

    @staticmethod
    def _has_covered(track, dur_s: float, span_frames: int) -> bool:
        """
        Has this track covered enough time?

        Measured off the clock whenever the caller supplied one, which is the only version a
        loop running slower than configured cannot distort. The frame span stays as the
        fallback for callers replaying recorded data with no timestamps.
        """
        t0, t1 = track.get("t0"), track.get("t1")
        if t0 is not None and t1 is not None:
            return (t1 - t0) >= dur_s
        return IncrementalIdentity._span(track["frames"]) >= span_frames

    # -- ingest ------------------------------------------------------------

    def observe(
        self,
        frame: int,
        track_id: int,
        ground_xy: Tuple[float, float],
        conf: float,
        emb: Optional[np.ndarray] = None,
        crop: Optional[bytes] = None,
        t: Optional[float] = None,
        cls: Optional[str] = None,
    ) -> None:
        """
        Records one tracked detection, already projected to the ground.

        The crop is optional and only one is kept per track: the one from the most confident
        sighting. A preliminary candidate is a request for verification, and what a verifier
        needs is the clearest look the drone ever got, not the latest -- the latest is often
        the target leaving the frame. Keeping one bounded the message at roughly 3 KB per
        candidate, which is what the whole architecture was sized around.

        cls is what the detector called this thing, and it is accumulated as a vote per class
        rather than stored as the latest label -- see dominant_class. Leaving it out costs
        nothing: a track with no votes reports no class and associates exactly as before.
        """
        sello = t          # `t` below is the track record; keep the timestamp first
        t = self._tracks.get(track_id)
        if t is None:
            t = {"imps": [], "conf_sum": 0.0,
                 "emb_sum": None, "n_emb": 0, "frames": set(),
                 "crop": None, "recorte_conf": -1.0,
                 "t0": None, "t1": None, "cls_votos": {},
                 "bins": set(), "imp_bins": []}
            self._tracks[track_id] = t
        t["imps"].append((float(ground_xy[0]), float(ground_xy[1])))
        t["conf_sum"] += float(conf)
        t["frames"].add(int(frame))
        # The look this sighting belongs to: off the clock when there is one, else off the frame
        # index at the declared rate, the same fallback the span mode uses.
        if sello is not None:
            mirada = int(math.floor(float(sello) / self.look_s))
        else:
            mirada = int(frame) // max(1, int(round(self._fps * self.look_s)))
        t["bins"].add(mirada)
        t["imp_bins"].append(mirada)
        if sello is not None:
            ts = float(sello)
            t["t0"] = ts if t["t0"] is None else min(t["t0"], ts)
            t["t1"] = ts if t["t1"] is None else max(t["t1"], ts)
        if cls is not None:
            t["cls_votos"][cls] = t["cls_votos"].get(cls, 0) + 1
        if crop and float(conf) > t["recorte_conf"]:
            t["crop"], t["recorte_conf"] = crop, float(conf)
        if emb is not None:
            v = np.asarray(emb, dtype=np.float32)
            t["emb_sum"] = v.copy() if t["emb_sum"] is None else t["emb_sum"] + v
            t["n_emb"] += 1

    # -- association -------------------------------------------------------

    def _summary(self, tid: int, t: dict) -> dict:
        """One track reduced to what association needs: where, how much, what it looks like."""
        n = len(t["imps"])
        ii = np.asarray(t["imps"])
        q = max(1, n // 4)
        desplaz = float(np.linalg.norm(
            np.median(ii[:q], axis=0) - np.median(ii[-q:], axis=0)))
        emb = None
        if t["n_emb"] > 0:
            emb = t["emb_sum"] / (np.linalg.norm(t["emb_sum"]) + 1e-9)
        return {
            "tid": tid, "n": n,
            "pos": np.median(ii, axis=0),  # robust lifetime center
            "pos_actual": _current_position(ii),
            "desplaz": desplaz,
            "conf": t["conf_sum"] / n,
            "emb": emb,
            "frames": t["frames"],
            "crop": t.get("crop"),
            "recorte_conf": t.get("recorte_conf", -1.0),
            "t0": t.get("t0"),
            "t1": t.get("t1"),
            "cls_votos": dict(t.get("cls_votos") or {}),
            "cls": dominant_class(t.get("cls_votos")),
            "bins": t["bins"],
        }

    def _track_summaries(self) -> List[dict]:
        """Tracks that earned a place in association: enough detections over enough time."""
        return [self._summary(tid, t) for tid, t in self._tracks.items() if self._track_ok(t)]

    def _fragment_summaries(self) -> List[dict]:
        """
        Tracks with enough detections that did not last track_dur_s.

        A tracker that loses its target and finds it again under a new id leaves exactly these
        behind. Measured on flight 3, over the hand-labelled window: of the operator's boxes that
        the flight's BoT-SORT gave an id, 62 % sit in tracks shorter than 8.6 s, and 96 % once
        the tracker is calibrated to the aerial detector's confidences.
        """
        return [self._summary(tid, t) for tid, t in self._tracks.items()
                if len(t["imps"]) >= self.n_pista and not self._track_ok(t)]

    def _track_ok(self, t: dict) -> bool:
        """Has this track earned a place in association, under the configured maturity mode?"""
        if self.maturity == "looks":
            return len(t["bins"]) >= self.track_min_looks
        return (len(t["imps"]) >= self.n_pista
                and self._has_covered(t, self.track_dur_s, self.span_pista))

    def _radius(self, c: dict) -> float:
        """
        Radius, in metres, of the circle that should hold the target 95 % of the time.

        Two errors add in quadrature. The bias (bias_sigma_m) is shared by every sighting and does
        not shrink. The random part is the spread between looks -- each look reduced to its mean
        impact, since sightings inside one look are not independent -- divided by the square root
        of the number of looks. With few looks the random part dominates; with many, the radius
        settles on the bias, which is the honest floor of a single-camera system.
        """
        por_mirada: Dict[int, list] = {}
        for tid in c["tids"]:
            t = self._tracks[tid]
            for imp, mirada in zip(t["imps"], t["imp_bins"]):
                por_mirada.setdefault(mirada, []).append(imp)
        medias = np.array([np.mean(v, axis=0) for v in por_mirada.values()])
        n = len(medias)
        var = float(medias.var(axis=0, ddof=1).mean()) if n >= 2 else 0.0
        return RADIO_95_2D * math.sqrt(self.bias_sigma_m ** 2 + var / max(1, n))

    def _match(self, tk: dict, cands: List[dict]) -> Optional[int]:
        """
        Index of the static candidate this track belongs to, or None.

        The rules apply in order -- class, co-occurrence, distance, appearance -- and among the
        candidates that pass all of them the one closest in position and appearance wins.
        """
        radio = self._radio(tk["cls"])
        mejor, smin = None, math.inf
        for k, c in enumerate(cands):
            if c["mobile"]:
                continue
            # Two names, two things. This veto comes before every other rule, the twin
            # exception included: a car is not the person standing beside it however
            # close they are and however alike their crops look at 35 m. Silence on
            # either side is not disagreement -- a track with no votes still merges the
            # way it always did, which is what keeps every camera without a class
            # working unchanged.
            c_cls = dominant_class(c["cls_votos"])
            if (tk["cls"] is not None and c_cls is not None
                    and tk["cls"] != c_cls):
                continue
            dp = float(np.linalg.norm(tk["pos"] - c["pos"]))
            if len(tk["frames"] & c["frames"]) >= COOCURRENCIA_MIN:
                # Seen together: two different things — unless this is the duplicate-box
                # case (same spot, same appearance).
                es_gemelo = (
                    dp < POS_FRAC_GEMELO * radio
                    and tk["emb"] is not None and c["emb"] is not None
                    and float(np.linalg.norm(tk["emb"] - c["emb"]))
                    < EMB_DIST_GEMELO)
                if not es_gemelo:
                    continue
            if dp >= radio:
                continue
            if tk["emb"] is not None and c["emb"] is not None:
                de = float(np.linalg.norm(tk["emb"] - c["emb"]))
                if de >= self.emb_dist_max:
                    continue
                s = dp / radio + 0.5 * de / self.emb_dist_max
            else:
                s = dp / radio
            if s < smin:
                mejor, smin = k, s
        return mejor

    @staticmethod
    def _absorb(c: dict, tk: dict) -> None:
        """Folds a track into a candidate, weighting position and appearance by evidence."""
        w = c["n"] / (c["n"] + tk["n"])
        c["pos"] = w * c["pos"] + (1 - w) * tk["pos"]
        if c["emb"] is not None and tk["emb"] is not None:
            e = w * c["emb"] + (1 - w) * tk["emb"]
            c["emb"] = e / (np.linalg.norm(e) + 1e-9)
        c["conf"] = w * c["conf"] + (1 - w) * tk["conf"]
        c["n"] += tk["n"]
        c["frames"] |= tk["frames"]
        c["bins"] |= tk["bins"]
        c["tids"].append(tk["tid"])
        for nombre, v in tk["cls_votos"].items():
            c["cls_votos"][nombre] = c["cls_votos"].get(nombre, 0) + v
        # Positions and appearances average; a photograph cannot. Keep the clearest
        # of the two, which is the one the verifier would have chosen.
        if tk["crop"] and tk["recorte_conf"] > c["recorte_conf"]:
            c["crop"], c["recorte_conf"] = tk["crop"], tk["recorte_conf"]
        # Two tracks of one target: the evidence spans the union of their intervals.
        if tk["t0"] is not None:
            c["t0"] = tk["t0"] if c["t0"] is None else min(c["t0"], tk["t0"])
            c["t1"] = tk["t1"] if c["t1"] is None else max(c["t1"], tk["t1"])

    def candidates(self, preliminary: bool = False, with_tracks: bool = False) -> List[dict]:
        """
        Returns the current candidate list, mobiles first, then by descending evidence.

        Each candidate: {x, y, cls, n_obs, conf, mobile, mature, crop}. For a mobile candidate (x, y) is
        its CURRENT position (a mobile's lifetime median points at the middle of its path).
        Static candidates report the lifetime median, which is the point of accumulating views.

        Args:
            with_tracks: also return, under "tracks", the ids of the tracks merged into each
                candidate. Off by default because the candidates are what the drone reports
                over the radio, and the list is only needed to score identity against ground
                truth, on the ground.
            preliminary: also return candidates that have formed a track but not yet earned
                a report, marked mature=False. They exist for the sweep case. Measured on
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
        for tk in sorted(self._track_summaries(), key=lambda p: -p["n"]):
            if (tk["desplaz"] > self._disp_movil(tk["cls"]) and tk["n"] >= self.n_movil
                    and self._has_covered(tk, self.mobile_dur_s, self.span_movil)):
                cands.append({"mobile": True, "pos": tk["pos_actual"].copy(),
                              "emb": tk["emb"], "conf": tk["conf"],
                              "n": tk["n"], "frames": set(tk["frames"]),
                              "crop": tk["crop"],
                              "recorte_conf": tk["recorte_conf"],
                              "t0": tk["t0"], "t1": tk["t1"],
                              "cls_votos": dict(tk["cls_votos"]), "tids": [tk["tid"]],
                              "bins": set(tk["bins"])})
                continue
            mejor = self._match(tk, cands)
            if mejor is None:
                cands.append({"mobile": False, "pos": tk["pos"].copy(),
                              "emb": tk["emb"], "conf": tk["conf"],
                              "n": tk["n"], "frames": set(tk["frames"]),
                              "crop": tk["crop"],
                              "recorte_conf": tk["recorte_conf"],
                              "t0": tk["t0"], "t1": tk["t1"],
                              "cls_votos": dict(tk["cls_votos"]), "tids": [tk["tid"]],
                              "bins": set(tk["bins"])})
            else:
                self._absorb(cands[mejor], tk)

        # Fragments come last and only reinforce: every candidate above was opened by a track
        # that lasted, so no amount of short noise can add a point to the map.
        if self.reinforce_with_fragments:
            for tk in sorted(self._fragment_summaries(), key=lambda p: -p["n"]):
                mejor = self._match(tk, cands)
                if mejor is not None:
                    self._absorb(cands[mejor], tk)

        def mature(c):
            if self.maturity == "looks":
                return len(c["bins"]) >= self.report_min_looks
            return (c["n"] >= self.n_reporte
                    and self._has_covered(c, self.report_dur_s, self.span_reporte))

        out = [c for c in cands if mature(c) or preliminary]
        # Mature first, then mobiles, then by evidence: whatever the caller truncates, it
        # truncates the least certain rows.
        out.sort(key=lambda c: (not mature(c), not c["mobile"], -c["n"]))
        return [{
            "x": round(float(c["pos"][0]), 2),
            "y": round(float(c["pos"][1]), 2),
            # What it is, next to where it is. None when no camera ever said.
            "cls": dominant_class(c["cls_votos"]),
            "n_obs": int(c["n"]),
            "conf": round(float(c["conf"]), 3),
            "mobile": bool(c["mobile"]),
            "mature": mature(c),
            # Raw JPEG bytes, or None. Serialising it is the transport's problem, not this
            # layer's; the protocol base64-encodes it on the way out.
            "crop": c.get("crop"),
            # The appearance vector, same convention as the crop: raw here, encoded by the
            # transport. It leaves the drone because deciding that two drones are looking at
            # one target is a comparison neither of them can make alone, and position is not
            # enough: two people three metres apart are two people.
            "emb": c.get("emb"),
            # How much independent evidence, and how far off the point may be. Only in "looks"
            # mode, so the validated chain's report is unchanged byte for byte.
            **({"looks": len(c["bins"]), "radius_m": round(self._radius(c), 2)}
               if self.maturity == "looks" else {}),
            **({"tracks": sorted(c["tids"])} if with_tracks else {}),
        } for c in out]
