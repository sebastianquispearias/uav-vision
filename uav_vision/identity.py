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

CONSTANTS THAT ARE DECISIONS
    EMB_DIST_MAX_MEDIDO
        Fusion threshold for appearance embeddings, as an L2 distance between unit vectors.

    COOCURRENCIA_MIN
        Two tracks seen together in this many frames are two different physical things,
        whatever position and appearance say: nothing appears twice in the same photo.

    DUTY_MIN
        Minimum fraction of the frames a track spans in which it must actually have been
        detected. Below it the track is a handful of sightings spread thin, and its frame span
        overstates the evidence behind it.

    EMB_DIST_GEMELO, POS_FRAC_GEMELO
        The exception to the co-occurrence veto, for the duplicate boxes a detector emits for
        one person. The veto is lifted only when the evidence says "same person": nearly
        identical position AND an embedding distance deep inside the same-identity zone.

    EMB_DIST_REUNE
        What appearance has to say before a moving candidate is re-associated, which is the one
        case where position argues for splitting and appearance for joining. THIS NUMBER IS NOT
        SAFE AND THAT IS WHY REJOINING IS OFF BY DEFAULT: the same-person and different-person
        ranges overlap, so no threshold separates them, and the value was chosen by looking at
        the one flight it is judged on.

    RADIO_95_2D
        Radius of the circle holding 95 % of a two-dimensional isotropic Gaussian, in sigmas:
        sqrt(-2 ln 0.05). Derived, not tuned. It turns a per-axis position uncertainty into
        something an operator can draw on a map and walk to.

    RANGO_REFERENCIA_M
        The slant range at which the chain's median error was measured. A position margin
        quoted without the range it holds at is incomplete: a heading error moves the impact
        by range times angle.

    GPS_SIGMA_M
        Horizontal GPS standard deviation assumed for a consumer receiver, per axis. Not
        measured here: with one calibration point and two error sources, one of them has to be
        assumed, and this is the better known of the two.

    SEPARACION_MINIMA_M
        Smallest centre-to-centre distance at which two members of a class are still two
        things, in metres. This is parking geometry, not noise, and people are deliberately
        absent from it.

    Every value above, what it trades off and the measurement behind it: NOTES.md.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

EMB_DIST_MAX_MEDIDO = 0.95

COOCURRENCIA_MIN = 3

DUTY_MIN = 0.10

EMB_DIST_GEMELO = 0.70
POS_FRAC_GEMELO = 0.4

EMB_DIST_REUNE = 0.63

RADIO_95_2D = 2.4477

RANGO_REFERENCIA_M = 20.3

GPS_SIGMA_M = 1.5


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


def _current_position(ii: np.ndarray, ts: Optional[List[Optional[float]]] = None) -> np.ndarray:
    """
    Estimates where a track is NOW from its recent impacts.

    The median of the recent window systematically lags a moving target, since it is the center
    of the recent past. A straight-line fit over the same window, evaluated at the last sample,
    removes that lag while still averaging out projection noise.

    The fit is against the time of each sighting whenever the caller supplied one. It used to be
    against the observation index, on the assumption that sightings arrive at a near-uniform rate,
    and they do not: a target in frame is detected in 4.6 % to 38 % of the frames on flight 3, in
    bursts. Against the index, a burst of sightings a second apart and two sightings twenty seconds
    apart count as the same step, and the line through a moving target bends accordingly. The
    index remains the fallback for callers with no clock.
    """
    q = max(2, len(ii) // 4)
    v = ii[-q:]
    if len(v) < 3:
        return np.median(v, axis=0)
    if ts is not None and len(ts) == len(ii) and all(t is not None for t in ts[-q:]):
        x = np.asarray(ts[-q:], dtype=float)
        x = x - x[-1]
        if np.ptp(x) > 0:
            ajuste = np.polynomial.polynomial.polyfit(x, v, 1)
            return np.asarray(ajuste[0])
    idx = np.arange(len(v), dtype=float)
    ajuste = np.polynomial.polynomial.polyfit(idx, v, 1)
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
            in seconds, leaving this number used for nothing but the duty-cycle floor. It has
            been wrong three ways and NOTES.md keeps the history; a clock cannot be
            misconfigured, which is why the clock is the primary path.
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
            one. Set a class to min(noise radius, smallest plausible separation for that
            class); SEPARACION_MINIMA_M holds those separations and NOTES.md the geometry
            behind them.
        reinforce_with_fragments: let a track too short to stand on its own join a candidate
            that already exists, by the same rules that merge two full tracks. It never creates
            a candidate: track_dur_s is there so that a few seconds of noise cannot put a point
            on the map, and that stays true. What it removes is the other effect of the same
            gate: a target the tracker keeps losing and re-finding under new ids contributes
            only its longest pieces to the candidate that is plainly the same thing. Off by
            default, because the reported chain is the validated one.
        maturity: what "enough evidence" means. "looks", the default, counts the distinct
            look_s-long intervals in which the thing was detected at all: consecutive frames of
            one second are one look, because they share the same pose error and the same
            background, while a sighting a minute later is another. "span" is the time between
            the first and the last sighting, which is the rule the chain used until it was
            measured against a flight; it is kept so those numbers stay reproducible, and
            NOTES.md shows what it gets wrong.

            NEITHER MODE separates a real target from a persistent false detection. Measured,
            the chain's own signals do not either. That decision belongs to whoever looks at
            the crop.
        track_min_looks, report_min_looks: the "looks" thresholds, to open a track and to
            report a candidate. Decisions and not measurements: they encode how costly a false
            report is against a late one, which changes per mission.
        look_s: the length of one look, in seconds.
        bias_sigma_m: per-axis standard deviation of the error that more looks cannot average
            away -- GPS and heading bias, shared by every sighting of a flight. It holds at the
            range it was measured at, RANGO_REFERENCIA_M; when observations carry range_m the
            shared error is modelled as sqrt(gps_sigma_m^2 + (range * yaw_sigma)^2), with the
            heading error solved so the model gives bias_sigma_m back at that range.
        gps_sigma_m: the GPS part of that error, per axis. Assumed, see GPS_SIGMA_M.
        motion_window_s: how far back, in seconds, "where is it now and how fast is it going"
            looks, in "looks" mode. A target that moves has to be described by its recent past,
            because a line fitted over its whole life crosses its turns.
        mobile_speed_mps: speed above which a target counts as moving, in "looks" mode,
            provided it also exceeds three times the standard error of its own estimate and the
            fitted motion carries the target further than the mobile displacement. Both extra
            tests are there because projection noise on a standing person fakes a speed, and
            speed alone turned one standing person into several points on the map. A decision,
            not a measurement: a walking person is about 1.4 m/s.
        extrapolation_max_s: how far ahead, in seconds, a moving candidate is carried from its
            last sighting to the report time. Separate from the window its velocity is
            estimated over: a target that turns keeps going on paper for as long as this
            allows. No value wins at every speed, so this one is a decision about which failure
            to tolerate rather than an optimum.
        rejoin_mobile: let a moving candidate be re-associated with a track it was already
            matched to, deciding by appearance where position argues for splitting. OFF BY
            DEFAULT because emb_dist_rejoin is not a safe number; see EMB_DIST_REUNE.
        emb_dist_rejoin: the appearance distance that rejoining demands, stricter than the one
            that decides a static merge.
        rejoin_max_gap_s: how long a moving candidate may be unseen and still be rejoined.
        crop_choice: which crop a candidate carries. "confidence", the default, keeps the most
            confident sighting of each track and the most confident of its tracks. "appearance"
            keeps, per track, the sighting whose vector is closest to the track's mean
            appearance, and per candidate the track crop closest to the candidate's, so the
            photograph shows what the evidence mostly is. Opt-in, because it trades the
            confidence of the picture for its representativeness and the measurement shows
            both sides.

    Every default that is a decision rather than a derivation, and the measurement behind it:
    NOTES.md.
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
        maturity: str = "looks",
        track_min_looks: int = 3,
        report_min_looks: int = 20,
        look_s: float = 1.0,
        bias_sigma_m: float = 2.4 / 1.1774,
        gps_sigma_m: float = GPS_SIGMA_M,
        motion_window_s: float = 5.0,
        mobile_speed_mps: float = 0.5,
        extrapolation_max_s: float = 3.0,
        rejoin_mobile: bool = False,
        emb_dist_rejoin: float = EMB_DIST_REUNE,
        rejoin_max_gap_s: float = 30.0,
        crop_choice: str = "confidence",
    ) -> None:
        """Builds the thresholds the association rules work with.

        Three of the derived quantities are worth naming, because they are not restatements of
        their arguments:

        yaw_sigma_rad
            The heading error that, together with gps_sigma_m, reproduces bias_sigma_m at the
            reference range.

        span_pista, span_movil, span_reporte
            Maturity in the "span" mode, measured as a span of FRAMES and not as a count of
            detections. Both say "enough evidence", but only the span says it in wall-clock
            terms: a target found in every frame and one found in every third frame become
            reportable at the same moment, which is what an operator waiting for an alert
            expects.

        n_pista, n_movil, n_reporte
            The duty-cycle floor under those spans. A track detected in a tenth of the frames
            it spans is not being tracked, it is being rediscovered, and the span would flatter
            it.

        mobile_disp_m is kept as given as well as derived, so that "derive it from the radius"
        stays a per-class answer while an explicit value stays one number for the whole scene,
        as the caller asked.

        _frames_vistos is every frame index this layer was ever handed a detection in, and it
        is the denominator of opportunity: asking in what fraction of a candidate's life it was
        seen is only meaningful against the frames something was seen in at all. Seconds will
        not do, because the camera delivers in bursts. Note what it is NOT: a frame the
        detector found nothing in never reaches observe, so the denominator is frames in which
        detection was producing something, not frames the camera captured. That is the stricter
        of the two readings, and the one available without a second channel from the camera.
        """
        if maturity not in ("span", "looks"):
            raise ValueError("maturity must be 'span' or 'looks', got %r" % (maturity,))
        if crop_choice not in ("confidence", "appearance"):
            raise ValueError("crop_choice must be 'confidence' or 'appearance', got %r" % (crop_choice,))
        self.crop_choice = crop_choice
        self.rejoin_mobile = bool(rejoin_mobile)
        self.emb_dist_rejoin = float(emb_dist_rejoin)
        self.rejoin_max_gap_s = float(rejoin_max_gap_s)
        self.fusion_radius_m = fusion_radius_m
        self.reinforce_with_fragments = reinforce_with_fragments
        self.maturity = maturity
        self.track_min_looks = track_min_looks
        self.report_min_looks = report_min_looks
        self.look_s = look_s
        self.bias_sigma_m = bias_sigma_m
        self.gps_sigma_m = gps_sigma_m
        self.motion_window_s = motion_window_s
        self.mobile_speed_mps = mobile_speed_mps
        self.extrapolation_max_s = extrapolation_max_s
        self.yaw_sigma_rad = math.sqrt(max(0.0, bias_sigma_m ** 2 - gps_sigma_m ** 2)) / RANGO_REFERENCIA_M
        self._fps = fps
        self.fusion_radius_by_class = dict(fusion_radius_by_class or {})
        self.emb_dist_max = emb_dist_max
        self._disp_dado = mobile_disp_m
        self.mobile_disp_m = (mobile_disp_m if mobile_disp_m is not None
                                else 1.15 * fusion_radius_m)
        self.track_dur_s = track_dur_s
        self.mobile_dur_s = mobile_dur_s
        self.report_dur_s = report_dur_s
        self.span_pista = max(3, round(track_dur_s * fps))
        self.span_movil = max(6, round(mobile_dur_s * fps))
        self.span_reporte = max(8, round(report_dur_s * fps))

        self.n_pista = max(3, round(DUTY_MIN * self.span_pista))
        self.n_movil = max(4, round(DUTY_MIN * self.span_movil))
        self.n_reporte = max(5, round(DUTY_MIN * self.span_reporte))

        self._tracks: Dict[int, dict] = {}
        self._frames_vistos: set = set()

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


    def olvidar_todo(self) -> None:
        """Drops every track and every candidate, as if the layer had just been built.

        This is the only operation in this layer that loses information, and it exists for one
        reason: nothing else could. Everything here grows monotonically, on purpose -- a
        candidate is never forgotten, so a target lost a minute ago is still reported -- which
        means a station that wants a clean slate has nowhere to ask for one. Hiding the pins on
        the operator's screen only hides them: the next report brings them all back, because the
        originals live here and not there.

        What it deliberately does NOT touch is the operator's refusals, which the protocol keeps
        separately. A refusal is a judgement about the world, not a drawing on a screen: clearing
        the board should not make the drone start reporting again the point a person already
        said was not a person.
        """
        self._tracks.clear()
        self._frames_vistos.clear()

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
        range_m: Optional[float] = None,
    ) -> None:
        """
        Records one tracked detection, already projected to the ground.

        The crop is optional and only one is kept per track: under crop_choice="confidence" the
        one from the most confident sighting, under "appearance" the one that looks most like the
        rest of the track. A preliminary candidate is a request for verification, and what a
        verifier needs is a look that shows the target, not the latest -- the latest is often
        the target leaving the frame. Keeping one bounded the message at roughly 3 KB per
        candidate, which is what the whole architecture was sized around.

        cls is what the detector called this thing, and it is accumulated as a vote per class
        rather than stored as the latest label -- see dominant_class. Leaving it out costs
        nothing: a track with no votes reports no class and associates exactly as before.

        The sighting is also filed under the look it belongs to, taken off the clock when there
        is one and off the frame index at the declared rate when there is not, which is the same
        fallback the span mode uses. The timestamp is read out before `t` is rebound to the
        track record it names for the rest of the method.
        """
        sello = t
        t = self._tracks.get(track_id)
        if t is None:
            t = {"imps": [], "conf_sum": 0.0,
                 "emb_sum": None, "n_emb": 0, "frames": set(),
                 "crop": None, "recorte_conf": -1.0,
                 "t0": None, "t1": None, "cls_votos": {},
                 "bins": set(), "imp_bins": [], "ts": [], "rangos": []}
            self._tracks[track_id] = t
        t["imps"].append((float(ground_xy[0]), float(ground_xy[1])))
        t["ts"].append(float(sello) if sello is not None else None)
        if range_m is not None:
            t["rangos"].append(float(range_m))
        t["conf_sum"] += float(conf)
        t["frames"].add(int(frame))
        self._frames_vistos.add(int(frame))
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
        if emb is not None:
            v = np.asarray(emb, dtype=np.float32)
            t["emb_sum"] = v.copy() if t["emb_sum"] is None else t["emb_sum"] + v
            t["n_emb"] += 1
        if crop:
            if self.crop_choice == "appearance":
                self._keep_representative_crop(t, crop, float(conf), emb)
            elif float(conf) > t["recorte_conf"]:
                t["crop"], t["recorte_conf"] = crop, float(conf)

    @staticmethod
    def _keep_representative_crop(t: dict, crop: bytes, conf: float, emb) -> None:
        """
        Keeps, of the kept crop and a new one, the one whose appearance is closer to the track's.

        The track's appearance is the running mean of every vector it has received, this sighting
        included, so the comparison is always against the current mean and a crop kept early on is
        displaced as soon as the mean moves away from it. Memory stays one crop and one vector per
        track. A crop with a vector beats one without, since only one of them can be judged;
        between two without, confidence decides, which is the "confidence" rule. An exact tie in
        distance also goes to confidence.
        """
        u = None
        if emb is not None:
            u = np.asarray(emb, dtype=np.float32)
            u = u / (np.linalg.norm(u) + 1e-9)
        guardada = t.get("recorte_emb")
        if t["crop"] is None:
            tomar = True
        elif u is None or guardada is None:
            tomar = (u is not None) or (guardada is None and conf > t["recorte_conf"])
        else:
            media = t["emb_sum"] / (np.linalg.norm(t["emb_sum"]) + 1e-9)
            d_nueva = float(np.linalg.norm(u - media))
            d_guardada = float(np.linalg.norm(guardada - media))
            tomar = d_nueva < d_guardada or (d_nueva == d_guardada and conf > t["recorte_conf"])
        if tomar:
            t["crop"], t["recorte_conf"], t["recorte_emb"] = crop, conf, u


    def _summary(self, tid: int, t: dict) -> dict:
        """One track reduced to what association needs: where, how much, what it looks like.

        'pos' is the median over the track's whole life, which is its robust centre;
        'pos_actual' is where it is now, which for a mover is a different place.
        """
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
            "pos": np.median(ii, axis=0),
            "pos_actual": _current_position(ii, t.get("ts")),
            "desplaz": desplaz,
            "conf": t["conf_sum"] / n,
            "emb": emb,
            "frames": t["frames"],
            "crop": t.get("crop"),
            "recorte_conf": t.get("recorte_conf", -1.0),
            "recorte_emb": t.get("recorte_emb"),
            "t0": t.get("t0"),
            "t1": t.get("t1"),
            "cls_votos": dict(t.get("cls_votos") or {}),
            "cls": dominant_class(t.get("cls_votos")),
            "bins": t["bins"],
            **self._movimiento_reciente(ii, t.get("ts")),
        }

    def _movimiento_reciente(self, ii: np.ndarray, ts) -> dict:
        """
        Position now and speed, from the last motion_window_s seconds of sightings, in "looks" mode.

        A straight-line fit against time over that window, evaluated at the last sighting. The speed
        comes with its standard error from the fit residuals; a speed that noise alone could produce
        is reported as None, which leaves the older rule to decide. Empty outside "looks" mode or
        without a clock, so the span mode stays exactly as it was.
        """
        if self.maturity != "looks" or not ts or any(x is None for x in ts):
            return {}
        tt = np.asarray(ts, dtype=float)
        sel = tt >= tt[-1] - self.motion_window_s
        if sel.sum() < 3 or np.ptp(tt[sel]) < 1.0:
            return {}
        x = tt[sel] - tt[-1]
        v = ii[sel]
        coef = np.polynomial.polynomial.polyfit(x, v, 1)
        pos, vel = np.asarray(coef[0]), np.asarray(coef[1])
        resid = v - (coef[0] + np.outer(x, coef[1]))
        dof = max(1, len(x) - 2)
        s_r = math.sqrt(float((resid ** 2).sum()) / (2 * dof))
        se = s_r / math.sqrt(max(1e-9, float(((x - x.mean()) ** 2).sum())))
        rapidez = float(np.linalg.norm(vel))
        return {"pos_reciente": pos, "rapidez": rapidez, "rapidez_se": se,
                "desplaz_ventana": rapidez * float(np.ptp(x)),
                "vel": vel, "t_ultimo": float(tt[-1]),
                "var_pos": 4.0 * s_r ** 2 / len(x)}

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

        A MOVING target is the exception, and it has to be: the spread between its looks is the
        length of its path, not how uncertain its position is, which on a straight run inflated
        the base radius to tens of metres. What is uncertain for a mover is where it is along
        its motion, and the fit residuals are what measure that.

        The bias is taken at the typical distance the candidate was seen from, not averaged,
        because every sighting shares it.
        """
        por_mirada: Dict[int, list] = {}
        for tid in c["tids"]:
            t = self._tracks[tid]
            for imp, mirada in zip(t["imps"], t["imp_bins"]):
                por_mirada.setdefault(mirada, []).append(imp)
        medias = np.array([np.mean(v, axis=0) for v in por_mirada.values()])
        n = len(medias)
        var = float(medias.var(axis=0, ddof=1).mean()) if n >= 2 else 0.0
        if c.get("mobile") and c.get("var_pos") is not None:
            var, n = float(c["var_pos"]), 1
        rangos = [r for tid in c["tids"] for r in self._tracks[tid]["rangos"]]
        if rangos:
            r = float(np.median(rangos))
            sesgo2 = self.gps_sigma_m ** 2 + (r * self.yaw_sigma_rad) ** 2
        else:
            sesgo2 = self.bias_sigma_m ** 2
        return RADIO_95_2D * math.sqrt(sesgo2 + var / max(1, n))

    def _match(self, tk: dict, cands: List[dict]) -> Optional[int]:
        """
        Index of the candidate this track belongs to, or None.

        The rules apply in order -- class, co-occurrence, distance, appearance -- and among the
        candidates that pass all of them the one closest in position and appearance wins.

        A moving candidate is compared where it would BE when this track was seen, not where it was
        last seen, and against the margin that goes with having been unseen that long. Comparing a
        walker against a stale position is what broke one person into three points on the map: on
        the 02ago flight the walking woman's three pieces sit 2.9, 7.0 and 9.1 m apart with a fusion
        radius of 3.5 m, while OSNet scores them 0.41 to 0.61 against a threshold of 0.95. Position
        said three people and appearance said one, and position was the one that was wrong, because
        she had walked. Rejoining asks more of appearance than a static merge does, and refuses
        outright when either side has no appearance at all: guessing that two points on a projected
        path are the same person, with nothing but geometry, is how two people become one.

        TWO NAMES MEAN TWO THINGS, and that veto comes before every other rule, the twin
        exception included: a car is not the person standing beside it however close they are and
        however alike their crops look from the air. Silence on either side is not disagreement,
        so a track with no votes still merges the way it always did, which is what keeps every
        camera without a class working unchanged.

        Two tracks seen together in the same frames are two different things, unless this is the
        duplicate-box case: the same spot and the same appearance.
        """
        radio = self._radio(tk["cls"])
        cuando = tk.get("t_ultimo") if tk.get("t_ultimo") is not None else tk.get("t0")
        mejor, smin = None, math.inf
        for k, c in enumerate(cands):
            if c["mobile"]:
                if not self.rejoin_mobile:
                    continue
                if tk["emb"] is None or c["emb"] is None:
                    continue
                if (cuando is not None and c.get("t_ultimo") is not None
                        and abs(float(cuando) - float(c["t_ultimo"])) > self.rejoin_max_gap_s):
                    continue
            c_cls = dominant_class(c["cls_votos"])
            if (tk["cls"] is not None and c_cls is not None
                    and tk["cls"] != c_cls):
                continue
            radio_c = self._radio_ahora(c, cuando) if c["mobile"] else radio
            emb_max = self.emb_dist_rejoin if c["mobile"] else self.emb_dist_max
            dp = float(np.linalg.norm(tk["pos"] - self._en(c, cuando)))
            if len(tk["frames"] & c["frames"]) >= COOCURRENCIA_MIN:
                es_gemelo = (
                    dp < POS_FRAC_GEMELO * radio
                    and tk["emb"] is not None and c["emb"] is not None
                    and float(np.linalg.norm(tk["emb"] - c["emb"]))
                    < EMB_DIST_GEMELO)
                if not es_gemelo:
                    continue
            if dp >= radio_c:
                continue
            if tk["emb"] is not None and c["emb"] is not None:
                de = float(np.linalg.norm(tk["emb"] - c["emb"]))
                if de >= emb_max:
                    continue
                s = dp / radio_c + 0.5 * de / emb_max
            else:
                s = dp / radio_c
            if s < smin:
                mejor, smin = k, s
        return mejor

    @staticmethod
    def _absorb(c: dict, tk: dict) -> None:
        """Folds a track into a candidate, weighting position and appearance by evidence.

        Positions and appearances average; a photograph cannot, so the clearest of the two crops
        is kept, which is the one the verifier would have chosen. Under crop_choice="appearance"
        every track's crop stays eligible until the report, by which time the candidate's
        appearance has taken all of them in; see _crop_for.

        Two tracks of one target means the evidence spans the union of their intervals.
        """
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
        if tk["crop"] and tk["recorte_conf"] > c["recorte_conf"]:
            c["crop"], c["recorte_conf"] = tk["crop"], tk["recorte_conf"]
        c["recortes"].append((tk["crop"], tk["recorte_emb"], tk["recorte_conf"]))
        if tk["t0"] is not None:
            c["t0"] = tk["t0"] if c["t0"] is None else min(c["t0"], tk["t0"])
            c["t1"] = tk["t1"] if c["t1"] is None else max(c["t1"], tk["t1"])

    def _crop_for(self, c: dict) -> Optional[bytes]:
        """
        The crop a candidate carries: under "appearance", the one of its tracks' crops whose vector
        is closest to the candidate's appearance.

        The candidate's appearance is the mean of its tracks' weighted by their sightings, so a
        track that is most of the evidence pulls the choice towards its own crop, and a short track
        of something else -- a box that caught two people, a patch of ground -- does not speak for
        the candidate however confident the detector was about it. Confidence breaks ties. With no
        vectors to compare the most confident crop is kept, as under "confidence".
        """
        if self.crop_choice != "appearance" or c.get("emb") is None:
            return c.get("crop")
        comparables = [r for r in c["recortes"] if r[0] and r[1] is not None]
        if not comparables:
            return c.get("crop")
        return min(comparables,
                   key=lambda r: (float(np.linalg.norm(r[1] - c["emb"])), -r[2]))[0]

    def _en(self, c: dict, now: Optional[float]) -> np.ndarray:
        """
        Where a candidate is at `now`: a moving one carried forward along its recent velocity.

        A report goes out on its own clock, and the last sighting of a moving target is already some
        time old when it does: a boat at 8 m/s seen one second ago is 8 m away from its last sighting.
        The extrapolation stops at motion_window_s -- past the window its velocity was estimated
        over, following the line is guessing. Static candidates, and any call without `now`, are
        returned where they were estimated.
        """
        if now is None or not c.get("mobile") or c.get("vel") is None or c.get("t_ultimo") is None:
            return c["pos"]
        dt = min(max(0.0, float(now) - c["t_ultimo"]), self.extrapolation_max_s)
        return np.asarray(c["pos"]) + np.asarray(c["vel"]) * dt

    def _radio_ahora(self, c: dict, now: Optional[float]) -> float:
        """
        The 95 % radius at `now`: for a moving candidate, grown by how far it can have gone unseen.

        The radius was the uncertainty at the last sighting. Between sightings a moving target keeps
        going, and extrapolating along its velocity is right only while it does not turn. So the
        margin grows by speed times the time since it was last seen: on a synthetic patrol at 4 m/s
        the reported point was 22.4 m from the target while its circle still said 5 m. A margin that
        does not hold the truth sends someone to the wrong place with confidence; a wide one says
        honestly that the drone has lost precise track.

        Both distances go into the margin, because the reported point has itself been carried
        forward by up to extrapolation_max_s: a target that turned round at its last sighting
        can be that far behind the point, plus as far again as it went since.
        """
        base = self._radius(c)
        if now is None or not c.get("mobile") or c.get("vel") is None or c.get("t_ultimo") is None:
            return base
        edad = max(0.0, float(now) - c["t_ultimo"])
        adelantado = min(edad, self.extrapolation_max_s)
        return base + float(np.linalg.norm(c["vel"])) * (edad + adelantado)

    def candidates(self, preliminary: bool = False, with_tracks: bool = False,
                   now: Optional[float] = None) -> List[dict]:
        """
        Returns the current candidate list, mobiles first, then by descending evidence.

        Each candidate: {x, y, cls, n_obs, conf, mobile, mature, crop}. For a mobile candidate (x, y) is
        its CURRENT position (a mobile's lifetime median points at the middle of its path).
        Static candidates report the lifetime median, which is the point of accumulating views.

        Args:
            now: the time the report is for, on the observation clock. A moving candidate is
                given where it is then, not where it was last seen; see _en. In "looks" mode each
                candidate also carries age_s, the seconds since its last sighting: a candidate is never
                forgotten, so a target lost a minute ago is still reported, and measured in an escort
                simulation a drone that could not tell old news from a fresh sighting chased points
                67-97 m away from the target.
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

        How a track becomes a MOBILE candidate, in "looks" mode: its recent speed has to clear
        both the threshold and three standard errors of its own estimate, AND the fitted motion
        has to carry it further than projection noise moves a standing target. The older
        displacement rule stays as an alternative so a patrol does not flicker to "static" at
        the instant it turns round. A moving track still has to be somebody, so the mobile
        branch asks _match first and only opens a candidate when nobody there matches; it used
        to be the one path that never asked, which turned a standing operator whose duplicate
        boxes fake a speed into several points on the map.

        Fragments come last and only ever reinforce. Every candidate above them was opened by a
        track that lasted, so no amount of short noise can add a point to the map.

        Rows come out mature first, then mobiles, then by evidence, so whatever the caller
        truncates it truncates the least certain rows. Each row carries:

        cls
            What it is, next to where it is. None when no camera ever said.

        crop
            Raw JPEG bytes, or None. Serialising it is the transport's problem, not this
            layer's; the protocol base64-encodes it on the way out.

        emb
            The appearance vector, same convention as the crop. It leaves the drone because
            deciding that two drones are looking at one target is a comparison neither of them
            can make alone, and position is not enough: two people three metres apart are two
            people.

        looks, looks_min, evidence, radius_m
            How much independent evidence there is, how much it takes to be reported, and the
            fraction of the way there; radius_m is how far off the point may be. Only in
            "looks" mode, so the validated chain's report is unchanged byte for byte. The count
            travels with its threshold because the count alone cannot be read: a number of
            looks means nothing to an operator who does not know where the bar is.

        duty
            Of the frames this layer was handed a detection in while the candidate was alive,
            the fraction in which the candidate itself was seen. A static false positive is a
            flicker spread thin over a long time; a person being tracked is dense while they
            are in view. Reported as a measurement and nothing more: no threshold travels with
            it, because none is enforced, and publishing a bar nobody applies would be the same
            fault this field exists to fix. Frames delivered are the unit of opportunity and
            seconds are NOT, for which see NOTES.md.

        age_s
            Seconds since the candidate was last seen. The position of a lost target keeps
            being reported, and without this a consumer cannot tell a fresh sighting from old
            news.
        """
        cands: List[dict] = []
        for tk in sorted(self._track_summaries(), key=lambda p: -p["n"]):
            regla_vieja = (tk["desplaz"] > self._disp_movil(tk["cls"]) and tk["n"] >= self.n_movil
                           and self._has_covered(tk, self.mobile_dur_s, self.span_movil))
            rapido = ("rapidez" in tk and tk["rapidez"] > max(self.mobile_speed_mps, 3.0 * tk["rapidez_se"])
                      and tk["desplaz_ventana"] > self._disp_movil(tk["cls"]))
            if rapido or regla_vieja:
                ya = self._match(tk, cands)
                if ya is not None:
                    self._absorb(cands[ya], tk)
                    cands[ya]["mobile"] = True
                    cands[ya]["pos"] = np.asarray(
                        tk["pos_reciente"] if "pos_reciente" in tk else tk["pos_actual"]).copy()
                    for campo in ("vel", "t_ultimo", "var_pos"):
                        if tk.get(campo) is not None:
                            cands[ya][campo] = tk.get(campo)
                    continue
                ahora = tk["pos_reciente"] if "pos_reciente" in tk else tk["pos_actual"]
                cands.append({"mobile": True, "pos": np.asarray(ahora).copy(),
                              "vel": tk.get("vel"), "t_ultimo": tk.get("t_ultimo"),
                              "var_pos": tk.get("var_pos"),
                              "emb": tk["emb"], "conf": tk["conf"],
                              "n": tk["n"], "frames": set(tk["frames"]),
                              "crop": tk["crop"],
                              "recorte_conf": tk["recorte_conf"],
                              "recortes": [(tk["crop"], tk["recorte_emb"], tk["recorte_conf"])],
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
                              "recortes": [(tk["crop"], tk["recorte_emb"], tk["recorte_conf"])],
                              "t0": tk["t0"], "t1": tk["t1"],
                              "cls_votos": dict(tk["cls_votos"]), "tids": [tk["tid"]],
                              "bins": set(tk["bins"])})
            else:
                self._absorb(cands[mejor], tk)

        if self.reinforce_with_fragments:
            for tk in sorted(self._fragment_summaries(), key=lambda p: -p["n"]):
                mejor = self._match(tk, cands)
                if mejor is not None:
                    self._absorb(cands[mejor], tk)

        vistos = np.fromiter(sorted(self._frames_vistos), dtype=np.int64, count=len(self._frames_vistos))

        def duty(c):
            if not c["frames"]:
                return 0.0
            lo, hi = min(c["frames"]), max(c["frames"])
            entregados = int(np.searchsorted(vistos, hi, "right") - np.searchsorted(vistos, lo, "left"))
            return len(c["frames"]) / max(1, entregados)

        def mature(c):
            if self.maturity == "looks":
                return len(c["bins"]) >= self.report_min_looks
            return (c["n"] >= self.n_reporte
                    and self._has_covered(c, self.report_dur_s, self.span_reporte))

        out = [c for c in cands if mature(c) or preliminary]
        out.sort(key=lambda c: (not mature(c), not c["mobile"], -c["n"]))
        return [{
            "x": round(float(self._en(c, now)[0]), 2),
            "y": round(float(self._en(c, now)[1]), 2),
            "cls": dominant_class(c["cls_votos"]),
            "n_obs": int(c["n"]),
            "conf": round(float(c["conf"]), 3),
            "mobile": bool(c["mobile"]),
            "mature": mature(c),
            "crop": self._crop_for(c),
            "emb": c.get("emb"),
            **({"looks": len(c["bins"]),
                "looks_min": int(self.report_min_looks),
                "evidence": round(min(1.0, len(c["bins"]) / max(1, self.report_min_looks)), 3),
                "duty": round(duty(c), 3),
                "radius_m": round(self._radio_ahora(c, now), 2)}
               if self.maturity == "looks" else {}),
            **({"age_s": round(float(now) - float(c["t1"]), 2)}
               if self.maturity == "looks" and now is not None and c.get("t1") is not None else {}),
            **({"tracks": sorted(c["tids"])} if with_tracks else {}),
        } for c in out]
