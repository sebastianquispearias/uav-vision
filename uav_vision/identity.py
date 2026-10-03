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

# Re-associating a moving candidate is the one case where position argues for splitting and
# appearance argues for joining, so appearance decides, and it is asked for more than the 0.95 that
# decides a static merge.
#
# THIS NUMBER IS NOT SAFE AND THAT IS WHY REJOINING IS OFF BY DEFAULT. Measured over the 02ago
# flight, track by track, against the letters a human put on every box:
#
#   pieces of the SAME person   0.26 (the operator) and 0.41 0.41 0.60 0.61 0.64 0.81 (the walker)
#   DIFFERENT people            0.64 (B vs C)  0.67 (H vs C)  0.68 (H vs B)  0.73 ...
#
# The two ranges OVERLAP by 0.18: seven pairs of different people are closer than the furthest pair
# of the same person. No threshold separates them. What joins the walker without fusing anybody is a
# window between 0.61 and 0.67, and 0.63 sits in it -- chosen by looking at the flight it is judged
# on, which is the error this repository has already paid for once. At 0.70 the boy on the balcony
# is absorbed into somebody else and DISAPPEARS FROM THE MAP, which in a search is the worst failure
# there is: the operator is not told there is a person there at all.
#
# It stays off until a second flight exists to choose the number on.
EMB_DIST_REUNE = 0.63

# Radius of the circle holding 95 % of a two-dimensional isotropic Gaussian, in sigmas:
# sqrt(-2 ln 0.05). Used to turn a per-axis position uncertainty into something an operator can
# draw on a map and walk to.
RADIO_95_2D = 2.4477

# The slant range, in metres, at which the chain's 2.4 m median error was measured: the median
# camera-to-target distance of the operator's observations on flight 3 (p10 10.1 m, p90 53.0 m, at a
# median height of 17.1 m). A position margin quoted without the range it holds at is incomplete:
# a heading error moves the impact by range times angle.
RANGO_REFERENCIA_M = 20.3

# Horizontal GPS standard deviation assumed for a consumer receiver, per axis. Not measured here:
# with one calibration point and two error sources, one of them has to be assumed, and this is the
# better known of the two. The heading error then follows from the measured floor.
GPS_SIGMA_M = 1.5


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
            return np.asarray(ajuste[0])            # the line at x = 0, the last sighting
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
        maturity: what "enough evidence" means. "looks", the default, counts independent looks;
            "span" is the rule the chain used until it was measured against a flight, and is kept
            so those numbers stay reproducible. "span" is the time between the first and the
            last sighting: track_dur_s to open a track, report_dur_s to report. It measures the
            wrong thing. On flight 3 a static object
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
            not measurements. The default of 20 is the value that, on flight 3 with the flight's
            tracker, confirmed the same targets the span rule did in 22 s instead of 52, and met the
            provisional mission requirements (docs/requisitos_mision.json); 5 confirms in 6 s but
            also confirmed a non-person.
        look_s: the length of one look, in seconds.
        bias_sigma_m: per-axis standard deviation of the error that more looks cannot average
            away -- GPS and heading bias, shared by every sighting of a flight. The default
            comes from the chain's measured median error on real flights, 2.4 m: for a
            two-dimensional Gaussian the median radial error is 1.1774 sigma. It holds at the range
            it was measured at, RANGO_REFERENCIA_M; when observations carry range_m the shared error
            is modelled as sqrt(gps_sigma_m^2 + (range * yaw_sigma)^2), with the heading error solved
            so the model gives bias_sigma_m back at that range. From 12 m the 95 % radius is about
            4.2 m, from 20 m 5.0 m, from 90 m about 15 m.
        gps_sigma_m: the GPS part of that error, per axis. Assumed, see GPS_SIGMA_M.
        motion_window_s: how far back, in seconds, "where is it now and how fast is it going" looks,
            in "looks" mode. A target that moves has to be described by its recent past. Over its
            whole life, a boat patrolling back and forth has its median in the middle of the patrol
            and a straight line through its last quarter of sightings -- 20 to 29 s, 81 to 235 m on
            flight 3 with a synthetic target at 4 and 8 m/s -- crosses the turns and points there
            too: it was reported 10-14 m from where it was.
        mobile_speed_mps: speed above which a target counts as moving, in "looks" mode, provided it
            also exceeds three times the standard error of its own estimate, so that projection
            noise on a standing person is not read as motion. A decision, not a measurement: a
            walking person is about 1.4 m/s. The speed alone is not enough on a short track: on flight 3
            the standing operator left tracks of 6-21 sightings over 1-5 s with fitted speeds of 0.75 to
            1.95 m/s, and each became a separate mobile candidate -- eight points for one person. So the
            fitted motion must also carry the target further than the mobile displacement across the
            sightings it was fitted on, the distance projection noise alone can move a standing target.
        extrapolation_max_s: how far ahead, in seconds, a moving candidate is carried from its last
            sighting to the report time. Separate from the window its velocity is estimated over: a
            target that turns keeps going on paper for as long as this allows. With 5 s, a synthetic
            patrol turning every 6.25 s at 8 m/s was reported past its turn. Measured on that patrol
            (error median / p90 at 1.5, 4 and 8 m/s): 5 s gives 0.35/11.39, 2.80/21.38, 7.91/17.09 m;
            3 s gives 2.30/14.40, 2.82/13.44, 6.69/17.09 m; 2 s gives 2.56/15.90, 1.89/11.29,
            6.69/17.91 m. No value wins at every speed; 3 s has the smallest worst case, and that is
            the choice -- a decision about which failure to tolerate, not an optimum.
        crop_choice: which crop a candidate carries. "confidence", the default, keeps the most
            confident sighting of each track and the most confident of its tracks. "appearance"
            keeps, per track, the sighting whose vector is closest to the track's mean appearance,
            and per candidate the track crop closest to the candidate's, so the photograph shows
            what the evidence mostly is. The difference is the case of a box that caught two people
            or a person next to clutter: the detector is surest there and the appearance filter on
            the ground agrees with the detector. On flight 3, with the stand-in tracker, the only
            confirmed candidate that is mostly not a person carried such a box under "confidence"
            and passed the filter; under "appearance" it carries one of its non-person boxes and is
            filtered out. Opt-in because the same measurement also shows the cost: the crops chosen
            are less confident (median 0.41 against 0.77), and one more preliminary non-person
            passes the filter.
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
        # The heading error that, with gps_sigma_m, reproduces bias_sigma_m at the reference range.
        self.yaw_sigma_rad = math.sqrt(max(0.0, bias_sigma_m ** 2 - gps_sigma_m ** 2)) / RANGO_REFERENCIA_M
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
        # Every frame index this layer was ever handed a detection in. It is the denominator of
        # opportunity: asking in what fraction of a candidate's life it was seen is only
        # meaningful against the frames something was seen in at all. Seconds will not do,
        # because the camera delivers in bursts: on flight 3 the operator is in nearly every
        # frame that exists and still shows 0.18 looks per second.
        #
        # Note what this is NOT: a frame the detector found nothing in never reaches observe, so
        # it is not counted here. The denominator is therefore frames in which detection was
        # producing something, not frames the camera captured. That is the stricter of the two
        # readings -- it refuses to credit a candidate for frames where nothing was working --
        # and it is the one available without a second channel from the camera.
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

    # -- ingest ------------------------------------------------------------

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
        """
        sello = t          # `t` below is the track record; keep the timestamp first
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
                # How far the fitted motion carries the target across the sightings it was fitted on.
                "desplaz_ventana": rapidez * float(np.ptp(x)),
                "vel": vel, "t_ultimo": float(tt[-1]),
                # Variance of the fitted position at the last sighting, per axis. A line's value at the
                # end of its window is about four times as uncertain as its mean.
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
            # For a moving target the spread between looks is the length of its path, not how
            # uncertain its position is: on a straight run at 6 m/s it made the base radius 19.8 m.
            # What is uncertain is where it is along its motion, which the fit residuals measure.
            var, n = float(c["var_pos"]), 1
        rangos = [r for tid in c["tids"] for r in self._tracks[tid]["rangos"]]
        if rangos:
            # The bias is shared by every sighting, so it is not averaged: it is taken at the typical
            # distance the candidate was seen from.
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
            radio_c = self._radio_ahora(c, cuando) if c["mobile"] else radio
            emb_max = self.emb_dist_rejoin if c["mobile"] else self.emb_dist_max
            dp = float(np.linalg.norm(tk["pos"] - self._en(c, cuando)))
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
        # Under crop_choice="appearance" every track's crop stays eligible until the report, when
        # the candidate's appearance has taken in all of them; see _crop_for.
        c["recortes"].append((tk["crop"], tk["recorte_emb"], tk["recorte_conf"]))
        # Two tracks of one target: the evidence spans the union of their intervals.
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
        """
        base = self._radius(c)
        if now is None or not c.get("mobile") or c.get("vel") is None or c.get("t_ultimo") is None:
            return base
        edad = max(0.0, float(now) - c["t_ultimo"])
        # The reported point has itself been carried forward by up to extrapolation_max_s, so a target
        # that turned round at its last sighting can be that far behind the point plus as far again
        # as it went since: both distances go into the margin.
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
        """
        cands: List[dict] = []
        for tk in sorted(self._track_summaries(), key=lambda p: -p["n"]):
            regla_vieja = (tk["desplaz"] > self._disp_movil(tk["cls"]) and tk["n"] >= self.n_movil
                           and self._has_covered(tk, self.mobile_dur_s, self.span_movil))
            # In "looks" mode a target is moving when its recent speed clears both the threshold and
            # three standard errors of its own estimate. The older rule stays as an alternative so a
            # patrol does not flicker to "static" at the instant it turns round.
            # A speed that carries the target less far than projection noise moves a standing one is not
            # motion: the displacement over the fitted sightings must also clear the mobile displacement.
            rapido = ("rapidez" in tk and tk["rapidez"] > max(self.mobile_speed_mps, 3.0 * tk["rapidez_se"])
                      and tk["desplaz_ventana"] > self._disp_movil(tk["cls"]))
            if rapido or regla_vieja:
                # A moving track still has to be somebody. Opening a candidate without asking whether
                # one already exists for this person is what turned a standing operator, whose
                # duplicate boxes fake a speed, into several points on the map: the mobile branch was
                # the only path that never consulted _match. Ask first, and only open when the answer
                # is that nobody here matches.
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

        # Fragments come last and only reinforce: every candidate above was opened by a track
        # that lasted, so no amount of short noise can add a point to the map.
        if self.reinforce_with_fragments:
            for tk in sorted(self._fragment_summaries(), key=lambda p: -p["n"]):
                mejor = self._match(tk, cands)
                if mejor is not None:
                    self._absorb(cands[mejor], tk)

        # Of the frames this layer was handed a detection in while this candidate was alive, the
        # fraction in which the candidate itself was seen. A static false positive is a flicker spread thin over a
        # long time; a person being tracked is dense while she is in view. Measured on flight 3
        # against the letters a human put on every box, that is what tells them apart: the six
        # ghosts top out at 0.397 and the seven real-person candidates floor at 0.714, with
        # nothing in between. Sightings per second does the same on this flight and must not be
        # used: it is not scale free, so a figure of 5.7 sightings per second exists only
        # because this recording is bursty at 1.64 FPS, and on a steady 3 FPS board the same
        # quantity cannot exceed 3. Frames delivered is the unit of opportunity; seconds are not.
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
        # Mature first, then mobiles, then by evidence: whatever the caller truncates, it
        # truncates the least certain rows.
        out.sort(key=lambda c: (not mature(c), not c["mobile"], -c["n"]))
        return [{
            "x": round(float(self._en(c, now)[0]), 2),
            "y": round(float(self._en(c, now)[1]), 2),
            # What it is, next to where it is. None when no camera ever said.
            "cls": dominant_class(c["cls_votos"]),
            "n_obs": int(c["n"]),
            "conf": round(float(c["conf"]), 3),
            "mobile": bool(c["mobile"]),
            "mature": mature(c),
            # Raw JPEG bytes, or None. Serialising it is the transport's problem, not this
            # layer's; the protocol base64-encodes it on the way out.
            "crop": self._crop_for(c),
            # The appearance vector, same convention as the crop: raw here, encoded by the
            # transport. It leaves the drone because deciding that two drones are looking at
            # one target is a comparison neither of them can make alone, and position is not
            # enough: two people three metres apart are two people.
            "emb": c.get("emb"),
            # How much independent evidence there is, how much it takes to be reported, and the
            # fraction of the way there. Only in "looks" mode, so the validated chain's report is
            # unchanged byte for byte. The count travels with its threshold because the count
            # alone cannot be read: "22 looks" means nothing to an operator who does not know
            # that 20 is the bar. radius_m is how far off the point may be.
            **({"looks": len(c["bins"]),
                "looks_min": int(self.report_min_looks),
                "evidence": round(min(1.0, len(c["bins"]) / max(1, self.report_min_looks)), 3),
                # The density defined above, reported as a measurement and nothing more: no
                # threshold travels with it yet because none is enforced yet, and publishing a
                # bar nobody applies would be the same fault this field exists to fix.
                "duty": round(duty(c), 3),
                "radius_m": round(self._radio_ahora(c, now), 2)}
               if self.maturity == "looks" else {}),
            # Seconds since the candidate was last seen. The position of a lost target keeps being reported, and
            # without this a consumer cannot tell a fresh sighting from old news.
            **({"age_s": round(float(now) - float(c["t1"]), 2)}
               if self.maturity == "looks" and now is not None and c.get("t1") is not None else {}),
            **({"tracks": sorted(c["tids"])} if with_tracks else {}),
        } for c in out]
