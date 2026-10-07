"""The multi-view fusion algorithms, exercised: rays in, a position out.

uav_vision/fusion.py carries the paper's bearing-only geolocation: the point of closest
approach between two rays, the N-ray gradient descent over orthogonal distances, their
ground-constrained variants, and three robust wrappers (RANSAC, PROSAC, LO-RANSAC). It had
NO GATE in this repository until now, and the chain that flies does not call it at all --
`vision_protocol.py` intersects each ray with the ground plane and reaches consensus over 2D
impact points instead. So this file guards code that is correct, present, and unused, which is
exactly the code that rots without anybody noticing.

These assertions come from the GrADyS showcase (gradys-sim-nextgen/showcases/geolocation),
pointed at this package's copy of the module. They are kept as pytest classes rather than
rewritten, because rewriting 31 working checks to change their shape buys nothing.

WHY THE MAIN BLOCK AT THE BOTTOM EXISTS, and it is not boilerplate. This repository's
conftest.py overrides pytest collection and runs every test file AS A SCRIPT
(`[sys.executable, path]`). A file of pytest classes with no entry point therefore executes
its imports, defines its classes and exits 0: measured when this file first landed, 31 test
methods and 28 assertions ran ZERO times and the suite reported it as passing. A gate that
cannot fail proves nothing, so the block below calls every method and says how many ran.

Run with: python tests/test_fusion.py
"""

import math

import numpy as np
import pytest

from uav_vision.fusion import (
    gradient_descent_uniform,
    gradient_descent_weighted,
    lo_ransac_fusion,
    orthogonal_distance,
    prosac_fusion,
    ransac_fusion,
    ray_intersection_2,
)

SEED = 42

# ---------------------------------------------------------------------------
# Helpers: build measurements pointing from drone positions toward a target
# ---------------------------------------------------------------------------

def _make_measurement(drone_pos, target_pos):
    """Create a perfect (noiseless) measurement from drone toward target."""
    d = np.asarray(target_pos, dtype=np.float64) - np.asarray(drone_pos, dtype=np.float64)
    d = d / np.linalg.norm(d)
    return (tuple(drone_pos), (float(d[0]), float(d[1]), float(d[2])))


def _make_noisy_measurement(drone_pos, target_pos, sigma_deg, rng):
    """Create a measurement with angular noise added to the direction."""
    d = np.asarray(target_pos, dtype=np.float64) - np.asarray(drone_pos, dtype=np.float64)
    d = d / np.linalg.norm(d)
    # Add small random rotation via perturbation in the plane perpendicular to d
    perturb = rng.normal(0, 1, size=3)
    perturb -= np.dot(perturb, d) * d  # project out component along d
    norm_p = np.linalg.norm(perturb)
    if norm_p > 1e-12:
        perturb = perturb / norm_p
    angle = math.radians(rng.normal(0, sigma_deg))
    d_noisy = d * math.cos(angle) + perturb * math.sin(angle)
    d_noisy = d_noisy / np.linalg.norm(d_noisy)
    return (tuple(drone_pos), (float(d_noisy[0]), float(d_noisy[1]), float(d_noisy[2])))


# ---------------------------------------------------------------------------
# Target and drone setup for tests
# ---------------------------------------------------------------------------

# Target on the ground
TARGET = (50.0, 50.0, 0.0)

# 3 drones in V formation at 50m altitude
DRONES_V = [
    (50.0, 30.0, 50.0),   # center, behind
    (30.0, 35.0, 50.0),   # left wing
    (70.0, 35.0, 50.0),   # right wing
]

# Perfect measurements
PERFECT_MEASUREMENTS = [_make_measurement(d, TARGET) for d in DRONES_V]


# ---------------------------------------------------------------------------
# Tests: orthogonal_distance
# ---------------------------------------------------------------------------

class TestOrthogonalDistance:
    def test_point_on_ray(self):
        """Distance is 0 when point lies exactly on the ray."""
        origin = (0.0, 0.0, 50.0)
        direction = (0.0, 0.0, -1.0)
        point = (0.0, 0.0, 10.0)
        assert orthogonal_distance(point, origin, direction) == pytest.approx(0.0, abs=1e-10)

    def test_known_distance(self):
        """Ray along z-axis, point offset in x by 5m."""
        origin = (0.0, 0.0, 50.0)
        direction = (0.0, 0.0, -1.0)
        point = (5.0, 0.0, 0.0)
        assert orthogonal_distance(point, origin, direction) == pytest.approx(5.0, abs=1e-10)

    def test_unnormalised_direction(self):
        """Should handle non-unit direction vectors."""
        origin = (0.0, 0.0, 50.0)
        direction = (0.0, 0.0, -100.0)  # not unit
        point = (3.0, 4.0, 0.0)
        assert orthogonal_distance(point, origin, direction) == pytest.approx(5.0, abs=1e-10)

    def test_non_negative(self):
        rng = np.random.default_rng(SEED)
        for _ in range(100):
            p = tuple(rng.uniform(-100, 100, 3))
            o = tuple(rng.uniform(-100, 100, 3))
            d = tuple(rng.normal(0, 1, 3))
            assert orthogonal_distance(p, o, d) >= 0


# ---------------------------------------------------------------------------
# Tests: ray_intersection_2
# ---------------------------------------------------------------------------

class TestRayIntersection2:
    def test_perfect_two_rays(self):
        """Two noiseless rays should intersect very close to target."""
        result = ray_intersection_2(PERFECT_MEASUREMENTS[0], PERFECT_MEASUREMENTS[1])
        error = np.linalg.norm(np.array(result) - np.array(TARGET))
        assert error < 0.01

    def test_three_different_pairs(self):
        """All three pairs of perfect rays should give close results."""
        pairs = [(0, 1), (0, 2), (1, 2)]
        for i, j in pairs:
            result = ray_intersection_2(PERFECT_MEASUREMENTS[i], PERFECT_MEASUREMENTS[j])
            error = np.linalg.norm(np.array(result) - np.array(TARGET))
            assert error < 0.01, f"Pair ({i},{j}) error: {error:.4f}"

    def test_parallel_rays_no_crash(self):
        """Parallel rays should return midpoint of origins, not crash."""
        m1 = ((0.0, 0.0, 50.0), (0.0, 0.0, -1.0))
        m2 = ((10.0, 0.0, 50.0), (0.0, 0.0, -1.0))
        result = ray_intersection_2(m1, m2)
        assert isinstance(result, tuple)
        assert len(result) == 3
        # Should be midpoint of origins
        assert result[0] == pytest.approx(5.0)
        assert result[1] == pytest.approx(0.0)
        assert result[2] == pytest.approx(50.0)

    def test_forward_clamping(self):
        """Rays pointing away from each other should still produce a result."""
        # Two rays pointing away — alpha/gamma would be negative without clamping
        m1 = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))   # pointing right
        m2 = ((10.0, 0.0, 0.0), (1.0, 0.0, 0.0))   # also pointing right
        result = ray_intersection_2(m1, m2)
        # Parallel, returns midpoint
        assert isinstance(result, tuple)


# ---------------------------------------------------------------------------
# Tests: gradient_descent_uniform
# ---------------------------------------------------------------------------

class TestGradientDescentUniform:
    def test_perfect_no_noise(self):
        """With perfect measurements, error should be < 0.01m."""
        result = gradient_descent_uniform(PERFECT_MEASUREMENTS)
        error = np.linalg.norm(np.array(result) - np.array(TARGET))
        assert error < 0.01

    def test_with_moderate_noise(self):
        """With 2° angular noise, error should be reasonable (< 5m at 50m altitude)."""
        rng = np.random.default_rng(SEED)
        noisy = [_make_noisy_measurement(d, TARGET, sigma_deg=2.0, rng=rng)
                 for d in DRONES_V]
        result = gradient_descent_uniform(noisy)
        error = np.linalg.norm(np.array(result) - np.array(TARGET))
        assert error < 5.0

    def test_two_measurements(self):
        """Should work with exactly 2 measurements."""
        result = gradient_descent_uniform(PERFECT_MEASUREMENTS[:2])
        error = np.linalg.norm(np.array(result) - np.array(TARGET))
        assert error < 0.01

    def test_fewer_than_2_raises(self):
        with pytest.raises(ValueError, match="At least 2"):
            gradient_descent_uniform([PERFECT_MEASUREMENTS[0]])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="At least 2"):
            gradient_descent_uniform([])


# ---------------------------------------------------------------------------
# Tests: gradient_descent_weighted
# ---------------------------------------------------------------------------

class TestGradientDescentWeighted:
    def test_perfect_no_noise(self):
        """Uniform weights on perfect data should match uniform GD."""
        weights = [1.0, 1.0, 1.0]
        result = gradient_descent_weighted(PERFECT_MEASUREMENTS, weights)
        error = np.linalg.norm(np.array(result) - np.array(TARGET))
        assert error < 0.01

    def test_weighted_beats_uniform_with_mixed_quality(self):
        """When some measurements are noisy and some clean, weighting should help.

        Setup: 2 good measurements (low noise) + 2 bad measurements (high noise).
        Weighted GD with high weight on good ones should beat uniform GD.
        """
        drones = [
            (30.0, 20.0, 50.0),
            (70.0, 20.0, 50.0),
            (20.0, 30.0, 50.0),
            (80.0, 30.0, 50.0),
        ]
        # The upstream version built one set of measurements here and never used it: the loop
        # below rebuilds them per trial with its own generator, which is what the averaged
        # comparison needs. ruff found the dead assignment when the file entered this repo.

        # Weighted: trust good measurements much more
        weights = [100.0, 100.0, 1.0, 1.0]

        n_trials = 30
        uniform_errors = []
        weighted_errors = []
        for trial in range(n_trials):
            trial_rng = np.random.default_rng(SEED + trial)
            good_t = [_make_noisy_measurement(drones[i], TARGET, 0.5, trial_rng) for i in range(2)]
            bad_t = [_make_noisy_measurement(drones[i], TARGET, 10.0, trial_rng) for i in range(2, 4)]
            meas_t = good_t + bad_t

            r_u = gradient_descent_uniform(meas_t)
            r_w = gradient_descent_weighted(meas_t, weights)
            uniform_errors.append(np.linalg.norm(np.array(r_u) - np.array(TARGET)))
            weighted_errors.append(np.linalg.norm(np.array(r_w) - np.array(TARGET)))

        assert np.mean(weighted_errors) < np.mean(uniform_errors)

    def test_weight_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="must match"):
            gradient_descent_weighted(PERFECT_MEASUREMENTS, [1.0, 1.0])

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="positive"):
            gradient_descent_weighted(PERFECT_MEASUREMENTS, [1.0, -1.0, 1.0])

    def test_zero_weight_raises(self):
        with pytest.raises(ValueError, match="positive"):
            gradient_descent_weighted(PERFECT_MEASUREMENTS, [1.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# Tests: ransac_fusion
# ---------------------------------------------------------------------------

class TestRansacFusion:
    def test_perfect_no_outliers(self):
        """RANSAC on clean data should converge close to target."""
        rng = np.random.default_rng(SEED)
        result = ransac_fusion(PERFECT_MEASUREMENTS, rng=rng)
        error = np.linalg.norm(np.array(result) - np.array(TARGET))
        assert error < 0.1

    def test_robust_to_outliers(self):
        """RANSAC should tolerate 1-2 outliers among good measurements."""
        rng = np.random.default_rng(SEED)

        # 5 good measurements from different positions
        drones_good = [
            (30.0, 20.0, 50.0),
            (70.0, 20.0, 50.0),
            (50.0, 20.0, 50.0),
            (40.0, 25.0, 50.0),
            (60.0, 25.0, 50.0),
        ]
        good_meas = [_make_noisy_measurement(d, TARGET, 1.0, rng) for d in drones_good]

        # 2 outliers pointing at completely wrong target
        wrong_target = (200.0, 200.0, 0.0)
        outlier_drones = [(45.0, 30.0, 50.0), (55.0, 30.0, 50.0)]
        outlier_meas = [_make_measurement(d, wrong_target) for d in outlier_drones]

        all_meas = good_meas + outlier_meas
        result = ransac_fusion(all_meas, n_iterations=200, threshold_m=5.0,
                               rng=np.random.default_rng(SEED))
        error = np.linalg.norm(np.array(result) - np.array(TARGET))
        assert error < 5.0, f"RANSAC error with outliers: {error:.2f}m"

    def test_fewer_than_2_raises(self):
        with pytest.raises(ValueError, match="At least 2"):
            ransac_fusion([PERFECT_MEASUREMENTS[0]])

    def test_reproducibility(self):
        r1 = ransac_fusion(PERFECT_MEASUREMENTS, rng=np.random.default_rng(SEED))
        r2 = ransac_fusion(PERFECT_MEASUREMENTS, rng=np.random.default_rng(SEED))
        assert r1 == r2


# ---------------------------------------------------------------------------
# Tests: prosac_fusion
# ---------------------------------------------------------------------------

class TestProsacFusion:
    def test_perfect_no_outliers(self):
        rng = np.random.default_rng(SEED)
        confs = [0.9, 0.8, 0.7]
        result = prosac_fusion(PERFECT_MEASUREMENTS, confs, rng=rng)
        error = np.linalg.norm(np.array(result) - np.array(TARGET))
        assert error < 0.1

    def test_confidence_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="confidences"):
            prosac_fusion(PERFECT_MEASUREMENTS, [0.9, 0.8])

    def test_fewer_than_2_raises(self):
        with pytest.raises(ValueError, match="At least 2"):
            prosac_fusion([PERFECT_MEASUREMENTS[0]], [0.9])

    def test_reproducibility(self):
        confs = [0.9, 0.8, 0.7]
        r1 = prosac_fusion(PERFECT_MEASUREMENTS, confs, rng=np.random.default_rng(SEED))
        r2 = prosac_fusion(PERFECT_MEASUREMENTS, confs, rng=np.random.default_rng(SEED))
        assert r1 == r2


# ---------------------------------------------------------------------------
# Tests: lo_ransac_fusion
# ---------------------------------------------------------------------------

class TestLoRansacFusion:
    def test_perfect_no_outliers(self):
        rng = np.random.default_rng(SEED)
        result = lo_ransac_fusion(PERFECT_MEASUREMENTS, rng=rng)
        error = np.linalg.norm(np.array(result) - np.array(TARGET))
        assert error < 0.1

    def test_fewer_than_2_raises(self):
        with pytest.raises(ValueError, match="At least 2"):
            lo_ransac_fusion([PERFECT_MEASUREMENTS[0]])

    def test_reproducibility(self):
        r1 = lo_ransac_fusion(PERFECT_MEASUREMENTS, rng=np.random.default_rng(SEED))
        r2 = lo_ransac_fusion(PERFECT_MEASUREMENTS, rng=np.random.default_rng(SEED))
        assert r1 == r2


# ---------------------------------------------------------------------------
# Tests: degenerate / edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_nearly_parallel_rays_gd(self):
        """GD with nearly parallel rays: should not crash, may have large error."""
        m1 = ((0.0, 0.0, 50.0), (0.0, 0.001, -1.0))
        m2 = ((10.0, 0.0, 50.0), (0.0, 0.002, -1.0))
        result = gradient_descent_uniform([m1, m2])
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(math.isfinite(v) for v in result)

    def test_exactly_two_measurements_gd_matches_intersection(self):
        """With 2 perfect measurements, GD and ray_intersection should agree."""
        m1, m2 = PERFECT_MEASUREMENTS[0], PERFECT_MEASUREMENTS[1]
        ri_result = np.array(ray_intersection_2(m1, m2))
        gd_result = np.array(gradient_descent_uniform([m1, m2]))
        diff = np.linalg.norm(ri_result - gd_result)
        assert diff < 0.1, f"GD and ray_intersection differ by {diff:.4f}m"


if __name__ == "__main__":
    # Every Test* class, every test_* method, in definition order. The count is printed because
    # the failure this guards against is silent: a file that runs nothing also reports no error.
    clases = [v for k, v in sorted(globals().items())
              if k.startswith("Test") and isinstance(v, type)]
    corridos = 0
    for clase in clases:
        metodos = [m for m in vars(clase) if m.startswith("test")]
        print("%-28s %d metodos" % (clase.__name__, len(metodos)))
        instancia = clase()
        for nombre in sorted(metodos):
            getattr(instancia, nombre)()
            corridos += 1
    print()
    assert clases, "no se encontro ninguna clase Test*: la coleccion quedo vacia"
    assert corridos >= 31, (
        "corrieron %d metodos y el archivo tiene al menos 31. Si se quitaron tests a proposito, "
        "bajar este numero a mano; si no, la coleccion se rompio y el gate dejo de probar."
        % corridos)
    print("%d clases, %d metodos de test, todos pasaron" % (len(clases), corridos))
    print()
    print("TODO OK")
