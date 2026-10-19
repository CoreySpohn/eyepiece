"""Seeded Poisson arrivals replay, never decrease, and average to rate x exposure."""

import numpy as np
import pytest

import eyepiece as ep


def _clock():
    return np.array([0.0, 0.5, 1.0, 2.0, 2.0, 10.0])


def test_same_seed_replays_and_another_seed_differs():
    rate = np.full((40, 40), 2.0)
    a = ep.arrivals(rate, _clock(), seed=7)
    b = ep.arrivals(rate, _clock(), seed=7)
    c = ep.arrivals(rate, _clock(), seed=8)
    np.testing.assert_array_equal(a.counts, b.counts)
    assert not np.array_equal(a.counts, c.counts)
    assert a.seed == 7
    assert a.counts.dtype.kind == "i"
    assert a.counts.shape == (6, 40, 40)


def test_counts_never_decrease_and_start_empty_at_zero_exposure():
    rate = np.random.default_rng(0).uniform(0.0, 50.0, (30, 30))
    a = ep.arrivals(rate, _clock(), seed=3)
    assert np.all(np.diff(a.counts, axis=0) >= 0)
    assert np.all(a.counts[0] == 0)
    np.testing.assert_array_equal(a.counts[3], a.counts[4])  # a held time
    for k in range(a.n):
        np.testing.assert_array_equal(
            a.new(k), a.counts[k] - (a.counts[k - 1] if k else 0)
        )
    np.testing.assert_array_equal(a.new(-1), a.counts[-1] - a.counts[-2])


def test_final_frame_is_the_still_at_any_number_of_samples():
    rate = np.random.default_rng(1).uniform(0.0, 1e3, (16, 16))
    still = ep.arrivals(rate, [100.0], seed=11).counts[-1]
    for clock in (np.geomspace(1.0, 100.0, 7), np.geomspace(1e-3, 100.0, 61)):
        np.testing.assert_array_equal(
            ep.arrivals(rate, clock, seed=11).counts[-1], still
        )


@pytest.mark.parametrize("rate", [0.3, 3.0, 4e3])
def test_counts_and_increments_are_poisson_with_mean_rate_times_exposure(rate):
    n_pix = 200 * 200
    a = ep.arrivals(np.full((200, 200), rate), _clock(), seed=5)
    for k in range(1, a.n):
        dt = a.exposure[k] - a.exposure[k - 1]
        for got, lam in ((a.counts[k], rate * a.exposure[k]), (a.new(k), rate * dt)):
            if lam == 0.0:
                assert np.all(got == 0)
                continue
            assert abs(got.mean() - lam) < 5.0 * np.sqrt(lam / n_pix)
            # Poisson: the variance equals the mean (sd of a sample variance
            # is about lam * sqrt(2 / n) for large lam).
            assert abs(got.var() / lam - 1.0) < 5.0 * np.sqrt(
                2.0 / n_pix + 1.0 / (lam * n_pix)
            )
    np.testing.assert_allclose(a.expected(2), rate * 1.0)


def test_increments_are_uncorrelated():
    a = ep.arrivals(np.full((300, 300), 5.0), [1.0, 2.0, 3.0], seed=2)
    first, second = a.new(1).ravel(), a.new(2).ravel()
    r = np.corrcoef(first, second)[0, 1]
    assert abs(r) < 5.0 / np.sqrt(first.size)


def test_large_rates_stay_exact_and_monotone():
    rate = np.array([[1e9, 1e3], [1e-2, 0.0]])
    a = ep.arrivals(rate, np.geomspace(1.0, 1e5, 40), seed=9)
    assert np.all(np.diff(a.counts, axis=0) >= 0)
    np.testing.assert_allclose(a.counts[-1, 0, 0], 1e14, rtol=1e-6)
    assert np.all(a.counts[:, 1, 1] == 0)


def test_rejects_bad_inputs():
    with pytest.raises(ValueError, match="rate"):
        ep.arrivals([[-1.0]], [1.0], seed=0)
    with pytest.raises(ValueError, match="rate"):
        ep.arrivals([[np.nan]], [1.0], seed=0)
    with pytest.raises(ValueError, match="non-decreasing"):
        ep.arrivals([[1.0]], [2.0, 1.0], seed=0)
    with pytest.raises(ValueError, match="1D"):
        ep.arrivals([[1.0]], [], seed=0)
    with pytest.raises(ValueError, match="non-negative"):
        ep.arrivals([[1.0]], [-1.0, 1.0], seed=0)
    with pytest.raises(TypeError, match="seed"):
        ep.arrivals([[1.0]], [1.0], seed=None)
    with pytest.raises(TypeError, match="seed"):
        ep.arrivals([[1.0]], [1.0], seed=1.5)
    with pytest.raises(ValueError, match="seed"):
        ep.arrivals([[1.0]], [1.0], seed=-1)
