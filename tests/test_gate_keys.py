"""Explicit state keys preserve repeated review observations."""

from dataclasses import fields

import matplotlib.pyplot as plt
import numpy as np
import pytest

import eyepiece.testing as ept


def _records(faults):
    return [tuple(getattr(f, field.name) for field in fields(f)) for f in faults]


def test_readability_key_reuses_pose_but_keeps_every_fault_time():
    fig = plt.figure(figsize=(2, 1), dpi=60)
    text = fig.text(0.95, 0.5, "over edge", fontsize=5)
    times = [0.0, 0.1, 0.2, 0.3, 0.4]
    posed = []

    def at(t):
        posed.append(t)
        text.set_x(0.95 if t < 0.3 else 0.94)

    try:
        expected = ept.readability_faults(fig, at=at, times=times)
        posed.clear()
        actual = ept.readability_faults(
            fig, at=at, times=times, frame_key=lambda t: t < 0.3
        )
        assert _records(actual) == _records(expected)
        assert posed == [0.0, 0.3]
        assert text.get_position()[0] == 0.94
    finally:
        plt.close(fig)


def test_motion_key_reuses_exact_geometry_and_pixels():
    fig, ax = plt.subplots(figsize=(2, 1), dpi=60)
    (line,) = ax.plot([0.2], [0.5], "o")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    times = np.arange(12) / 10
    posed = []

    def key(t):
        return int(t * 10) // 3

    def at(t):
        posed.append(t)
        line.set_xdata([0.2 + key(t) * 0.15])

    try:
        expected = ept.motion_faults(fig, at, times, rest=[line])
        posed.clear()
        actual = ept.motion_faults(fig, at, times, rest=[line], frame_key=key)
        assert _records(actual) == _records(expected)
        assert len(posed) == 4
        assert any(f.kind == "drift" for f in actual)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("kind", ["readability", "motion"])
def test_none_keys_disable_reuse_and_unhashable_keys_raise(kind):
    fig = plt.figure(figsize=(2, 1), dpi=60)
    posed = []
    times = [0.0, 0.1, 0.2]
    check = ept.readability_faults if kind == "readability" else ept.motion_faults
    try:
        check(fig, at=posed.append, times=times, frame_key=lambda t: None)
        assert posed == times
        with pytest.raises(TypeError, match="hashable"):
            check(fig, at=posed.append, times=times, frame_key=lambda t: [])
    finally:
        plt.close(fig)


@pytest.mark.parametrize("filtered", [False, True])
def test_layout_and_filters_keep_every_pose_even_with_equal_keys(filtered):
    fig, ax = plt.subplots(figsize=(3, 2), dpi=60)
    text = ax.text(0.5, 0.5, "same")
    if filtered:
        text.set_agg_filter(lambda image, dpi: (image, 0, 0))
    else:
        fig.set_layout_engine("constrained")
    times = [0.0, 0.1, 0.2]
    poses = []
    try:
        ept.readability_faults(fig, at=poses.append, times=times, frame_key=lambda t: 0)
        assert poses == times
    finally:
        plt.close(fig)
