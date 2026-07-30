"""Vibration metrics for the reality-gap case.

The reality gap shows up at the *end* of a move: the real robot overshoots the
commanded target and rings down before it settles. These functions turn that
wobble into single numbers you can plot (Bronze) and later optimize (Gold).

Two metrics are provided (both defined over the *settling window*, the stretch
after the command has reached its final value):

    peak overshoot : max | actual(t) - target |   -> worst single spike
    rms error      : sqrt(mean((actual(t)-target)^2)) -> sustained wobble

``jerk_rms`` is optional reference code (needs SciPy) for a smoothness term.

All functions take plain 1-D NumPy arrays for one joint. Use ``run_metrics`` to
get the per-joint numbers for a whole recorded run at once.
"""
from __future__ import annotations

import numpy as np


# A move "arrives" when the commanded target stops changing (within EPS, in
# radians). Everything after that is the settling window we score.
SETTLE_EPS = 1e-4


def settling_mask(target: np.ndarray, eps: float = SETTLE_EPS) -> np.ndarray:
    """Boolean mask selecting the settling window of one joint.

    The window starts at the first sample from which ``target`` stays within
    ``eps`` of its final commanded value, and runs to the end of the recording.

    Args:
        target: Commanded joint angle over time, shape ``(T,)`` (radians).
        eps: Tolerance (radians) for "target has reached its final value".

    Returns:
        Boolean array, shape ``(T,)``; ``True`` inside the settling window.
    """
    target = np.asarray(target, dtype=float)
    final = target[-1]
    moving = np.abs(target - final) > eps
    moving_idx = np.where(moving)[0]
    arrival = 0 if moving_idx.size == 0 else int(moving_idx[-1]) + 1
    mask = np.zeros_like(target, dtype=bool)
    mask[min(arrival, target.size - 1):] = True
    return mask


def peak_overshoot(actual: np.ndarray, target: np.ndarray,
                   mask: np.ndarray | None = None) -> float:
    """Largest deviation from target within the settling window (radians)."""
    actual = np.asarray(actual, dtype=float)
    target = np.asarray(target, dtype=float)
    if mask is None:
        mask = settling_mask(target)
    if not mask.any():
        return 0.0
    return float(np.max(np.abs(actual[mask] - target[mask])))


def rms_error(actual: np.ndarray, target: np.ndarray,
              mask: np.ndarray | None = None) -> float:
    """Root-mean-square position error within the settling window (radians)."""
    actual = np.asarray(actual, dtype=float)
    target = np.asarray(target, dtype=float)
    if mask is None:
        mask = settling_mask(target)
    if not mask.any():
        return 0.0
    err = actual[mask] - target[mask]
    return float(np.sqrt(np.mean(err ** 2)))


def jerk_rms(actual: np.ndarray, dt: float, *, cutoff_hz: float = 10.0,
             order: int = 2) -> float:
    """RMS jerk (rad/s^3) of a joint trajectory, optional smoothness metric.

    Jerk is the third time-derivative of position and is noisy, so the signal is
    low-pass filtered (Butterworth) before differencing. Requires SciPy; if it is
    not installed, raises ``ImportError`` with a hint (the metric is optional).

    Args:
        actual: Measured joint angle over time, shape ``(T,)`` (radians).
        dt: Sample period in seconds.
        cutoff_hz: Low-pass cutoff for the Butterworth filter.
        order: Butterworth filter order.
    """
    try:
        from scipy.signal import butter, filtfilt
    except ImportError as exc:  # optional dependency
        raise ImportError(
            "jerk_rms needs SciPy (pip install scipy). The jerk metric is "
            "optional; peak_overshoot and rms_error do not need it."
        ) from exc
    actual = np.asarray(actual, dtype=float)
    fs = 1.0 / dt
    b, a = butter(order, cutoff_hz / (0.5 * fs), btype="low")
    smooth = filtfilt(b, a, actual)
    jerk = np.gradient(np.gradient(np.gradient(smooth, dt), dt), dt)
    return float(np.sqrt(np.mean(jerk ** 2)))


def run_metrics(target_q: np.ndarray, actual_q: np.ndarray) -> dict:
    """Per-joint peak overshoot and RMS error for a whole run.

    Args:
        target_q: Commanded joint angles, shape ``(T, 6)`` (radians).
        actual_q: Measured joint angles, shape ``(T, 6)`` (radians).

    Returns:
        Dict with ``peak`` and ``rms`` lists (one value per joint) and their
        max/mean summaries, all in radians.
    """
    target_q = np.asarray(target_q, dtype=float)
    actual_q = np.asarray(actual_q, dtype=float)
    n_joints = target_q.shape[1]
    peaks, rmss = [], []
    for j in range(n_joints):
        mask = settling_mask(target_q[:, j])
        peaks.append(peak_overshoot(actual_q[:, j], target_q[:, j], mask))
        rmss.append(rms_error(actual_q[:, j], target_q[:, j], mask))
    return {
        "peak": peaks,
        "rms": rmss,
        "peak_max": max(peaks),
        "rms_mean": float(np.mean(rmss)),
    }
