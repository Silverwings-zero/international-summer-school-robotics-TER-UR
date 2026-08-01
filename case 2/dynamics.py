"""Commanded torque and current for a candidate motion, without re-running URSim.

For a move (fixed joint geometry) and a candidate speed profile, produces the
commanded trajectory (q, qd, qdd) and the commanded joint current, from the UR10e
inverse dynamics:

    tau     = M(q) qdd + g(q)          (utils.UR10e; Coriolis dropped by default)
    current = tau / Kt                 (Kt fit from the recording: moment / current)

Units: the ``vel``/``acc`` registers (and the URScript numbers) are deg/s and
deg/s^2; joint speeds in the recording are rad/s. The trajectory is built in rad,
with the speed clamped to the joint limit (URScript clamps a movej speed above the
limit).

``Dynamics`` is an interface; subclass it for a different torque model (friction,
Coriolis, identified inertial parameters, a learned model). The default is fast
because the pose-dependent terms are precomputed along each move's geometry once
(they do not depend on the speed profile).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from utils import (ACC_COL, N_JOINTS, TIME_COL, VEL_COL, UR10e, get_block,
                   joint_cols)

DEG2RAD = np.pi / 180.0
MAX_JOINT_SPEED = np.pi          # rad/s: URScript clamps a movej speed to this
MAX_JOINT_ACC = 4.0 * np.pi      # rad/s^2: a generous joint acceleration ceiling
GRID = 50                        # samples along a move's geometry for the M,g cache


def fit_kt(rec) -> np.ndarray:
    """Per-joint torque-to-current constant ``Kt`` from a recording.

    ``target_moment = Kt * target_current`` per joint (motor and gearing). Fit the
    slope through the origin from the recorded pair, so torque can be converted to
    current in the same units the recording uses. Joints with no current in the
    run fall back to 1.0 (they carry no information to fit).
    """
    moment = get_block(rec.df, "target_moment")
    current = get_block(rec.df, "target_current")
    num = (moment * current).sum(axis=0)
    den = (current * current).sum(axis=0)
    return np.where(den > 1e-9, num / np.where(den > 1e-9, den, 1.0), 1.0)


def trapezoidal(distance: float, vel: float, acc: float, dt: float) -> np.ndarray:
    """Progress ``s(t)`` in [0,1] for a joint move of ``distance`` (rad).

    Standard trapezoidal profile at peak speed ``vel`` (rad/s) and acceleration
    ``acc`` (rad/s^2), sampled at ``dt`` (s); triangular if it never reaches
    ``vel``. All joints of a movej share this one profile (they start and stop
    together), so ``q(t) = start + s(t) * (dest - start)``.
    """
    if distance <= 1e-9 or vel <= 0 or acc <= 0:
        return np.array([0.0, 1.0])                  # degenerate: no motion
    t_acc = vel / acc
    d_acc = 0.5 * acc * t_acc ** 2
    if 2 * d_acc >= distance:                        # triangular: never reaches vel
        t_acc = np.sqrt(distance / acc)
        d_acc = 0.5 * distance
        t_flat = 0.0
    else:
        t_flat = (distance - 2 * d_acc) / vel
    total = 2 * t_acc + t_flat
    t = np.arange(0.0, total + dt, dt)
    d = np.where(t < t_acc, 0.5 * acc * t ** 2,
                 np.where(t < t_acc + t_flat, d_acc + vel * (t - t_acc),
                          distance - 0.5 * acc * np.clip(total - t, 0, None) ** 2))
    return np.clip(d, 0.0, distance) / distance


class Dynamics(ABC):
    """Interface: commanded joint current for a candidate motion.

    ``current(q, qd, qdd)`` is the one thing a torque model must provide. The rest
    (differentiation, assembling a recording-like frame) is handled for you, so a
    student overrides just the physics.
    """

    @abstractmethod
    def current(self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray,
                s: np.ndarray = None, key=None) -> np.ndarray:
        """Commanded joint current, ``(n, N_JOINTS)``, for a trajectory.

        ``q``/``qd``/``qdd`` are ``(n, N_JOINTS)`` in rad. ``s`` is each row's
        progress along the move geometry in [0,1]; ``key`` identifies the move.
        Both let an implementation reuse pose-dependent terms cached per move.
        """

    def frame(self, q: np.ndarray, dt: float, vel_deg: float, acc_deg: float,
              s: np.ndarray = None, key=None) -> pd.DataFrame:
        """Assemble a commanded-trajectory DataFrame for a candidate ``q(t)``.

        Differentiates ``q`` for ``qd``/``qdd``, asks ``current`` for the commanded
        current, and lays it out with the same columns a URSim recording has (the
        ``actual_*`` columns are zeros for the distill model to overwrite). ``dt``
        is the sample period; ``vel_deg``/``acc_deg`` are stored as the register
        columns (still deg/s). ``s`` is each row's progress along the move geometry
        in [0,1] (how the cached pose terms are looked up).
        """
        q = np.asarray(q, dtype=float)
        qd = np.gradient(q, dt, axis=0)
        qdd = np.gradient(qd, dt, axis=0)
        cur = self.current(q, qd, qdd, s=s, key=key)
        n = len(q)
        out = {TIME_COL: np.arange(n) * dt}
        for j in range(N_JOINTS):
            out[f"target_q{j}"] = q[:, j]
            out[f"target_qd{j}"] = qd[:, j]
            out[f"target_current{j}"] = cur[:, j]
            out[f"actual_current{j}"] = 0.0          # distill overwrites these
        out[VEL_COL] = float(vel_deg)
        out[ACC_COL] = float(acc_deg)
        return pd.DataFrame(out)


class UR10eDynamics(Dynamics):
    """Default torque model: ``tau = M(q) qdd + g(q)`` via ``utils.UR10e``.

    Fast because the pose-dependent terms ``M(q)`` and ``g(q)`` are precomputed on
    a grid along each move's geometry (they do not depend on the speed profile) and
    interpolated per candidate. Coriolis is dropped (small here, and its finite
    difference is expensive); add it in a subclass if you need it.
    """

    def __init__(self, rec, payload: float = 0.0, grid: int = GRID):
        self.ur = UR10e(payload=payload)
        self.kt = fit_kt(rec)
        self.grid = grid
        self._cache = {}          # key -> (s_grid, M_grid, g_grid)

    def prepare(self, key, q_geo: np.ndarray):
        """Precompute ``M(q)``, ``g(q)`` along a move's geometry ``q_geo`` (G, 6).

        Call once per move. ``q_geo`` is the joint path the move traces (a straight
        line for movej, the recorded curve for a re-timed path), sampled uniformly.
        Every candidate of that move reuses the result.
        """
        q_geo = np.asarray(q_geo, dtype=float)
        idx = np.linspace(0, len(q_geo) - 1, min(self.grid, len(q_geo))).round().astype(int)
        s_grid = idx / max(len(q_geo) - 1, 1)
        M_grid = np.array([self.ur.mass_matrix(q_geo[i]) for i in idx])
        g_grid = np.array([self.ur.gravity(q_geo[i]) for i in idx])
        self._cache[key] = (s_grid, M_grid, g_grid, q_geo)
        return self

    def current(self, q, qd, qdd, s=None, key=None) -> np.ndarray:
        q, qd, qdd = np.asarray(q), np.asarray(qd), np.asarray(qdd)
        if key is not None and key in self._cache and s is not None:
            # Interpolate the cached pose terms at each row's progress along the move.
            s_grid, M_grid, g_grid, _ = self._cache[key]
            s = np.asarray(s, dtype=float)
            M = np.empty((len(q), N_JOINTS, N_JOINTS))
            for a in range(N_JOINTS):
                for b in range(N_JOINTS):
                    M[:, a, b] = np.interp(s, s_grid, M_grid[:, a, b])
            g = np.column_stack([np.interp(s, s_grid, g_grid[:, a]) for a in range(N_JOINTS)])
        else:
            # No cache: compute per row (correct but slow; fine for a few candidates).
            M = np.array([self.ur.mass_matrix(row) for row in q])
            g = np.array([self.ur.gravity(row) for row in q])
        tau = np.einsum("nij,nj->ni", M, qdd) + g
        return tau / self.kt


def default_dynamics(rec) -> Dynamics:
    """The dynamics model the pipeline uses. Swap the return to change it globally."""
    return UR10eDynamics(rec)
