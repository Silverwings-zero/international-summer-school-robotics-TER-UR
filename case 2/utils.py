"""Shared helpers for the case 2 scripts: constants, robot physics, scripts.

- constants: joint count/names and the CSV column names.
- `UR10e`: numpy-only kinematics and dynamics (FK, Jacobian, gravity, mass
  matrix, Coriolis) usable as physics features for the gap model.
- URScript helpers: load a `.script`, read/replace its `vel`/`acc` parameters.
"""
from __future__ import annotations

import re

import numpy as np

# --- constants ---------------------------------------------------------------
N_JOINTS = 6                                             # base, shoulder, elbow, wrist1, wrist2, wrist3
JOINT_NAMES = ("base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3")

TIME_COL = "t"                                           # seconds since recording started
# The optimized motion parameters, from the URScript `movej(..., a=acc, v=vel)`,
# logged by record.py through these output float registers (raw script values,
# e.g. 100).
VEL_COL = "vel"                                          # output_double_register_1
ACC_COL = "acc"                                          # output_double_register_2
SCL_COL = "script_control_line"                          # URScript line running now
SCRIPT_COL = "script"                                    # source script of each row


# --- recording column helpers ------------------------------------------------
# The CSV stores each per-joint channel as six columns, e.g. actual_current0..5.
# These read/write a whole channel as one (n, N_JOINTS) block, addressed by the
# base name.

def joint_cols(base: str) -> list[str]:
    """Column names for one per-joint channel, e.g. ``actual_current0..5``."""
    return [f"{base}{j}" for j in range(N_JOINTS)]


def get_block(df, base: str) -> "np.ndarray":
    """Read a per-joint channel from a DataFrame as an ``(n, N_JOINTS)`` array."""
    return df[joint_cols(base)].to_numpy(dtype=float)


def set_block(df, base: str, block) -> None:
    """Overwrite a per-joint channel in a DataFrame with an ``(n, N_JOINTS)`` array."""
    df[joint_cols(base)] = np.asarray(block, dtype=float)


def frame_dt(df) -> float:
    """Median sample period (s) of a recording DataFrame (from its ``t`` column)."""
    return float(np.median(np.diff(df[TIME_COL].to_numpy(dtype=float))))


# --- UR10e kinematics and dynamics (numpy only) ------------------------------
# UR10e parameters, from Universal Robots "DH Parameters for calculations of
# kinematics and dynamics" (universal-robots.com). Standard (classic) DH.
#   joint i rotates by q[i] about z of the previous frame.
_A = np.array([0.0, -0.6127, -0.57155, 0.0, 0.0, 0.0])          # link length a [m]
_D = np.array([0.1807, 0.0, 0.0, 0.17415, 0.11985, 0.11655])    # link offset d [m]
_ALPHA = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0])  # twist [rad]

_MASS = np.array([7.369, 13.051, 3.989, 2.1, 1.98, 0.615])      # link masses [kg]

# Centre of mass of each link, in that link's DH frame [m].
_COM = np.array([
    [0.021, 0.000, 0.027],
    [0.380, 0.000, 0.158],
    [0.240, 0.000, 0.068],
    [0.000, 0.007, 0.018],
    [0.000, 0.007, 0.018],
    [0.000, 0.000, -0.026],
])

# Inertia tensor of each link about its centre of mass, in the link frame [kg m^2].
_INERTIA = np.array([
    [[0.0341, 0.0000, -0.0043], [0.0000, 0.0353, 0.0001], [-0.0043, 0.0001, 0.0216]],
    [[0.0281, 0.0001, -0.0156], [0.0001, 0.7707, 0.0000], [-0.0156, 0.0000, 0.7694]],
    [[0.0101, 0.0001, 0.0092], [0.0001, 0.3093, 0.0000], [0.0092, 0.0000, 0.3065]],
    [[0.0030, 0.0000, 0.0000], [0.0000, 0.0022, -0.0002], [0.0000, -0.0002, 0.0026]],
    [[0.0030, 0.0000, 0.0000], [0.0000, 0.0022, -0.0002], [0.0000, -0.0002, 0.0026]],
    [[0.0000, 0.0000, 0.0000], [0.0000, 0.0004, 0.0000], [0.0000, 0.0000, 0.0003]],
])

_G = 9.80665  # gravity [m/s^2]


class UR10e:
    """UR10e kinematics and dynamics. Extend by overriding the parameter arrays.

        ur = UR10e(payload=0.8)             # 0.8 kg at the tool flange
        q  = [0, -1.57, 1.57, -1.57, -1.57, 0]
        ur.fk(q)                            # 4x4 base -> flange pose
        ur.jacobian(q)                      # 6x6 geometric Jacobian (base frame)
        ur.gravity(q)                       # (6,) gravity torque per joint, Nm
        ur.mass_matrix(q)                   # 6x6 joint-space inertia, symmetric
        ur.coriolis(q, qd)                  # (6,) Coriolis + centrifugal torque

    `gravity(q)` and `diag(mass_matrix(q))` are cheap per-pose features for the
    gap model.
    """

    def __init__(self, payload: float = 0.0):
        """Args:
            payload: point mass at the tool flange (TCP) in kg. The recorded
                runs used the flange as TCP, so this is the tool mass, e.g. 0.8.
        """
        self.payload = float(payload)

    # --- kinematics -----------------------------------------------------------

    @staticmethod
    def _dh(theta: float, a: float, d: float, alpha: float) -> np.ndarray:
        """Standard DH homogeneous transform from one frame to the next."""
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array([
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ])

    def _frames(self, q) -> list[np.ndarray]:
        """Cumulative base->frame transforms T[0..6]; T[0] is the base (identity)."""
        q = np.asarray(q, dtype=float)
        frames = [np.eye(4)]
        for i in range(6):
            frames.append(frames[-1] @ self._dh(q[i], _A[i], _D[i], _ALPHA[i]))
        return frames

    def fk(self, q) -> np.ndarray:
        """Base->flange (TCP) pose as a 4x4 homogeneous transform."""
        return self._frames(q)[6]

    def tcp_position(self, q) -> np.ndarray:
        """Flange (TCP) position in the base frame, shape (3,)."""
        return self.fk(q)[:3, 3]

    def _point_jacobian(self, frames: list[np.ndarray], point: np.ndarray,
                        up_to: int) -> np.ndarray:
        """Linear + angular Jacobian (6x6) of a point rigidly on link ``up_to``.

        Only the first ``up_to`` joints move the point; later columns are zero.
        Columns use the classic revolute-joint form with axis z of the previous
        frame.
        """
        J = np.zeros((6, 6))
        for j in range(up_to):
            z = frames[j][:3, 2]           # joint j axis (z of previous frame)
            p = frames[j][:3, 3]           # origin of previous frame
            J[:3, j] = np.cross(z, point - p)
            J[3:, j] = z
        return J

    def jacobian(self, q) -> np.ndarray:
        """Geometric Jacobian (6x6) of the flange in the base frame.

        Rows 0..2 map joint rates to TCP linear velocity, rows 3..5 to angular
        velocity: ``[v; w] = J(q) @ qd``.
        """
        frames = self._frames(q)
        return self._point_jacobian(frames, frames[6][:3, 3], up_to=6)

    # --- dynamics -------------------------------------------------------------

    def _link_terms(self, q):
        """Per-link COM Jacobians and base-frame inertias, plus the payload.

        Returns a list of ``(mass, Jv, Jw, I_base)`` where Jv/Jw are the 3x6
        linear/angular Jacobians of the link COM and I_base is its inertia tensor
        rotated into the base frame. The payload is appended as a point mass at
        the flange (Jw and inertia zero).
        """
        frames = self._frames(q)
        terms = []
        for i in range(6):
            R = frames[i + 1][:3, :3]                      # base <- link frame
            com = frames[i + 1] @ np.append(_COM[i], 1.0)  # COM in base frame
            J = self._point_jacobian(frames, com[:3], up_to=i + 1)
            I_base = R @ _INERTIA[i] @ R.T
            terms.append((_MASS[i], J[:3], J[3:], I_base))
        if self.payload > 0.0:
            J = self._point_jacobian(frames, frames[6][:3, 3], up_to=6)
            terms.append((self.payload, J[:3], J[3:], np.zeros((3, 3))))
        return terms

    def gravity(self, q) -> np.ndarray:
        """Gravity torque per joint (6,), Nm: the torque to hold against gravity."""
        g_vec = np.array([0.0, 0.0, -_G])
        tau = np.zeros(6)
        for mass, Jv, _Jw, _I in self._link_terms(q):
            tau -= mass * (Jv.T @ g_vec)
        return tau

    def mass_matrix(self, q) -> np.ndarray:
        """Joint-space inertia matrix M(q) (6x6), symmetric positive definite."""
        M = np.zeros((6, 6))
        for mass, Jv, Jw, I_base in self._link_terms(q):
            M += mass * (Jv.T @ Jv) + Jw.T @ I_base @ Jw
        return 0.5 * (M + M.T)  # symmetrize away tiny numerical asymmetry

    def coriolis(self, q, qd) -> np.ndarray:
        """Coriolis and centrifugal torque per joint (6,), Nm: the term
        ``C(q,qd) @ qd`` in ``M(q) qdd + C(q,qd) qd + g(q) = tau``.

        Scales with velocity products (centrifugal ~ qd_i^2, Coriolis
        ~ qd_i qd_j). Built from the mass matrix via Christoffel symbols, with
        dM/dq by finite difference.
        """
        q = np.asarray(q, dtype=float)
        qd = np.asarray(qd, dtype=float)
        eps = 1e-6
        # dM[i] = dM/dq_i, a 6x6 matrix.
        dM = np.array([
            (self.mass_matrix(q + eps * e) - self.mass_matrix(q - eps * e)) / (2 * eps)
            for e in np.eye(6)
        ])
        c = np.zeros(6)
        for k in range(6):
            for i in range(6):
                for j in range(6):
                    christoffel = 0.5 * (dM[i, k, j] + dM[j, k, i] - dM[k, i, j])
                    c[k] += christoffel * qd[i] * qd[j]
        return c


# --- URScript helpers --------------------------------------------------------
# Matches `vel = 100` / `acc = 100` (int or float), with an optional `global`
# prefix that is preserved on replacement. Plain (non-global) assignments let
# send.py wrap the motion in a repeat loop, since URScript rejects a `global`
# declaration inside a loop.
_PARAM = r"((?:global\s+)?{name}\s*=\s*)([0-9]+(?:\.[0-9]+)?)"


def load_script(path: str) -> str:
    """Read a URScript file to text."""
    with open(path) as f:
        return f.read()


def get_param(text: str, name: str) -> float:
    """Read a `<name> = <number>` value from URScript text."""
    m = re.search(_PARAM.format(name=name), text)
    if not m:
        raise ValueError(f"no `{name} = ...` line in the script")
    return float(m.group(2))


def set_param(text: str, name: str, value: float) -> str:
    """Return the script with `<name>` set to ``value`` (rounded int)."""
    return re.sub(_PARAM.format(name=name), rf"\g<1>{int(round(value))}", text)
