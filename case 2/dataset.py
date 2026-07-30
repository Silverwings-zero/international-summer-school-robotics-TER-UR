"""Load recorded UR runs (target vs actual) from CSV.

Each CSV is ONE run: a single joint-space move recorded over time, with the
motion parameters (speed, acceleration, blend, payload) held constant for that
run. ``record.py`` writes exactly this schema; this module reads it back.

CSV schema (one row per RTDE sample, ~2-8 ms apart)
---------------------------------------------------
    t                       seconds since the move started
    target_q0 .. target_q5  commanded joint angles (radians), base..wrist3
    actual_q0 .. actual_q5  measured joint angles  (radians)
    speed                   move speed parameter used for this run
    accel                   move acceleration parameter used for this run
    blend                   blend radius used (0 for a plain point-to-point move)
    payload                 payload mass at the TCP (kg)

Optional extra signals (loaded into ``Run.extra`` if present): target_qd*,
actual_qd*, current*, tcp_*, ft_*. The two metrics only need target_q / actual_q.

The dataset is a folder of these CSVs (default ``data/``), one file per run,
covering a sweep of speed / acceleration / payload so a model can learn how they
change the gap. The dataset is delivered by UR (or recorded with ``record.py``).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


N_JOINTS = 6
JOINT_NAMES = ("base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3")

TIME_COL = "t"
TARGET_Q_COLS = [f"target_q{i}" for i in range(N_JOINTS)]
ACTUAL_Q_COLS = [f"actual_q{i}" for i in range(N_JOINTS)]
META_COLS = ["speed", "accel", "blend", "payload"]

# Signals we keep if the recording includes them, but the metrics do not require.
OPTIONAL_PREFIXES = ("target_qd", "actual_qd", "current", "tcp_", "ft_")

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

REQUIRED_COLS = [TIME_COL, *TARGET_Q_COLS, *ACTUAL_Q_COLS, *META_COLS]


@dataclass
class Run:
    """One recorded joint-space move.

    Attributes:
        run_id: Identifier (the CSV file name without extension).
        t: Sample times, shape ``(T,)`` seconds.
        target_q: Commanded joint angles, shape ``(T, 6)`` radians.
        actual_q: Measured joint angles, shape ``(T, 6)`` radians.
        params: ``{speed, accel, blend, payload}`` for this run (constants).
        extra: Any optional signals present, name -> array.
    """

    run_id: str
    t: np.ndarray
    target_q: np.ndarray
    actual_q: np.ndarray
    params: dict
    extra: dict = field(default_factory=dict)

    @property
    def dt(self) -> float:
        """Median sample period in seconds."""
        return float(np.median(np.diff(self.t))) if self.t.size > 1 else 0.0


def expected_columns() -> list[str]:
    """The required CSV columns, in order (see module docstring)."""
    return list(REQUIRED_COLS)


def load_run(path: str) -> Run:
    """Load one run CSV into a :class:`Run`.

    Raises:
        ValueError: if required columns are missing (with the offending names).
    """
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{os.path.basename(path)} is missing columns {missing}. "
            f"Expected schema: {REQUIRED_COLS}"
        )
    params = {k: float(df[k].iloc[0]) for k in META_COLS}
    extra = {
        c: df[c].to_numpy(dtype=float)
        for c in df.columns
        if c.startswith(OPTIONAL_PREFIXES)
    }
    return Run(
        run_id=os.path.splitext(os.path.basename(path))[0],
        t=df[TIME_COL].to_numpy(dtype=float),
        target_q=df[TARGET_Q_COLS].to_numpy(dtype=float),
        actual_q=df[ACTUAL_Q_COLS].to_numpy(dtype=float),
        params=params,
        extra=extra,
    )


def load_runs(data_dir: str = DEFAULT_DATA_DIR) -> list[Run]:
    """Load every ``*.csv`` run in ``data_dir``, sorted by file name.

    Raises:
        FileNotFoundError: if the folder is absent or holds no CSVs, with a hint
            to run ``record.py`` or drop in the dataset UR provides.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"No data folder at {data_dir}. Put the UR-provided run CSVs there, "
            f"or record your own with record.py."
        )
    paths = sorted(
        os.path.join(data_dir, f) for f in os.listdir(data_dir)
        if f.lower().endswith(".csv")
    )
    if not paths:
        raise FileNotFoundError(
            f"{data_dir} has no CSV runs yet. See record.py and the schema in "
            f"dataset.py."
        )
    return [load_run(p) for p in paths]
