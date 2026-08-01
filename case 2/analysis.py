"""Per-joint stats and a plot for a recorded UR run.

Load a CSV from ``record.py``, print per-joint numbers (range of motion, current
gap, position lag), and plot one joint's target vs actual current.

    python analysis.py --csv data/test-4.csv --joint 1

``--joint`` selects the joint (0=base ... 5=wrist3).

``Recording`` is the shared CSV data loader used across the pipeline
(``train_distillation_model.py``, ``train_rla.py``); it wraps a run's CSV as
numpy arrays.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from utils import ACC_COL, JOINT_NAMES, N_JOINTS, SCL_COL, SCRIPT_COL, TIME_COL, VEL_COL


class Recording:
    """One recorded run loaded from a ``record.py`` CSV, as numpy arrays."""

    def __init__(self, path: str, df=None):
        # Loads ``path``, or wraps an already-loaded table if ``df`` is given
        # (e.g. one a preprocess.Preprocess transformed).
        if df is None:
            df = pd.read_csv(path)
        self.path = path
        self.df = df             # raw table; callers can slice rows by index
        self.t = df[TIME_COL].to_numpy(dtype=float)
        self.target_q = np.column_stack([df[f"target_q{j}"] for j in range(6)])
        self.actual_q = np.column_stack([df[f"actual_q{j}"] for j in range(6)])
        self.target_qd = np.column_stack([df[f"target_qd{j}"] for j in range(6)])
        self.target_current = np.column_stack([df[f"target_current{j}"] for j in range(6)])
        self.actual_current = np.column_stack([df[f"actual_current{j}"] for j in range(6)])
        # Commanded movej parameters, if record.py logged them (raw URScript
        # values, e.g. 100). None when the run has no vel/acc registers.
        self.vel_cmd = df[VEL_COL].to_numpy(dtype=float) if VEL_COL in df else None
        self.acc_cmd = df[ACC_COL].to_numpy(dtype=float) if ACC_COL in df else None
        # URScript line currently executing (RTDE): nonzero while a movej runs, 0
        # during a sleep/dwell. The value is script-specific and not comparable
        # across scripts, but a change marks a move boundary. Zeros if not logged.
        self.scl = (df[SCL_COL].to_numpy() if SCL_COL in df
                    else np.zeros(len(self.t), dtype=int))
        # Source script of each row, tagged when several runs are pooled into one
        # file, so segmentation never spans two scripts. One value if not tagged.
        self.script = (df[SCRIPT_COL].to_numpy() if SCRIPT_COL in df
                       else np.zeros(len(self.t), dtype=int))

    @property
    def dt(self) -> float:
        """Median sample period (s)."""
        return float(np.median(np.diff(self.t)))

    def current_gap(self, joint: int) -> np.ndarray:
        """Actual minus target current for one joint (A), per row."""
        return self.actual_current[:, joint] - self.target_current[:, joint]

    def plot(self, joint: int = 1):
        """Plot one joint's target vs actual current, and the current gap below.

        Top: the two current traces over the run. Bottom: actual minus target
        current.
        """
        import matplotlib.pyplot as plt

        name = JOINT_NAMES[joint]
        gap = self.current_gap(joint)
        fig, (ax_c, ax_g) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

        ax_c.plot(self.t, self.target_current[:, joint], label="target current", lw=2)
        ax_c.plot(self.t, self.actual_current[:, joint], label="actual current", lw=1)
        ax_c.set_ylabel("current (A)")
        ax_c.set_title(f"{name} joint")
        ax_c.legend(loc="best")

        ax_g.plot(self.t, gap, color="tab:red", lw=1)
        ax_g.axhline(0, color="grey", lw=0.8)
        ax_g.set_ylabel("actual - target (A)")
        ax_g.set_xlabel("time (s)")

        fig.tight_layout()
        return fig


def main():
    # Print per-joint stats, then plot one joint.
    ap = argparse.ArgumentParser(description="Per-joint stats and a plot for a recorded run.")
    ap.add_argument("--csv", default="data/test-4.csv", help="recorded run CSV")
    ap.add_argument("--joint", type=int, default=1,
                    help="joint index 0..5 to plot (default 1 = shoulder)")
    ap.add_argument("--no-plot", action="store_true", help="print stats only")
    args = ap.parse_args()

    rec = Recording(args.csv)
    print(f"{args.csv}  ({len(rec.t)} rows, dt {rec.dt*1e3:.1f} ms)")

    # Per-joint summary: range of motion, current gap (RMS and max magnitude),
    # and RMS position lag (actual minus target).
    print(f"{'joint':10s} {'moved':>9s} {'gap RMS':>9s} {'gap max':>9s} {'pos err':>10s}")
    for j in range(N_JOINTS):
        moved = rec.target_q[:, j].max() - rec.target_q[:, j].min()   # range of motion (rad)
        gap = rec.current_gap(j)
        gap_rms = float(np.sqrt(np.mean(gap ** 2)))
        gap_max = float(np.max(np.abs(gap)))
        pos_err = float(np.sqrt(np.mean((rec.actual_q[:, j] - rec.target_q[:, j]) ** 2)))
        print(f"{JOINT_NAMES[j]:10s} {moved:8.3f}r {gap_rms:8.3f}A {gap_max:8.3f}A "
              f"{pos_err*1e3:7.2f}mrad")

    if not args.no_plot:
        import matplotlib.pyplot as plt
        rec.plot(args.joint)
        plt.show()


if __name__ == "__main__":
    main()
