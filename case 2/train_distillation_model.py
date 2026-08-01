"""Distilled model that predicts ``actual_*`` channels for a commanded trajectory.

Trained on recorded real runs, the model predicts what an actual channel would be
for a given commanded trajectory. Applied to the sim's targets (whose
``actual_*`` columns read 0.0), it overwrites those columns with predictions so
the metric and the RL agent can run without hardware.

``DistillModel`` is the interface the pipeline depends on:

    fit(recordings)  -> learn from recorded real runs
    predicts()       -> which actual_* channels this model fills in
    predict(df)      -> for a commanded-trajectory DataFrame, the predicted
                        actual channels, as {channel: (n, N_JOINTS) array}

The channels a model fills must be the ones the metric reads (see metrics.py).
``LinearModel`` below predicts ``actual_current``.

    from train_distillation_model import LinearModel, augment
    from analysis import Recording
    m = LinearModel().fit([Recording("data/test-4.csv"), Recording("data/test-6.csv")])
    m.predicts()                          # ['actual_current']
    m.save("models/distill.pkl")
    augment(m, "sim_to_real.csv")         # overwrite actual_current with predictions

Run as a script to train on the recorded runs, print held-out error, and save:

    python train_distillation_model.py --csvs data/test-*.csv --out models/distill.pkl

train_rla.py and run.py depend only on the interface, so a custom subclass of
DistillModel (or LinearModel) can replace the baseline via its pickle.
"""
from __future__ import annotations

import argparse
import glob
import pickle
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from preprocess import Identity, Preprocess, default_preprocess
from utils import (JOINT_NAMES, N_JOINTS, VEL_COL, ACC_COL, frame_dt,
                   get_block, set_block)


class DistillModel(ABC):
    """Interface every distilled model must implement.

    Subclasses implement ``fit``, ``predicts``, and ``predict``, and optionally
    override ``bounds``.
    """

    @abstractmethod
    def fit(self, recordings) -> "DistillModel":
        """Train on a list of ``analysis.Recording`` (real robot runs)."""

    @abstractmethod
    def predicts(self) -> list[str]:
        """Per-joint channel bases this model predicts, e.g. ``["actual_current"]``.

        These are the ``actual_*`` columns ``predict`` returns and ``augment``
        overwrites in the recording.
        """

    @abstractmethod
    def predict(self, df) -> dict:
        """Predicted actual channels for a commanded-trajectory DataFrame.

        ``df`` carries ``t``, ``target_q*``, ``target_qd*`` and the commanded
        ``vel``/``acc``. Return ``{base: (n, N_JOINTS) array}`` for every base in
        ``predicts()``. The caller overwrites those columns with the result.
        """

    def bounds(self):
        """Optional ``((vel_lo, vel_hi), (acc_lo, acc_hi))`` the model trusts.

        Return the raw-number range the training data covered so the optimizer
        stays in-distribution, or ``None`` to let the caller pick its own.
        """
        return None

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "DistillModel":
        with open(path, "rb") as f:
            return pickle.load(f)


class LinearModel(DistillModel):
    """Least-squares linear baseline that predicts the actual current.

    One linear model per row: for each joint the actual current is
    ``w . [target_current, qd, qdd, pos, vel, acc, joint one-hot]``, where
    ``target_current`` is the commanded current the joint is tracking, ``qd`` is
    the commanded velocity (from ``target_qd``), ``qdd`` its time derivative,
    ``pos`` the commanded angle, and ``vel``/``acc`` the raw movej numbers the
    script commanded. Fit against the measured ``actual_current`` of the real
    runs.

    The one-hot joint block gives each joint its own intercept with shared
    slopes; there is no separate bias term (it would be collinear with the
    one-hot). Being linear and smooth in its inputs, it cannot reproduce the ring
    after a stop (README "Why the optimizer stalls").

    Extension points:

    - Fit per joint or add joint-interaction terms; shared slopes leak one
      joint's behaviour onto another.
    - Normalize the features: ``pos`` (radians) and ``vel``/``acc`` (raw movej
      numbers up to ~1000) are on very different scales.
    - Add physics from ``utils.UR10e`` (gravity torque, mass matrix, Coriolis).
    - Use a non-linear regressor (MLP, trees) that can capture the ring.

    Override ``_row_features`` to change the inputs, or ``predicts``/``predict``
    to model a different channel.
    """

    FEATURE_NAMES = ["target_current", "qd", "qdd", "pos", "vel", "acc"] + \
        [f"is_{n}" for n in JOINT_NAMES]

    def __init__(self):
        self.coef = None                     # (12,) per-row weights
        self.vel_range = None                # (lo, hi) commanded vel seen in training
        self.acc_range = None                # (lo, hi) commanded acc seen in training

    def predicts(self) -> list[str]:
        return ["actual_current"]

    # --- features -------------------------------------------------------------

    def _row_features(self, joint: int, tgt_i, pos, qd, qdd, vel, acc) -> np.ndarray:
        """Feature rows for one joint over a whole trajectory, shape ``(n, 12)``.

        Every argument except ``joint`` is a length-``n`` array. ``tgt_i`` is the
        commanded ``target_current`` for this joint. Override to feed the model
        more inputs (gravity torque, mass, neighbouring joints); keep it a
        function of the commanded trajectory so it also applies to candidates.
        """
        tgt_i, pos = np.asarray(tgt_i), np.asarray(pos)
        qd, qdd = np.asarray(qd), np.asarray(qdd)
        vel, acc = np.asarray(vel), np.asarray(acc)
        onehot = np.zeros((len(pos), N_JOINTS))
        onehot[:, joint] = 1.0
        return np.column_stack([tgt_i, qd, qdd, pos, vel, acc, onehot])

    # --- fit ------------------------------------------------------------------

    def _design(self, recordings) -> tuple[np.ndarray, np.ndarray]:
        """Stack (features, measured actual_current) over every joint of every run.

        Shared by ``fit`` and the script's held-out error report, so both build
        the feature matrix the same way.
        """
        X, y = [], []
        for rec in recordings:
            if rec.vel_cmd is None or rec.acc_cmd is None:
                raise ValueError(f"{rec.path} has no vel/acc registers; record "
                                 "with `--float-register 1 vel 2 acc`")
            qdd = np.gradient(rec.target_qd, rec.dt, axis=0)     # commanded accel
            for j in range(N_JOINTS):
                X.append(self._row_features(j, rec.target_current[:, j], rec.target_q[:, j],
                                            rec.target_qd[:, j], qdd[:, j],
                                            rec.vel_cmd, rec.acc_cmd))
                y.append(rec.actual_current[:, j])
        return np.vstack(X), np.concatenate(y)

    def fit(self, recordings) -> "LinearModel":
        """Fit the row model on every row of every joint of every real run."""
        X, y = self._design(recordings)
        self.coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        # Feature columns 4 and 5 are vel and acc: their spans are the bounds.
        self.vel_range = (float(X[:, 4].min()), float(X[:, 4].max()))
        self.acc_range = (float(X[:, 5].min()), float(X[:, 5].max()))
        return self

    # --- predict --------------------------------------------------------------

    def predict(self, df) -> dict:
        dt = frame_dt(df)
        ti = get_block(df, "target_current")
        q = get_block(df, "target_q")
        qd = get_block(df, "target_qd")
        qdd = np.gradient(qd, dt, axis=0)
        vel = df[VEL_COL].to_numpy(dtype=float)
        acc = df[ACC_COL].to_numpy(dtype=float)
        out = np.zeros_like(q)
        for j in range(N_JOINTS):
            out[:, j] = self._row_features(j, ti[:, j], q[:, j], qd[:, j],
                                           qdd[:, j], vel, acc) @ self.coef
        return {"actual_current": out}

    def bounds(self):
        if self.coef is None:
            return None
        return self.vel_range, self.acc_range


def augment(model: DistillModel, csv: str, pre: Preprocess = None):
    """Overwrite a recording's actual_* columns with the model's predictions.

    Reads ``csv``, asks ``model.predict`` for the channels in
    ``model.predicts()``, writes them back into the same columns, and saves in
    place. The schema is unchanged; the actual_* columns become predictions,
    ready for metrics.py.

    ``pre`` (a preprocess.Preprocess) is applied around the model as in training:
    ``transform_distill`` before predict, ``revert_distill`` after, so the saved
    columns land back in real units. Defaults to a no-op ``Identity``.
    """
    pre = pre or Identity()
    df = pre.transform_distill(pd.read_csv(csv))
    preds = model.predict(df)
    for base in model.predicts():
        set_block(df, base, preds[base])
    df = pre.revert_distill(df)
    df.to_csv(csv, index=False)
    print(f"overwrote {model.predicts()} with predictions -> {csv}")
    return df


def main():
    ap = argparse.ArgumentParser(description="Train the distillation model.")
    ap.add_argument("--csvs", nargs="+", default=sorted(glob.glob("data/test-*.csv")),
                    help="recorded runs to train on")
    ap.add_argument("--out", default="models/distill.pkl", help="pickle path")
    ap.add_argument("--holdout", type=float, default=0.2,
                    help="fraction of rows held out for the error report")
    args = ap.parse_args()

    # Import under the real module name (not "__main__") so the saved pickle
    # loads cleanly in train_rla.py and run.py.
    from train_distillation_model import LinearModel
    from analysis import Recording

    # Preprocess the training data the same way the model will see it later.
    pre = default_preprocess()
    recordings = [Recording(r.path, df=pre.transform_distill(r.df))
                  for r in (Recording(p) for p in args.csvs)]
    model = LinearModel()

    # Build the row matrix once (same features as fit) to estimate held-out
    # error: predict the measured actual_current on held-out rows.
    X, y = model._design(recordings)
    print(f"{len(y)} rows from {len(recordings)} runs")

    # Deterministic split (no RNG): every 1/holdout-th row is a test row.
    step = max(int(round(1 / args.holdout)), 2)
    is_test = np.arange(len(y)) % step == 0
    coef, *_ = np.linalg.lstsq(X[~is_test], y[~is_test], rcond=None)
    err = X[is_test] @ coef - y[is_test]
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss = float(1 - np.sum(err ** 2) / np.sum((y[is_test] - y[is_test].mean()) ** 2))
    print(f"held-out ({is_test.sum()} rows): actual_current RMSE {rmse:.3f} A   R2 {ss:.3f}")
    print("coefficients:")
    for name, c in zip(LinearModel.FEATURE_NAMES, coef):
        print(f"  {name:12s} {c:+.4f}")

    # Refit on everything and save.
    model.fit(recordings)
    model.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
