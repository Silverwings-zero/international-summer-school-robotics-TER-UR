"""Learn the reality gap: predict the real robot's actual motion.

    actual_q(t)  =  f( target_q(t), target_qd(t), speed, accel, payload )

URSim can never show you this deviation (there, actual == target). A model
trained on real recordings can. This module provides the *baseline*: a plain
linear regression, per timestep, mapping the command plus motion parameters to
the measured joint angles.

The baseline is deliberately simple, it is the bar to beat (Silver tier). Pass a
richer scikit-learn estimator to :class:`GapModel` and you have a better gap
model with no other code changes, for example::

    from sklearn.neural_network import MLPRegressor
    model = GapModel(MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500))

or a random forest, gradient boosting, polynomial features in a Pipeline, a small
PyTorch net, etc. Evaluate on held-out runs with :meth:`per_joint_rms`.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

from dataset import N_JOINTS, Run


def _run_features(run: Run) -> np.ndarray:
    """Per-timestep feature matrix for one run, shape ``(T, 3*N_JOINTS + 3)``.

    Features per sample: commanded angle and commanded velocity for each joint
    (velocity from the target by finite difference), the joint configuration
    (the target angles again, as the model may depend on pose), and the three
    scalar motion parameters broadcast across time (speed, accel, payload).
    """
    t, target_q = run.t, run.target_q
    target_qd = np.gradient(target_q, t, axis=0)  # commanded velocity
    n = target_q.shape[0]
    params = np.tile(
        [run.params["speed"], run.params["accel"], run.params["payload"]],
        (n, 1),
    )
    return np.hstack([target_q, target_qd, target_q, params])


class GapModel:
    """Predict actual joint angles from the command and motion parameters."""

    def __init__(self, estimator=None):
        """Args:
            estimator: Any scikit-learn regressor. Defaults to the linear
                baseline. Single-output regressors are wrapped for the 6 joints.
        """
        base = estimator if estimator is not None else LinearRegression()
        # LinearRegression handles multi-output natively; wrap others so any
        # estimator works unchanged.
        self.model = base if _is_multioutput(base) else MultiOutputRegressor(base)

    def fit(self, runs: list[Run]) -> "GapModel":
        """Fit on a list of runs (all timesteps pooled)."""
        X = np.vstack([_run_features(r) for r in runs])
        y = np.vstack([r.actual_q for r in runs])
        self.model.fit(X, y)
        return self

    def predict_run(self, run: Run) -> np.ndarray:
        """Predicted actual joint angles for one run, shape ``(T, 6)``."""
        return self.model.predict(_run_features(run))

    def per_joint_rms(self, runs: list[Run]) -> np.ndarray:
        """Prediction RMS error per joint over ``runs`` (radians), shape ``(6,)``.

        This is the score to report on *held-out* runs and to drive down when
        you beat the baseline.
        """
        sq = np.zeros(N_JOINTS)
        count = 0
        for r in runs:
            err = self.predict_run(r) - r.actual_q
            sq += np.sum(err ** 2, axis=0)
            count += err.shape[0]
        return np.sqrt(sq / max(count, 1))


def _is_multioutput(estimator) -> bool:
    """True if the estimator natively supports multi-output regression."""
    return isinstance(estimator, LinearRegression)


if __name__ == "__main__":
    # Quick baseline demo: fit on all-but-last run, score the held-out one.
    from dataset import load_runs

    runs = load_runs()
    train, test = runs[:-1], runs[-1:]
    model = GapModel().fit(train)
    rms = model.per_joint_rms(test)
    print("baseline per-joint RMS (rad):",
          ", ".join(f"{v:.4f}" for v in rms))
    print("beat this with a richer estimator, see the module docstring.")
