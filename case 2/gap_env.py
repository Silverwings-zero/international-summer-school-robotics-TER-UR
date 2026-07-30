"""Gymnasium environment skeleton: optimize a move for speed vs vibration.

The agent chooses motion parameters (speed, acceleration) for a joint-space move.
The environment does NOT touch a real robot: it synthesizes the commanded
trajectory for those parameters, asks the learned :class:`GapModel` what the real
robot would actually do, and scores the result with the vibration metrics. That
is the whole point of Case 2, train against the learned gap, not the hardware.

This is a one-step (bandit-style) env: ``reset`` presents a move, ``step`` picks
the parameters for it and the episode ends. That is enough to learn a policy
"given this move, how fast can I go before it shakes".

WHAT IS DONE FOR YOU (provided)
    - observation space, action space
    - trajectory synthesis (trapezoidal profile) + gap-model rollout
    - peak-overshoot and RMS scoring via metrics.py

WHAT YOU BUILD (Gold tier, the core of the case)
    - ``_reward``: trade cycle time against vibration. This is where your
      engineering judgement goes. A stub is left that raises NotImplementedError
      with an example shape to start from.

Once ``_reward`` returns a number, train with Stable-Baselines3 (see __main__).
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from dataset import N_JOINTS, Run
from gap_model import GapModel
from metrics import run_metrics


# Action bounds: the parameter ranges the agent may choose from.
SPEED_MIN, SPEED_MAX = 0.2, 3.0      # rad/s
ACCEL_MIN, ACCEL_MAX = 0.5, 8.0      # rad/s^2

DT = 0.008          # control period (s), 125 Hz, matches typical RTDE logging
HOLD_S = 0.5        # seconds held at the goal, so there is a settling window
Q_LIMIT = 2 * np.pi  # joint travel bound used for the observation space


def synth_target(q_start: np.ndarray, q_goal: np.ndarray, speed: float,
                 accel: float, dt: float = DT, hold_s: float = HOLD_S) -> np.ndarray:
    """Commanded joint trajectory for a moveJ-style point-to-point move.

    All joints start and finish together (as UR's ``moveJ`` does); the timing is
    a trapezoidal velocity profile on the joint with the largest travel, then a
    hold at the goal so the settling window exists. Reference implementation,
    replace with UR's generator if one is provided.

    Returns:
        Target joint angles over time, shape ``(T, 6)`` (radians).
    """
    q_start = np.asarray(q_start, dtype=float)
    q_goal = np.asarray(q_goal, dtype=float)
    delta = q_goal - q_start
    dmax = float(np.max(np.abs(delta)))
    if dmax < 1e-9:
        return np.tile(q_goal, (2, 1))

    t_acc = speed / accel
    d_acc = 0.5 * accel * t_acc ** 2
    if 2 * d_acc >= dmax:                 # triangular profile (never reaches speed)
        t_acc = float(np.sqrt(dmax / accel))
        t_flat = 0.0
    else:                                 # trapezoidal profile
        t_flat = (dmax - 2 * d_acc) / speed
    T = 2 * t_acc + t_flat

    times = np.arange(0.0, T + hold_s, dt)
    dist = np.empty_like(times)
    for k, tt in enumerate(times):
        if tt < t_acc:                            # accelerate
            d = 0.5 * accel * tt ** 2
        elif tt < t_acc + t_flat:                 # cruise
            d = d_acc + speed * (tt - t_acc)
        elif tt < T:                              # decelerate
            td = tt - t_acc - t_flat
            d = d_acc + speed * t_flat + speed * td - 0.5 * accel * td ** 2
        else:                                     # hold at goal
            d = dmax
        dist[k] = min(d, dmax)
    frac = (dist / dmax)[:, None]                 # 0..1 along the path
    return q_start[None, :] + delta[None, :] * frac


class GapMotionEnv(gym.Env):
    """Pick (speed, accel) for a move; get scored on speed vs vibration."""

    metadata = {"render_modes": []}

    def __init__(self, gap_model: GapModel, moves: list[tuple] | None = None,
                 vib_weight: float = 1.0, seed: int | None = None):
        """Args:
            gap_model: A fitted :class:`GapModel`.
            moves: List of ``(q_start, q_goal)`` pairs to sample from. Defaults
                to a handful of single-joint moves.
            vib_weight: Convenience weight you may use inside ``_reward``.
        """
        super().__init__()
        self.gap_model = gap_model
        self.moves = moves if moves is not None else _default_moves()
        self.vib_weight = vib_weight
        self._rng = np.random.default_rng(seed)

        self.action_space = spaces.Box(
            low=np.array([SPEED_MIN, ACCEL_MIN], dtype=np.float32),
            high=np.array([SPEED_MAX, ACCEL_MAX], dtype=np.float32),
        )
        # Observation = the move to perform, as per-joint travel (goal - start).
        self.observation_space = spaces.Box(
            low=-Q_LIMIT, high=Q_LIMIT, shape=(N_JOINTS,), dtype=np.float32,
        )
        self._q_start = None
        self._q_goal = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._q_start, self._q_goal = self.moves[self._rng.integers(len(self.moves))]
        return self._obs(), {}

    def step(self, action):
        speed = float(np.clip(action[0], SPEED_MIN, SPEED_MAX))
        accel = float(np.clip(action[1], ACCEL_MIN, ACCEL_MAX))

        target_q = synth_target(self._q_start, self._q_goal, speed, accel)
        t = np.arange(target_q.shape[0]) * DT
        run = Run(
            run_id="synthetic", t=t, target_q=target_q,
            actual_q=np.zeros_like(target_q),
            params={"speed": speed, "accel": accel, "blend": 0.0, "payload": 0.0},
        )
        actual_q = self.gap_model.predict_run(run)      # what the real robot would do
        metrics = run_metrics(target_q, actual_q)
        cycle_time = float(t[-1])

        reward = self._reward(metrics, cycle_time)
        info = {"cycle_time": cycle_time, "speed": speed, "accel": accel, **metrics}
        return self._obs(), reward, True, False, info

    def _obs(self) -> np.ndarray:
        return (self._q_goal - self._q_start).astype(np.float32)

    def _reward(self, metrics: dict, cycle_time: float) -> float:
        """Score a move: fast is good, vibration is bad. YOU IMPLEMENT THIS.

        ``metrics`` has ``peak_max`` (worst overshoot, rad) and ``rms_mean`` (mean
        per-joint RMS, rad); ``cycle_time`` is the move duration in seconds.

        A reasonable starting shape (tune the weight, then improve it):

            return -(cycle_time + self.vib_weight * (
                metrics["peak_max"] + metrics["rms_mean"]))

        Ideas to go further: normalize the terms, use a speed *bonus* instead of a
        time penalty, cap acceptable vibration, or go multi-objective (Diamond).
        """
        raise NotImplementedError(
            "Implement GapMotionEnv._reward, this is the core of the Gold tier. "
            "See the docstring for a starting point."
        )


def _default_moves() -> list[tuple]:
    """A few single-joint moves from the home pose, for a first training run."""
    home = np.array([0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0])
    moves = []
    for j in range(N_JOINTS):
        goal = home.copy()
        goal[j] += np.pi / 2        # 90 deg on one joint
        moves.append((home.copy(), goal))
    return moves


if __name__ == "__main__":
    # Wiring check + training template (needs _reward implemented and a dataset).
    from dataset import load_runs

    model = GapModel().fit(load_runs())
    env = GapMotionEnv(model)
    obs, _ = env.reset()
    print("obs (per-joint travel, rad):", np.round(obs, 3))

    # from stable_baselines3 import PPO
    # agent = PPO("MlpPolicy", env, verbose=1)
    # agent.learn(total_timesteps=50_000)
    # compare agent.predict(obs) against fixed (speed, accel) baselines.
