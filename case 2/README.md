# Case 2: Reality Gap and Motion Optimization

URSim shows a perfect robot: commanded angle equals measured angle, always. A
real UR does not. It overshoots the target, rings down at the end of a move,
drifts with temperature, sags under load, and each joint behaves a little
differently. That difference is the **reality gap**.

## The task

Work the gap in three layers:

1. **Analyze** it from recorded data (target vs actual joint angles).
2. **Learn** a model of it: `actual = f(target, joint config, velocity, accel, payload)`.
3. **Optimize** motion with reinforcement learning, move as fast as possible with
   as little vibration as possible, trained against the *learned gap*, not the
   hardware.

The robot seam and the metrics are done; your work is the model and the RL reward.

## What's provided vs what you build

| File | Role | Provided? |
|------|------|-----------|
| `metrics.py` | peak overshoot, RMS error (settling window), optional jerk | done, reference |
| `dataset.py` | CSV schema + loader into `Run` objects | done |
| `gap_model.py` | baseline gap model, per-timestep linear regression | baseline provided |
| `gap_env.py` | Gymnasium env: spaces, trajectory synthesis, rollout, scoring | env provided; `_reward` is a stub to implement |
| `record.py` | pure-socket recorder (URScript move + RTDE stream) in the dataset schema | provided, run once to make data |

The graded work is in the Tiers below. Any scikit-learn estimator drops into
`GapModel(...)` unchanged, and `GapMotionEnv._reward` is the one stub that raises
`NotImplementedError` until you fill it.

## Setup

1. **Python deps installed** in this folder:
   ```bash
   pip install -r requirements.txt
   ```
   `scipy` (optional jerk metric) and `stable-baselines3` (RL) are only needed for
   the later tiers.
2. **The dataset in `data/`.** One CSV per run (one point-to-point move at fixed
   speed / accel / payload); full schema at the top of `dataset.py`. Drop the
   UR-provided run CSVs into `data/`, or record your own:
   ```bash
   python record.py --robot-ip 192.168.1.10 --out data
   ```
   `record.py` needs a live robot (or URSim, where the gap is near zero); it is
   run once to build the dataset, not during the exercise. Until `data/` holds
   CSVs, the loaders raise a clear error telling you where to put them.

Load runs in code with:
```python
from dataset import load_runs
runs = load_runs()          # reads data/*.csv
```

## Run

```bash
python gap_model.py     # fit baseline, print held-out per-joint RMS
python gap_env.py       # wiring check (fails until _reward is implemented)
```
`metrics.py` is reference code imported by the others. All three need `data/`
populated first.

## Tiers

- **Bronze, analyze it:** load the dataset, plot target vs actual across speeds,
  compute peak overshoot and RMS per run.
- **Silver, predict it:** beat the linear baseline gap model; report held-out
  per-joint RMS (`GapModel.per_joint_rms`) on runs the model never saw.
- **Gold, optimize it:** implement `GapMotionEnv._reward` (trade cycle time
  against vibration), train with Stable-Baselines3, and beat a fixed-parameter
  baseline on speed vs vibration.
- **Diamond, push it:** go further, multi-objective or curriculum RL, per-joint
  models, extra features, or transfer the learned policy back to a real robot.
