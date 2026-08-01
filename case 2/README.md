# Case 2: Reality Gap and Motion Optimization

URSim shows a perfect robot: commanded angle equals measured angle. A real UR
overshoots and rings down at the end of a move, and each joint behaves a little
differently. That difference is the **reality gap**.

This case learns the gap from recorded runs, uses the learned model as a fast
stand-in for the real robot, optimizes a motion against it, and runs the result on
the robot to check whether the gain holds.

## The task

The pipeline runs end to end with baseline implementations. The work is to replace
them with better ones so the agent's improvement transfers to hardware. The swap
points are three models, `DistillModel`, `Dynamics`, `EvaluationMetric`, plus the
`Preprocess` helper; each is an interface with a working default (what each
does now and where to change it is in **What runs now** below). Then train the RL
agent, optimize a held-out motion, and compare baseline vs optimized on the robot.

## Prerequisites

```bash
pip install -r requirements.txt
```

- **URSim or a UR robot** in **Remote Control** mode, reachable at `--robot-ip`
  (default `127.0.0.1`). Without Remote Control the controller accepts the socket
  but does not run the script.
- **Recorded runs** in `data/`, one CSV per run. Use the provided runs or record
  your own:
  ```bash
  python record.py --robot-ip <ip> --out data/test-1.csv --float-register 1 vel 2 acc
  ```
  `record.py` reads the controller's RTDE stream and only logs; it never moves the
  robot. RTDE channels: <https://www.universal-robots.com/developer/communication-protocol/rtde/>.
- `gymnasium` and `stable-baselines3` (in `requirements.txt`) for training.

## How to run

Distill once, then either mode reuses the model.

```bash
# 1. distill the gap model from the recorded real runs
python train_distillation_model.py --csvs data/test-4.csv data/test-5.csv data/test-6.csv --out models/distill.pkl

# 2. train the RL agent on several scripts (--loop repeats each for more moves)
python train_rla.py --mode params --robot-ip 127.0.0.1 --loop 5 --model models/distill.pkl --steps 20000 \
    --scripts scripts/shoulder_swing.script scripts/vertical_swing.script scripts/horizontal_swing.script

# 3. optimize a held-out script the agent did not train on
python run.py --mode params --script scripts/triangle.script --model models/distill.pkl --robot-ip 127.0.0.1

# 4. run baseline and optimized on the robot, compare
python send.py --robot-ip 127.0.0.1 --script scripts/triangle.script --loop 10 --out baseline.csv
python send.py --robot-ip 127.0.0.1 --script scripts/triangle.optimized.script --loop 10 --out optimized.csv
python analysis.py --csv baseline.csv --joint 0
python analysis.py --csv optimized.csv --joint 0
```

For **path** mode, use `--mode path` in steps 2 and 3, then stream the result:

```bash
python send.py --robot-ip 127.0.0.1 --path scripts/triangle.path --out optimized.csv
```

Step 3 tests transfer: if the drop the model predicted holds on hardware, the model
matched the robot; if not, it was missing something, which sends you back to the
distillation.

## Folder contents

| File | Role |
|------|------|
| `record.py` | passive RTDE logger: stream robot state to a CSV, never moves the robot |
| `send.py` | send a URScript (or a `servoj` path) to the robot, run it, record it |
| `analysis.py` | `Recording` (shared CSV loader) + per-joint stats and a current plot |
| `common.py` | `segments`: split a recording into waypoint-to-waypoint moves |
| `dynamics.py` | `Dynamics` interface + `UR10eDynamics`: candidate target torque/current |
| `train_distillation_model.py` | `DistillModel` interface + `LinearModel`: predict the actual channels |
| `metrics.py` | `EvaluationMetric` interface + `CurrentGapMetric`: the per-row `score` to minimize |
| `preprocess.py` | `Preprocess` interface: reshape data into and out of the learners |
| `train_rla.py` | Gym envs over the models; trains a PPO agent (`GapEnv` params, `PathEnv` path) |
| `run.py` | ask the trained agent for a better motion, write the optimized script or path |
| `utils.py` | constants, UR10e physics (FK, Jacobian, gravity, mass matrix, Coriolis), URScript load/edit |

`scripts/` holds the URScript motions, `models/` the trained models, `data/` the
recordings.

**How it flows.** A recording is a CSV of per-joint channels over time (commanded
`target_*`, measured `actual_*`, `vel`/`acc`, the running URScript line);
`analysis.Recording` loads it, `common.segments` splits it into moves (one per
movej). `build_dataset` runs the scripts on URSim, has `DistillModel` fill the
`actual_*` columns and `EvaluationMetric` add a `score` column, into
`sim_to_real.csv`. The PPO agent proposes an action per move; to score it,
`Dynamics` builds the candidate's commanded trajectory, `DistillModel` predicts the
actuals, `EvaluationMetric` scores them.

**What runs now (all of it is yours to change):**

- **`DistillModel`** (`train_distillation_model.py`): a per-row least-squares
  `LinearModel` predicting `actual_current` from `[target_current, qd, qdd, pos,
  vel, acc, joint]`. Change the features (`_row_features`), the predicted channel
  (`predicts`), or the whole model.
- **`Dynamics`** (`dynamics.py`): `UR10eDynamics`, `tau = M(q)qdd + g(q)`,
  `current = tau/Kt` (Coriolis dropped); `vel`/`acc` deg/s to rad/s. Override
  `current(q, qd, qdd)` for friction, Coriolis, identified parameters.
- **`EvaluationMetric`** (`metrics.py`): `CurrentGapMetric`, `|actual_current -
  target_current|` summed over joints. Change `needs`/`per_row` for overshoot,
  jerk, a weighted mix.
- **`Preprocess`** (`preprocess.py`): `Identity` (no-op). Subclass to normalize or
  scale features into and out of the learners.
- **The RLA** (`train_rla.py`): observation = 16 numbers (the move + its distilled
  channels + baseline score); objective = `score + cycle_time` (`OBJECTIVE`).
  `params` action = `[vel, acc]` (deg/s), a trapezoidal speed along the movej line;
  `path` action = `[accel_frac, decel_frac, speed]`, replaying the recorded
  trajectory at a trapezoidal speed profile. Change `observe`, the `OBJECTIVE`
  weights, or the action.
- `utils.UR10e` supplies the robot physics (FK, Jacobian, gravity, mass matrix,
  Coriolis) for `Dynamics` and as `DistillModel` features.

## Tiers

- **Bronze, understand it:** run the pipeline end to end, use `analysis.py` to see
  the reality gap, explore the `Preprocess` step, and decide what `EvaluationMetric`
  should measure. Record more runs with `record.py` if you like.
- **Silver, build the model:** write your own `DistillModel`, choose the features
  and architecture, and beat the linear baseline on held-out runs.
- **Gold, optimize it:** improve the RL agent (observation, `OBJECTIVE`, reward) and
  the `Dynamics` torque model (friction, Coriolis, identified parameters), and beat
  a fixed baseline's score.
- **Diamond, push to real:** shape the servoj path (path mode) and transfer to a
  real UR10e, refit the `DistillModel`/`Dynamics` on the real recordings, and close
  the sim-to-real loop until the robot measurably improves.
