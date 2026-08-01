"""Optimize a motion against the models and write it out for the robot.

Runs train_rla.py's pipeline without training: builds the dataset for the chosen
script (`build_dataset`), loads the trained agent, asks it for a motion, scores it
against the models, and writes the result. Two modes:

  params  rewrite the script's `vel`/`acc` lines (movej). Writes
          `scripts/<name>.optimized.script`; run it with `send.py --script`.
  path    re-time each move into a servoj path. Writes `scripts/<name>.path`
          (a CSV of joint setpoints); run it with `send.py --path`.

    python run.py --mode params --script scripts/triangle.script --robot-ip 127.0.0.1
    python run.py --mode path   --script scripts/triangle.script --robot-ip 127.0.0.1

Requires an agent trained by `train_rla.py`; reads `models/agent_<mode>.zip`.
"""
from __future__ import annotations

import argparse
import csv
import os
import re

import numpy as np
from stable_baselines3 import PPO

from metrics import CurrentGapMetric
from preprocess import default_preprocess
from train_distillation_model import DistillModel
from train_rla import GapEnv, PathEnv, build_dataset, speed_profile
from utils import get_param, load_script, set_param


# --- params mode -------------------------------------------------------------

def script_moves(env):
    """The script's moves, dropping the first (the approach into the script from
    wherever the robot started, which is not part of the script's own motion)."""
    return env.targets[1:] if len(env.targets) > 1 else env.targets


def _gain(base, opt) -> float:
    """Percent reduction from baseline mean to optimized mean (positive = lower)."""
    return 100 * (base.mean() - opt.mean()) / base.mean()


def _compare(base, opt):
    """Print score and cycle-time change (base/opt are (score, cycle) arrays)."""
    print(f"predicted mean score: {base[:, 0].mean():.3f} -> {opt[:, 0].mean():.3f} "
          f"({_gain(base[:, 0], opt[:, 0]):+.1f}%)")
    print(f"predicted cycle time: {base[:, 1].mean():.3f} -> {opt[:, 1].mean():.3f}s "
          f"({_gain(base[:, 1], opt[:, 1]):+.1f}%)")


def search_agent(env: GapEnv, moves, path: str) -> tuple[float, float]:
    """Ask a trained PPO agent for vel/acc (deg/s), averaged over the moves."""
    agent = PPO.load(path)
    params = [env._unmap(agent.predict(env._obs(m), deterministic=True)[0]) for m in moves]
    return tuple(np.mean(params, axis=0))


def report(env: GapEnv, moves, vel: float, acc: float, label: str) -> np.ndarray:
    """Print mean score and cycle time for one (vel, acc); return (score, cycle) per move."""
    res = np.array([env.score(m, vel, acc) for m in moves])
    print(f"{label}: vel {vel:.0f}  acc {acc:.0f}  score {res[:, 0].mean():.3f}  "
          f"cycle {res[:, 1].mean():.3f}s (mean over {len(moves)} moves)")
    return res


def run_params(args, model, metric, rec, pre):
    """Optimize the script's vel/acc and write the optimized script."""
    text = load_script(args.script)
    env = GapEnv(model, metric, rec, pre=pre)
    moves = script_moves(env)

    base = report(env, moves, get_param(text, "vel"), get_param(text, "acc"), "baseline (from script)")
    vel, acc = search_agent(env, moves, args.agent)
    opt = report(env, moves, vel, acc, "optimized")
    _compare(base, opt)

    out = re.sub(r"\.script$", ".optimized.script", args.script)
    with open(out, "w") as f:
        f.write(set_param(set_param(text, "vel", vel), "acc", acc))
    print(f"wrote {out}\n  run it: python send.py --script {out} --out optimized.csv")


# --- path mode ---------------------------------------------------------------

def agent_paths(env: PathEnv, moves, path: str):
    """Per move, the agent's (accel_frac, decel_frac, servoj dt)."""
    agent = PPO.load(path)
    return [env.unpack(agent.predict(env._obs(m), deterministic=True)[0]) for m in moves]


def report_path(env: PathEnv, moves, plans, label: str) -> np.ndarray:
    """Print mean max-score and cycle time; return (score, cycle) per move."""
    res = np.array([env.score(m, af, df, dt) for m, (af, df, dt) in zip(moves, plans)])
    print(f"{label}: max score {res[:, 0].mean():.3f}  cycle {res[:, 1].mean():.3f}s "
          f"(mean over {len(moves)} moves)")
    return res


def build_full_path(rec, moves, plans):
    """6-DOF servoj setpoints for the whole script, one block per move.

    Each move replays its recorded joint trajectory, re-timed by the agent's speed
    profile, so the path is preserved and only the speed changes. Output at the
    recorded resolution (dense) so the servoj locus matches the recording. Each row
    is ``q0..q5`` plus the move's servoj dt (a 7th column send.py reads).
    """
    rows = []
    for m, (accel_frac, decel_frac, dt) in zip(moves, plans):
        rec_q = rec.target_q[m.i0:m.i1]                 # recorded geometry, all joints
        n = len(rec_q)
        u = np.linspace(0.0, 1.0, n)
        s = speed_profile(accel_frac, decel_frac, n)    # re-timed, recorded resolution
        block = np.column_stack([np.interp(s, u, rec_q[:, j]) for j in range(6)])
        for r in block:
            rows.append([*r, dt])
    return rows


def run_path(args, model, metric, rec, pre):
    """Re-time each move and write the path CSV."""
    env = PathEnv(model, metric, rec, pre=pre)
    moves = script_moves(env)

    base = np.array([env.baseline(m) for m in moves])
    print(f"baseline (recorded speed): max score {base[:, 0].mean():.3f}  "
          f"cycle {base[:, 1].mean():.3f}s (mean over {len(moves)} moves)")
    plans = agent_paths(env, moves, args.agent)
    opt = report_path(env, moves, plans, "optimized")
    _compare(base, opt)

    rows = build_full_path(rec, moves, plans)
    out = re.sub(r"\.script$", ".path", args.script)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"q{j}" for j in range(6)] + ["dt"])
        w.writerows([f"{v:.6f}" for v in row] for row in rows)
    print(f"wrote {out}\n  run it: python send.py --path {out} --out optimized.csv")


def main():
    ap = argparse.ArgumentParser(description="Optimize a URScript motion against the model.")
    ap.add_argument("--mode", choices=("params", "path"), default="params",
                    help="params: rewrite vel/acc; path: shape a servoj path")
    ap.add_argument("--script", default="scripts/shoulder_swing.script",
                    help="URScript to optimize")
    ap.add_argument("--model", default="models/distill.pkl", help="distilled model")
    ap.add_argument("--robot-ip", default="127.0.0.1",
                    help="URSim IP to run the script on (default local URSim)")
    ap.add_argument("--loop", type=int, default=None,
                    help="repeat the script N times when collecting its moves")
    ap.add_argument("--agent", default=None,
                    help="trained PPO agent (default: models/agent_<mode>.zip)")
    args = ap.parse_args()
    args.agent = args.agent or f"models/agent_{args.mode}.zip"

    model = DistillModel.load(args.model)
    metric = CurrentGapMetric()
    pre = default_preprocess()
    # Dataset written to the root, named after the script (e.g. triangle.sim_to_real.csv).
    out = re.sub(r"\.script$", ".sim_to_real.csv", os.path.basename(args.script))
    rec = build_dataset(model, metric, [args.script], args.robot_ip, args.loop, pre, out)
    (run_params if args.mode == "params" else run_path)(args, model, metric, rec, pre)


if __name__ == "__main__":
    main()
