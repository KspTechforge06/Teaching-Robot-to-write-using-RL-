#!/usr/bin/env python3
"""
Hardware RL with servo on a SINGLE-LINE stroke trajectory.
Servo rotation kept minimal (pen up/down angles close together).
One PPO batch per round + report, mirrored on our calibrated real robot.

Usage:
  python3 hw_rl_stroke.py --round 1 --timesteps 48                      # fresh model
  python3 hw_rl_stroke.py --round 2 --timesteps 80 --resume             # continue
  python3 hw_rl_stroke.py --round 3 --timesteps 80 --line diag          # other stroke
"""
import sys, time, json, argparse, shutil
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from writing_robot.env import WritingRobotEnv
from writing_robot.hardware import HardwareInterface

TRAVEL_X = 275; TRAVEL_Y = 250
STEPS_X  = 5.0; STEPS_Y = 4.8
MARGIN   = 15
MAX_X_MM = (TRAVEL_X - MARGIN) / STEPS_X   # 52.0
MAX_Y_MM = (TRAVEL_Y - MARGIN) / STEPS_Y   # ~48.96
PORT = "/dev/ttyACM0"

LINES = {
    # a single straight stroke (the "line") inside safe bounds
    # all START AT ORIGIN so the agent begins exactly on the path
    "horiz": np.array([[0, 0], [44, 0]], dtype=np.float32),           # along X axis
    "vert":  np.array([[0, 0], [0, 44]], dtype=np.float32),           # along Y axis
    "diag":  np.array([[0, 0], [44, 44]], dtype=np.float32),          # diagonal
    "longh": np.array([[0, 0], [50, 0]], dtype=np.float32),
}

WEIGHTS_FILE = Path(__file__).parent / "hw_stroke_weights.json"
MODEL_FILE   = Path(__file__).parent / "writing_robot_stroke_model"

# MINIMAL servo travel: pen-down just a few degrees from pen-up
PEN_UP   = 90
PEN_DOWN = 85   # keep rotation minimal (~5 deg)
SMOOTH_DELAY = 0.05

def load_weights():
    if WEIGHTS_FILE.exists():
        return json.loads(WEIGHTS_FILE.read_text())
    # emphasize keeping on the line + making progress; small smooth penalty
    # near_completion gives a dense gradient for the final stretch (0.9->1.0)
    return {"path": 15.0, "progress": 150.0, "smooth": 0.2, "pen": 0.0,
            "completion": 40.0, "near_completion": 60.0, "activity": 3.0}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--timesteps", type=int, default=48)
    ap.add_argument("--line", choices=list(LINES), default="horiz")
    ap.add_argument("--weights", type=str, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-backup", action="store_true",
                    help="Skip checkpoint backup before overwriting model")
    ap.add_argument("--port", default=PORT)
    ap.add_argument("--lr", type=float, default=None, help="override learning rate on resume")
    args = ap.parse_args()

    weights = load_weights()
    if args.weights:
        for kv in args.weights.split(","):
            k, v = kv.split("=")
            weights[k.strip()] = float(v.strip())
    WEIGHTS_FILE.write_text(json.dumps(weights, indent=2))

    trajectory = LINES[args.line]
    print(f"\nROUND {args.round}  line={args.line}  weights={weights}")
    print(f"  servo: up={PEN_UP} down={PEN_DOWN} (minimal ~{abs(PEN_UP-PEN_DOWN)} deg)")
    print(f"  traj: {trajectory[0]} -> {trajectory[1]}")

    env = WritingRobotEnv(
        trajectory=trajectory,
        use_hardware=True,
        port=args.port,
        render_mode=None,
        x_range=(0.0, MAX_X_MM),
        y_range=(0.0, MAX_Y_MM),
        max_step_mm=1.5,
        max_episode_steps=80,
        steps_per_mm=5.0,
        steps_per_mm_x=STEPS_X,
        steps_per_mm_y=STEPS_Y,
        max_steps=(TRAVEL_X, TRAVEL_Y),
        safety_margin_steps=MARGIN,
        reward_weights=weights,
        trajectory_density=1.0,
        pen_up_angle=PEN_UP,
        pen_down_angle=PEN_DOWN,
    )

    import stable_baselines3.common.callbacks as cb
    class BoundsCB(cb.BaseCallback):
        def __init__(self):
            super().__init__()
            self.max_x=0.0; self.max_y=0.0; self.viol=0; self.moves=0
            self.max_progress=0.0; self.min_dist=1e9; self.max_reach_x=0.0
        def _on_step(self):
            pos = self.locals["new_obs"]["position"]
            x, y = float(pos[0][0]), float(pos[0][1])
            self.moves += 1
            self.max_x = max(self.max_x, x); self.max_y = max(self.max_y, y)
            self.max_reach_x = max(self.max_reach_x, x)
            p = float(self.locals["new_obs"]["path_progress"][0][0])
            self.max_progress = max(self.max_progress, p)
            if x > MAX_X_MM or y > MAX_Y_MM or x < 0 or y < 0:
                self.viol += 1
            return True

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    vec_env = DummyVecEnv([lambda: env])
    model_path = str(MODEL_FILE) + ".zip"
    if args.resume or Path(model_path).exists():
        print("  resuming:", model_path)
        model = PPO.load(model_path, env=vec_env, verbose=0)
        if args.lr is not None:
            model.learning_rate = args.lr
            model.lr_schedule = lambda _: args.lr
            print(f"  learning rate -> {args.lr}")
    else:
        print("  new model (random init)")
        model = PPO("MultiInputPolicy", vec_env,
                    learning_rate=5e-3, n_steps=args.timesteps if args.timesteps >= 8 else 8,
                    batch_size=8, n_epochs=10, gamma=0.99, verbose=0)

    bcb = BoundsCB()
    print(f"  training {args.timesteps} steps...")
    model.learn(total_timesteps=args.timesteps, callback=bcb, reset_num_timesteps=(args.round == 1))
    # Backup previous model before overwriting
    model_zip = Path(str(MODEL_FILE) + ".zip")
    if model_zip.exists() and not args.no_backup:
        backup = Path(str(MODEL_FILE) + f"_R{args.round - 1}.zip")
        shutil.copy2(model_zip, backup)
        print(f"  checkpoint backup -> {backup.name}")

    model.save(str(MODEL_FILE))

    env2 = vec_env.envs[0]
    print("\n===== REPORT (round %d) =====" % args.round)
    for k in ("path","progress","smooth","pen","completion"):
        print(f"weight {k:10s}: {weights.get(k)}")
    print(f"moves             : {bcb.moves}")
    print(f"max X commanded   : {bcb.max_x:.2f}mm  (bound {MAX_X_MM:.1f})")
    print(f"max Y commanded   : {bcb.max_y:.2f}mm  (bound {MAX_Y_MM:.1f})")
    print(f"max path_progress (in-round): {bcb.max_progress:.2%}")
    print(f"best reach along line (X): {bcb.max_reach_x:.2f}mm")
    print(f"boundary violations: {bcb.viol}")
    print(f"path_progress     : {env2.path_progress:.2%}")
    print(f"distance to path  : {np.linalg.norm(env2.position - env2._closest_path_point()):.2f}mm")
    print(f"final position    : {env2.position}, steps: {env2.step_count}")
    if bcb.viol == 0:
        print("BOUNDS: OK ✓")
    else:
        print("BOUNDS: VIOLATION ✗")
    env.close()

if __name__ == "__main__":
    main()
