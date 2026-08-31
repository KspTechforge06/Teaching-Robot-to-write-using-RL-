#!/usr/bin/env python3
"""Stable simulation training for a SINGLE-LINE stroke with servo (minimal rotation).

Trains fast (~500 FPS) with proper PPO settings, using the same calibrated bounds,
single-line trajectories, and reward (activity + near_completion) as the hardware RL.
The resulting policy can be transferred to the real robot for validation.

Usage:
    python3 stroke_sim_train.py --line horiz --timesteps 20000 [--render]
"""
import argparse, time, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from writing_robot.env import WritingRobotEnv
from writing_robot.hardware import SimulatedHardware

MAX_X_MM = 52.0
MAX_Y_MM = 49.0

LINES = {
    "horiz": np.array([[0, 0], [44, 0]], dtype=np.float32),
    "vert":  np.array([[0, 0], [0, 44]], dtype=np.float32),
    "diag":  np.array([[0, 0], [44, 44]], dtype=np.float32),
    "L":     np.array([[0, 0], [0, 44], [44, 44]], dtype=np.float32),
}

PEN_UP = 90
PEN_DOWN = 85
WEIGHTS = {"path": 15.0, "progress": 150.0, "smooth": 0.2, "pen": 0.0,
           "completion": 40.0, "near_completion": 60.0, "activity": 3.0,
           "phase_bonus": 0.0}  # set to 30+ for L path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--line", choices=list(LINES), default="horiz")
    ap.add_argument("--timesteps", type=int, default=20000)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--phase-bonus", type=float, default=None,
                    help="One-time bonus per segment junction (auto=30 for L, 0 others)")
    args = ap.parse_args()

    weights = dict(WEIGHTS)
    if args.phase_bonus is not None:
        weights["phase_bonus"] = args.phase_bonus
    elif args.line == "L":
        weights["phase_bonus"] = 30.0  # auto-enable corner sub-goal for L
        print(f"  Auto-enabling phase_bonus=30.0 for L path")

    max_ep_steps = 250 if args.line == "L" else 120

    env = WritingRobotEnv(
        trajectory=LINES[args.line],
        use_hardware=False,
        render_mode="human" if args.render else None,
        x_range=(0.0, MAX_X_MM),
        y_range=(0.0, MAX_Y_MM),
        max_step_mm=1.5,
        max_episode_steps=max_ep_steps,
        steps_per_mm=5.0,
        steps_per_mm_x=5.0,
        steps_per_mm_y=4.8,
        max_steps=(275, 250),
        safety_margin_steps=15,
        trajectory_density=1.0,
        pen_up_angle=PEN_UP,
        pen_down_angle=PEN_DOWN,
        reward_weights=weights,
    )
    env.hardware = SimulatedHardware()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback

    vec_env = DummyVecEnv([lambda: env])

    class BoundsCB(BaseCallback):
        def __init__(self):
            super().__init__()
            self.max_x = self.max_y = 0.0
            self.viol = 0; self.steps = 0
            self.max_progress = 0.0; self.max_reach = 0.0
        def _on_step(self):
            pos = self.locals["new_obs"]["position"][0]
            p = float(self.locals["new_obs"]["path_progress"][0][0])
            self.steps += 1
            self.max_x = max(self.max_x, float(pos[0]))
            self.max_y = max(self.max_y, float(pos[1]))
            self.max_progress = max(self.max_progress, p)
            if args.line == "horiz":
                self.max_reach = max(self.max_reach, float(pos[0]))
            elif args.line == "vert":
                self.max_reach = max(self.max_reach, float(pos[1]))
            elif args.line in ("diag", "L"):
                self.max_reach = max(self.max_reach, float(pos[0]), float(pos[1]))
            if float(pos[0]) > MAX_X_MM or float(pos[1]) > MAX_Y_MM:
                self.viol += 1
            return True
        def _on_rollout_end(self):
            print(f"  sim steps={self.steps} maxX={self.max_x:.1f} maxY={self.max_y:.1f} "
                  f"max_progress={self.max_progress:.1%} reach={self.max_reach:.1f} viol={self.viol}")

    bcb = BoundsCB()
    if args.resume:
        print("resuming:", args.resume)
        model = PPO.load(args.resume, env=vec_env, verbose=1)
    else:
        print(f"new model: line={args.line}, weights={WEIGHTS}, servo {PEN_UP}<->{PEN_DOWN}")
        model = PPO("MultiInputPolicy", vec_env, learning_rate=3e-4,
                    n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99,
                    gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
                    vf_coef=0.5, max_grad_norm=0.5, verbose=1)

    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=bcb, progress_bar=False,
                reset_num_timesteps=not bool(args.resume))
    dt = time.time() - t0
    save = f"writing_robot_stroke_{args.line}_model"
    model.save(save)
    print(f"\nDONE in {dt:.0f}s ({args.timesteps/dt:.0f} fps) -> {save}.zip")
    print(f"FINAL: max_progress={bcb.max_progress:.1%} reach={bcb.max_reach:.1f}mm viol={bcb.viol}")
    if weights.get("phase_bonus", 0) > 0:
        print(f"  phase_bonus={weights['phase_bonus']} (corner sub-goal was active)")

    env.close()

if __name__ == "__main__":
    main()
