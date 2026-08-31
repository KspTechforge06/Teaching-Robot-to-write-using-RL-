#!/usr/bin/env python3
"""Draw an 'L' on real hardware using two sim-trained policies.

L shape: vertical (0,0)->(0,40) + horizontal (0,0)->(44,0), corner at origin.
Both models run in their trained domain — no position injection needed.

Usage:
    python3 stroke_draw_hw.py [--port /dev/ttyACM0] [--dry-run]
"""
import sys, time, os
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from writing_robot.env import WritingRobotEnv
from writing_robot.hardware import HardwareInterface
from stable_baselines3 import PPO

MAX_X_MM = 52.0
MAX_Y_MM = 49.0
PORT     = "/dev/ttyACM0"
PEN_UP, PEN_DOWN = 90, 85
W = {"path": 15.0, "progress": 150.0, "smooth": 0.2, "pen": 0.0,
     "completion": 40.0, "near_completion": 60.0, "activity": 3.0}

# Both trajectories start at origin — matching training exactly, increased size to 48mm
VERT  = np.array([[0,  0], [ 0, 48]], dtype=np.float32)  # up along Y
HORIZ = np.array([[0,  0], [48,  0]], dtype=np.float32)  # right along X


def make_hw_env(traj, port):
    return WritingRobotEnv(
        trajectory=traj, use_hardware=True, port=port, render_mode=None,
        x_range=(0.0, MAX_X_MM), y_range=(0.0, MAX_Y_MM),
        max_step_mm=1.5, max_episode_steps=120,
        steps_per_mm=5.0, steps_per_mm_x=5.0, steps_per_mm_y=4.8,
        max_steps=(275, 250), safety_margin_steps=15,
        trajectory_density=1.0, reward_weights=W,
        pen_up_angle=PEN_UP, pen_down_angle=PEN_DOWN,
    )


def run_policy(model, env, label):
    """Lift pen before reset so homing doesn't drag on paper."""
    # Pre-lift before reset homes the robot
    if env.hardware and env.hardware.connected:
        env.hardware.set_pen(PEN_UP)
        time.sleep(0.3)
    obs, _ = env.reset()   # homes, then sets pen down at origin
    done = False; n = 0
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(a)
        time.sleep(0.25)  # Wait for Arduino to finish physical step (prevents buffer overflow)
        done = term or trunc
        n += 1
    print(f"  {label}: {n} steps, progress={info['path_progress']:.1%}, end={env.position}")
    return env.position.copy()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=PORT)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    port = args.port

    # --- pre-flight ---
    if not os.path.exists(port):
        print(f"ERROR: {port} not found. Run: ls /dev/ttyACM*"); sys.exit(1)
    for f in ("writing_robot_stroke_vert_model.zip",
              "writing_robot_stroke_horiz_model.zip"):
        if not Path(f).exists():
            print(f"ERROR: model missing: {f}"); sys.exit(1)
        print(f"  model OK: {f}")

    if args.dry_run:
        print("\n--dry-run OK. Nothing sent to hardware."); return

    # --- [1/3] vertical stroke: (0,0) -> (0,44) ---
    print("\n[1/3] Drawing vertical stroke (0,0) -> (0,44) ...")
    env_v = make_hw_env(VERT, port)
    model_v = PPO.load("writing_robot_stroke_vert_model", env=env_v, verbose=0)
    end_v = run_policy(model_v, env_v, "vertical")
    env_v.close()
    time.sleep(0.8)

    # --- [2/3] lift pen (robot already homed by env_v.close) ---
    print("\n[2/3] Lifting pen at origin for horizontal start ...")
    hw = HardwareInterface(port=port, steps_per_mm_x=5.0, steps_per_mm_y=4.8,
                           max_steps=(275, 250), safety_margin_steps=15)
    hw.connect()
    hw.set_pen(PEN_UP); time.sleep(0.5)
    hw.disconnect()
    time.sleep(0.3)

    # --- [3/3] horizontal stroke: (0,0) -> (44,0) ---
    print("\n[3/3] Drawing horizontal stroke (0,0) -> (44,0) ...")
    env_h = make_hw_env(HORIZ, port)
    model_h = PPO.load("writing_robot_stroke_horiz_model", env=env_h, verbose=0)
    end_h = run_policy(model_h, env_h, "horizontal")
    env_h.close()
    time.sleep(0.5)

    # --- finish: pen up + home ---
    hw = HardwareInterface(port=port, steps_per_mm_x=5.0, steps_per_mm_y=4.8,
                           max_steps=(275, 250), safety_margin_steps=15)
    hw.connect(); hw.set_pen(PEN_UP); hw.home(); hw.disconnect()

    print(f"\n'L' shape:")
    print(f"  vertical   : (0,0) -> {end_v}  [up]")
    print(f"  horizontal : (0,0) -> {end_h}  [right]")
    print(f"  corner at origin (0,0) — robot homed.")


if __name__ == "__main__":
    main()
