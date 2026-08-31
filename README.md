# Writing Robot RL

Reinforcement Learning for 2-axis writing robot using recycled DVD stepper motors.

## Overview

This package provides:
- **Gymnasium Environment** (`WritingRobotEnv`) for both simulation and real hardware
- **PPO Training** via Stable Baselines3 with sub-goal curriculum learning (`phase_bonus`)
- **Hardware Integration** tools for firmware validation and execution
- **Vector Drawing** for direct hardware manipulation (e.g. `hw_write_text.py`)

## Installation

```bash
cd writing_robot_rl
pip install -r requirements.txt
```

## Quick Start (Simulation)

```bash
# Train PPO on square trajectory
python -m writing_robot.train --env sim --algo ppo --timesteps 100000 --shape square

# Train on circle
python -m writing_robot.train --env sim --algo ppo --timesteps 200000 --shape circle

# Manual control for testing
python -m writing_robot.evaluate --mode manual --env sim

# Benchmark environment speed
python -m writing_robot.evaluate --mode benchmark --steps 1000
```

## Real Hardware

The project now supports robust hardware execution with safeguards:

```bash
# 1. Verify firmware (REQUIRED before hardware runs)
python3 fw_check.py --port /dev/ttyACM0

# 2. Draw "L" shape using dual trained RL policies
python3 stroke_draw_hw.py --port /dev/ttyACM0

# 3. Draw direct vector text (e.g. "P")
python3 hw_write_text.py P --lw 24 --lh 44
```

*Note: The hardware scripts automatically apply rate-limiting (`time.sleep(0.25)`) to prevent Arduino serial buffer overflows.*

## Architecture

```
writing_robot_rl/
├── writing_robot/
│   ├── __init__.py
│   ├── env.py              # Gymnasium Environment
│   ├── hardware.py         # Serial interface
│   ├── trajectory.py       # SVG/font → waypoints
│   ├── reward.py           # Reward functions
│   └── sim/
│       └── viewer_2d.py    # Pygame 2D viewer
├── train.py                # Training entry point
├── evaluate.py             # Evaluation tools
├── configs/
│   └── default.yaml        # Config file
└── characters/             # SVG fonts, stroke definitions
```

## Environment Details

### Observation Space (Dict)
```python
{
    "position": (2,),           # current XY in mm
    "velocity": (2,),           # current velocity mm/step
    "pen_state": (1,),          # 0=up, 1=down
    "path_progress": (1,),      # 0.0 to 1.0 along trajectory
    "steps_remaining": (1,),    # steps left in episode
    "current_target": (2,),     # next target waypoint
    "target_waypoint": int,     # index in trajectory
}
```

### Action Space (Continuous)
```python
Box(-1, 1, shape=(3,))  # [dx, dy, pen]
# dx, dy: normalized step (-1 to 1 -> max_step_mm)
# pen: >0 = down, <=0 = up
```

### Reward Components
| Component | Weight | Description |
|---|---|---|
| `path` | 15.0 | Negative distance to trajectory |
| `progress` | 150.0 | Increase in path completion |
| `activity` | 3.0 | Punishes sitting still at origin |
| `completion` | 40.0 | Bonus for finishing trajectory |
| `phase_bonus` | 30.0 | Sub-goal bonus for passing corners/junctions |

## Custom Trajectories

### From SVG
```python
from writing_robot import svg_to_trajectory

traj = svg_to_trajectory("letter_a.svg", density=1.0, scale=1.0)
env = WritingRobotEnv(trajectory=traj)
```

### From Text (requires freetype-py)
```python
from writing_robot import trajectory_from_text

traj = trajectory_from_text("HELLO", font_size=30.0, density=1.0)
env = WritingRobotEnv(trajectory=traj)
```

### Built-in Shapes
```python
from writing_robot.trajectory import generate_basic_shapes

shapes = generate_basic_shapes()
# shapes["square"], shapes["circle"], shapes["figure8"], shapes["spiral"]
```

## Configuration

Copy `configs/default.yaml` and modify:

```yaml
env:
  x_range: [0.0, 100.0]
  y_range: [0.0, 100.0]
  max_step_mm: 2.0

ppo:
  learning_rate: 3e-4
  n_steps: 2048

reward:
  path: 10.0
  progress: 100.0
```

Use with:
```bash
python -m writing_robot.train --config configs/my_config.yaml
```

## Hardware Interface

The `HardwareInterface` class wraps the Arduino serial protocol:

```python
from writing_robot import HardwareInterface

hw = HardwareInterface(port="/dev/ttyACM0")
hw.connect()
hw.move_to_mm(50.0, 50.0)
hw.set_speed(60)
hw.disconnect()
```

### Arduino Firmware
Upload `dvd_stepper_serial.ino` from the main project. It implements:
- `x <steps>`, `y <steps>`, `b <steps>`
- `r` (return to home)
- `s <rpm>` (speed)
- `p <angle>` (pen servo - needs firmware update)

## Extending

### Add New Reward
```python
from writing_robot.reward import compute_reward

def my_reward(prev_obs, action, obs, target_path):
    base = compute_reward(prev_obs, action, obs, target_path)
    # Add custom terms
    return base + custom_term
```

### Custom Training Loop
```python
from writing_robot.env import WritingRobotEnv

env = WritingRobotEnv(render_mode="human")
# Your custom RL loop here
```

## Sim-to-Real Tips

1. **Domain Randomization**: Add noise to observations during sim training
2. **Dynamics Gap**: Calibrate `steps_per_mm` on real hardware
3. **Latency**: Account for serial communication delay (~10-50ms)
4. **Pen Servo**: Add `p <angle>` command to Arduino firmware for pen control

## License

MIT