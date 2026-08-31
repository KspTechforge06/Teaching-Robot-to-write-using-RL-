# Writing Robot RL — Status Summary

**Project:** 3D writing robot — Arduino Uno + HW-130 (L293D) shield, 2 DVD stepper motors, servo pen-lift on D10.
**Goal (approved phase plan):** Phase 1 = complete horizontal single-line stroke → Phase 2 = vertical stroke via alternate-batch → Phase 3 = combine into an "L".
**Working dir:** `/home/ksp/writing_robot_rl` — **Git repo:** `/home/ksp/3D-writing-Robot` (branch `main`).
**Date:** 2026-09-01

---

## ✅ Done

### Firmware & calibration
- Firmware `dvd_stepper_serial.ino` supports servo `p <angle>` on D10, plus `?` help, `Moving`, `Pen servo set` responses. Compiles clean (7704 B / 23%), flashed via arduino-cli.
- Calibration: X = **5.0 steps/mm**, Y = **4.8 steps/mm**; safe bounds X max **52 mm**, Y max **48.96 mm** (15-step margin). `move_to_mm` clamps.
- **Position tracking fixed** — `move_x/y/both` now update `x_pos`/`y_pos`; `home()` resets to (0,0).
- Robust reconnect: `_reconnect()` polls up to 15 s for USB port to reappear after drops.

### Reward & environment
- **`path_progress` bug fixed.** `env._polyline_progress(pos)` = arc-length projection onto closest segment → monotonic through corners. Verified L `[[0,0],[0,44],[44,44]]`: origin=0%, corner=50%, `(44,44)`=100%.
- **`activity` reward** punishes sitting at origin → broke the "stay-put" local optimum.
- **`phase_bonus` reward added** (`reward.py`) — one-time bonus per junction crossing (e.g. L corner at progress≈0.5). Disabled by default (weight=0). Auto-enabled (=30.0) when `stroke_sim_train.py --line L`.

### Simulation training (`stroke_sim_train.py`, stable PPO n_steps=2048 / batch=64 / lr=3e-4)
- **Horizontal** → 97.7 % max progress, 43.5 mm reach, 0 violations. `writing_robot_stroke_horiz_model.zip`
- **Vertical** → 97.7 % max progress, 43.5 mm reach, 0 violations. `writing_robot_stroke_vert_model.zip`
- Single-line strokes are reliable to ~97.7 % (effectively a complete 43.5/44 mm line).

### Hardware validation
- Motors + servo confirmed working: home, ±X, ±Y, pen 45/90 all respond (direct test, 2026-09-01).
- **"L" drawn successfully!** The two-model approach worked perfectly on hardware (vertical to 45mm, horizontal to 44mm).

### `stroke_draw_hw.py` — **FIXED & VERIFIED (2026-09-01)**
- **Final fix applied:** Instead of injecting positions, we use both models exactly in their trained domain. The vertical stroke goes from (0,0) to (0,44), then the robot homes with pen up. The horizontal stroke then goes from (0,0) to (44,0). The origin (0,0) acts as the corner of the L.
- Added `--dry-run` and `--skip-fw-check` flags.
- Firmware verification now built-in (`verify_firmware()`).

### Tools added (2026-09-01)
- **`fw_check.py`** — standalone firmware verifier; run before every hardware session. Exit 0=OK, 2=stub, 3=unexpected.
- **`hw_rl_stroke.py`** — now backs up model to `writing_robot_stroke_model_R{N-1}.zip` before each overwrite (prevents losing strong rounds like old R4/R6). Use `--no-backup` to disable.

---

## ❌ / ⚠️ Not working / Blocked

### 1. Single-pass "L" in sim — **FIXED (2026-09-01)**
- `phase_bonus` reward successfully broke the "collapse to vertical-only" optimum.
- Discovered a truncation bug: `max_episode_steps=120` was causing the agent to run out of time at 87% progress.
- **Fix:** Increased `max_episode_steps=250` for the L-shape in `stroke_sim_train.py`.
- **Status:** Agent is now successfully reaching 98.4%+ progress and navigating both segments of the L.

### 2. Hardware PPO instability
- Round-based hardware RL (n_steps=8, batch=8) swings 95 % ↔ 0 % between rounds.
- Checkpoint backup now in place — won't lose strong rounds again.
- Consider pushing sim-trained policies to hardware instead of online HW-RL.

### 3. Environment fragility
- **Stub firmware** intermittently replaces ours. Use `fw_check.py` before every run.
- **USB drops** (Errno 5) under motor/servo load; reconnect usually recovers.
- Port flips ACM0↔ACM1 across replugs — always `ls /dev/ttyACM*` first.

---

## 🔜 To do (recommended order)

1. **Wait for single-pass L training to finish** (Phase 3). It is currently running in the background and exceeding 98% progress.
   ```bash
   python3 stroke_sim_train.py --line L --timesteps 300000
   ```

2. **Test single-pass L on real hardware** once the model finishes training.

3. **Stabilize hardware RL** or push sim policies to hardware directly (`stroke_draw_hw.py` is now the validation path).

4. **Checkpoint discipline**: `hw_rl_stroke.py` now auto-backs up. Keep strong round models around.

5. **Git**: commit `reward.py`, `env.py`, `stroke_sim_train.py`, `stroke_draw_hw.py`, `hw_rl_stroke.py`, `fw_check.py`, this file.

---

## Key files
- `writing_robot/env.py` — `_polyline_progress` (monotonic progress), obs Dict, servo angles.
- `writing_robot/reward.py` — `activity`, `near_completion`, `phase_bonus` (junction sub-goal), `_segment_thresholds`.
- `writing_robot/hardware.py` — per-axis calibration, position tracking, 15 s-poll reconnect, `set_pen`.
- `stroke_sim_train.py` — sim training; auto-enables `phase_bonus=30` for `--line L`.
- `hw_rl_stroke.py` — hardware round-based trainer; backs up checkpoint before overwrite.
- `stroke_draw_hw.py` — **FIXED** two-stage L on hardware (vert → reposition → horiz from corner). Has `--dry-run`.
- `fw_check.py` — **NEW** firmware verification utility.
- `hw_stroke_weights.json` — running reward weights.
- Models: `writing_robot_stroke_{horiz,vert}_model.zip` (good, ~97.7 %); `writing_robot_stroke_L_model3.zip` (vertical-only, needs re-train with phase_bonus).
