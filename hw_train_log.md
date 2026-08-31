# Hardware Training Log — Writing Robot RL (servo + single-line strokes)

Log owner: ksp / big-pickle
Repo: /home/ksp/3D-writing-Robot  (branch main)  + working dir /home/ksp/writing_robot_rl
System: Fedora linux; Arduino Uno + HW-130 L293D shield; servo on D10 (SERVO 1)

## Hardware state
- Port: /dev/ttyACM0 (has flipped ACM0 <-> ACM1 across replugs — ALWAYS re-check with `ls /dev/ttyACM*`)
- Firmware: dvd_stepper_serial.ino (p <angle> servo cmd on D10, `?` help, "Moving", "Pen servo set" responses). Flashed via arduino-cli.
- STALE-FIRMWARE RISK: board has twice re-enumerated with a stub that echoes `ok` to everything (no motor/servo). Verify with `?` -> expect "p <angle>". Re-flash if not.
- USB drops (OSError Errno 5) during motor/servo load: hardware.py _reconnect() polls up to 15s for port reappear. Mostly recovers but set_pen inside RL step can crash a round.

## Motor position tracking (hardware.py)  [FIXED this session]
- move_x/move_y/move_both now update x_pos/y_pos (steps / per-axis steps/mm).
- Calibration: X = 5.0 steps/mm (dead-end 275 steps ~55mm), Y = 4.8 steps/mm (dead-end 250 ~52mm).
- Safe bounds (15-step margin): X max 52.0mm (260 steps), Y max 48.96mm (235 steps). move_to_mm clamps.
- move_both needs "Moving both" response recognized.
- WARNING: raw relative moves must stay <= 240 steps/axis (6x50 = 300 striped the Y past its 250 dead end).

## Reward (writing_robot/reward.py)
- Added `activity` weight (penalizes sitting still; breaks the "stay-at-origin" local optimum). Verified: moving along path reward +17.4 vs sit-still 0.0.
- Existing: path (dist), progress, smooth (accel), pen, completion (>=1.0), near_completion (gradient in last 10%, p>=0.9).
- hw_stroke_weights.json: path=15, progress=150, smooth=0.2, pen=0, completion=40, near_completion=60, activity=3.

## Rewards / progress metric  [CHANGED this session]
- path_progress was `closest_idx/(N-1)` -> WRONG for multi-segment paths (non-monotonic, lets agent "sit at corner").
- NEW env._polyline_progress(pos): arc-length projection onto the globally-closest segment. Monotonic along the full route (through corners). VECTORIZED (15k calc/s, restores ~500fps training).
- Verified L trajectory [[0,0],[0,44],[44,44]]: (0,0)=0%, (0,22)=25%, corner=50%, (44,44)=100%. Single-line still linear 0->100%.

## Simulation training (stroke_sim_train.py, stable PPO n_steps=2048 batch=64 lr=3e-4)
- horiz  -> 97.7% max progress, 43.5mm reach, 0 violations  (writing_robot_stroke_horiz_model.zip)
- vert   -> 97.7% max progress, 43.5mm reach, 0 violations   (writing_robot_stroke_vert_model.zip)
- L (two-segment) -> still collapses to vertical-only (x span [0..0], progress ~48%). See TODO.

## Hardware RL rounds (hw_rl_stroke.py, tiny batches n_steps=8 batch=8)
- R4: 95.35% progress / 42mm reach   <-- strong run
- R6: 97.67% / 43mm (best, near_completion added)
- R7/R8: collapsed to 0% (PPO instability, low LR "stay-put" conv). Checkpoint overwritten, no backup.
- Fresh model + activity reward R1: 30.23% / 13.5mm in 200 steps (promising), R2 regressed 16% at lower LR.
- CONCLUSION: hardware PPO with batch 8 is inherently unstable round-to-round; sim with proper settings is the stable path.

## Hardware direct test (this session)
- connect True; home True; x+50->(10,0); x-50->(0,0); y+50->(0,10.4); y-50->(0,0); pen45/pen90 ok. MOTORS + SERVO WORK.

## stroke_draw_hw.py (two-stage L)  [BUG found]
- Tried drawing L as vertical-RL then reposition then horizontal-RL.
- BUG: each env.reset() HOMES to (0,0), so horizontal stroke reset to origin (ended [40.5, 0]) instead of staying at corner [40.5, 40]. Need single-pass instead. See TODO.

## Open questions / blockers
1. Single-pass L: even with fixed monotonic progress + 160k sim steps, PPO collapses to vertical-only (~48%). Likely needs segmented/phase reward or curriculum (train vert, freeze, then train corner+horiz) rather than one flat reward.
2. Hardware PPO instability (batch=8).
3. Stub-firmware / USB-drop robustness.
