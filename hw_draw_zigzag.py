#!/usr/bin/env python3
"""Draw a zigzag pattern with a specified number of segments using the robot."""
import sys, argparse
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from writing_robot.hardware import HardwareInterface

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=10, help="Number of zigzag segments")
    ap.add_argument("--start_x", type=float, default=5.0, help="Starting X in mm")
    ap.add_argument("--start_y", type=float, default=24.0, help="Starting Y in mm")
    ap.add_argument("--width", type=float, default=40.0, help="Total width in mm")
    ap.add_argument("--height", type=float, default=20.0, help="Peak to peak height in mm")
    ap.add_argument("--port", type=str, default="/dev/ttyACM0", help="Arduino port")
    run = ap.parse_args()

    points = []
    dx = run.width / run.count
    dy = run.height / 2.0

    for i in range(run.count + 1):
        x = run.start_x + i * dx
        if i == 0 or i == run.count:
            y = run.start_y
        elif i % 2 == 1:
            y = run.start_y - dy
        else:
            y = run.start_y + dy
        points.append((x, y))

    allpts = np.array(points)
    print(f"Drawing zigzag with {run.count} segments...")
    print(f"extent x[{allpts[:,0].min():.1f}..{allpts[:,0].max():.1f}] "
          f"y[{allpts[:,1].min():.1f}..{allpts[:,1].max():.1f}]")
    
    # Check bounds against safe area (52x48.96)
    ok = allpts[:,0].min() >= 0 and allpts[:,0].max() <= 52 and \
         allpts[:,1].min() >= 0 and allpts[:,1].max() <= 48.96
    
    print("within bounds:", ok)
    if not ok:
        print("ABORTING — outside safe area")
        return

    hw = HardwareInterface(port=run.port, steps_per_mm_x=5.0, steps_per_mm_y=4.8,
                           max_steps=(275, 250), safety_margin_steps=15)
    hw.connect()
    hw.set_speed(40)
    hw.home()
    
    pen_up, pen_down = 90, 75
    hw.set_pen(pen_up)
    
    print("Moving to start...")
    hw.move_to_mm(points[0][0], points[0][1])
    hw.set_pen(pen_down)
    
    print("Drawing...")
    for i, (x, y) in enumerate(points[1:]):
        hw.move_to_mm(x, y)
        print(f"  segment {i+1}/{run.count} done")
        
    hw.set_pen(pen_up)
    hw.home()
    hw.disconnect()
    print("done ✓")

if __name__ == "__main__":
    main()
