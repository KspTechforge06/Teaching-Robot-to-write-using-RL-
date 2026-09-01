#!/usr/bin/env python3
"""Draw a big heart in the center of the workspace using the robot."""
import sys, argparse
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from writing_robot.hardware import HardwareInterface

def generate_heart(center_x=26.0, center_y=24.5, scale=1.25, num_points=60, rotation_deg=0.0):
    t = np.linspace(0, 2 * np.pi, num_points)
    x = 16 * np.sin(t)**3
    y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
    
    # Invert Y because the robot's Y axis increases downwards
    y = -y
    
    # Center the coordinates exactly at (0,0) based on their bounding box
    x = x - (x.max() + x.min()) / 2.0
    y = y - (y.max() + y.min()) / 2.0
    
    # Apply rotation
    if rotation_deg != 0.0:
        theta = np.radians(rotation_deg)
        x_rot = x * np.cos(theta) - y * np.sin(theta)
        y_rot = x * np.sin(theta) + y * np.cos(theta)
        x, y = x_rot, y_rot
    
    # Scale up and move to desired center
    x = x * scale + center_x
    y = y * scale + center_y
    
    # Combine into Nx2 array
    points = np.vstack((x, y)).T
    return points

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.25, help="Scale of the heart (1.25 = ~40mm wide)")
    ap.add_argument("--cx", type=float, default=26.0, help="Center X in mm")
    ap.add_argument("--cy", type=float, default=24.5, help="Center Y in mm")
    ap.add_argument("--points", type=int, default=60, help="Number of points to form the shape")
    ap.add_argument("--rotation", type=float, default=0.0, help="Rotation angle in degrees")
    ap.add_argument("--port", type=str, default="/dev/ttyACM0", help="Arduino port")
    run = ap.parse_args()

    points = generate_heart(run.cx, run.cy, run.scale, run.points, run.rotation)
    
    print(f"Drawing heart with {len(points)} points...")
    print(f"extent x[{points[:,0].min():.1f}..{points[:,0].max():.1f}] "
          f"y[{points[:,1].min():.1f}..{points[:,1].max():.1f}]")
    
    # Check bounds against safe area (52x48.96)
    ok = points[:,0].min() >= 0 and points[:,0].max() <= 52 and \
         points[:,1].min() >= 0 and points[:,1].max() <= 48.96
    
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
    
    print("Moving to start (top cleft of the heart)...")
    hw.move_to_mm(points[0][0], points[0][1])
    hw.set_pen(pen_down)
    
    print("Drawing...")
    for i, (x, y) in enumerate(points[1:]):
        hw.move_to_mm(x, y)
        if i % 10 == 0 or i == len(points) - 2:
            print(f"  point {i+1}/{len(points)-1} done")
            
    import time
    hw.set_pen(pen_up)
    
    print("Moving pen out of the way to display to camera...")
    hw.move_to_mm(52.0, 0.0)
    print("Holding for 5 seconds...")
    time.sleep(5.0)
    
    print("Resetting to home position...")
    hw.home()
    hw.disconnect()
    print("done ✓")

if __name__ == "__main__":
    main()
