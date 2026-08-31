#!/usr/bin/env python3
"""Quick firmware verification: connects to the Arduino and checks the help
output to confirm our dvd_stepper_serial.ino is flashed (not the stub).

Usage:
    python3 fw_check.py [--port /dev/ttyACM0]
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    import argparse
    import serial
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None, help="Serial port (auto-detect if omitted)")
    ap.add_argument("--baud", type=int, default=9600)
    args = ap.parse_args()

    import serial.tools.list_ports
    port = args.port
    if port is None:
        ports = serial.tools.list_ports.comports()
        for p in ports:
            if "ACM" in p.device or "Arduino" in p.description or "USB" in p.device:
                port = p.device
                break
    if port is None:
        print("ERROR: no Arduino port found. Plug in and retry, or use --port.")
        sys.exit(1)

    import os
    if not os.path.exists(port):
        print(f"ERROR: {port} does not exist. Run: ls /dev/ttyACM*")
        sys.exit(1)

    print(f"Connecting to {port} @ {args.baud}...")
    try:
        ser = serial.Serial(port, args.baud, timeout=3.0)
        time.sleep(2.0)  # wait for Arduino reset
        ser.reset_input_buffer()

        # Send '?' help command
        ser.write(b"?\n")
        ser.flush()
        time.sleep(0.5)

        lines = []
        while ser.in_waiting:
            line = ser.readline().decode(errors="replace").strip()
            if line:
                lines.append(line)

        # Also try 'p 90' and check response
        ser.write(b"p 90\n")
        ser.flush()
        time.sleep(0.5)
        while ser.in_waiting:
            line = ser.readline().decode(errors="replace").strip()
            if line:
                lines.append(line)

        ser.close()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print("\nFirmware response:")
    for l in lines:
        print(f"  {l}")

    full = " ".join(lines).lower()
    has_pen = "pen servo set" in full or "p <angle>" in full or "pen" in full
    has_move = "moving" in full or "x <steps>" in full or "r " in full

    if has_pen and has_move:
        print("\n✅ Firmware OK — dvd_stepper_serial.ino confirmed (pen servo + move commands found).")
        sys.exit(0)
    elif "ok" in full and len(lines) <= 3:
        print("\n❌ STUB firmware detected (echoes 'ok' to everything). Re-flash before running RL!")
        sys.exit(2)
    else:
        print("\n⚠️  Unexpected response — check manually. Expected 'Pen servo set' and movement commands.")
        sys.exit(3)


if __name__ == "__main__":
    main()
