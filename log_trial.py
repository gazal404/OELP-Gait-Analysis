#!/usr/bin/env python3
"""
log_trial.py - record ONE trial from the ESP32 into a CSV file.

WiFi (walking):
    1. Power the ESP32 (LED blinks slowly).
    2. Join the WiFi network  gait-esp32  (password gait12345) on the laptop.
    3. python log_trial.py --subject S01 --label normal --trial 1
USB (bench test): set USE_WIFI to 0 in the sketch, upload, then
    python log_trial.py --serial COM5 --subject S01 --label bench --trial 1
    (needs: pip install pyserial)

Press Enter to stop (or use --seconds 30 to stop automatically).

Files go to ./data/<subject>_<label>_T<trial>_<time>.csv and one line is added to
./data/trials_log.csv so you can see every trial in one place.
Units: accel in g, gyro in deg/s, t_us in microseconds since the stream started.
"""
import argparse
import math
import os
import socket
import statistics
import sys
import threading
import time
from datetime import datetime

COLS = ["n", "t_us",
        "th_ax_g", "th_ay_g", "th_az_g", "th_gx_dps", "th_gy_dps", "th_gz_dps",
        "sh_ax_g", "sh_ay_g", "sh_az_g", "sh_gx_dps", "sh_gy_dps", "sh_gz_dps"]
HZ = 100


def open_link(args):
    """Return read(): bytes, b'' when nothing arrived yet, None when the link closed."""
    if args.serial:
        import serial  # pip install pyserial
        port = serial.Serial(args.serial, args.baud, timeout=0.5)

        def read():
            return port.read(max(1, port.in_waiting))
        return read

    try:
        sock = socket.create_connection((args.host, args.port), timeout=5)
    except OSError as e:
        sys.exit(f"Could not connect to {args.host}:{args.port} ({e}).\n"
                 "  - Is the laptop joined to the WiFi network 'gait-esp32'?\n"
                 "  - Is the ESP32 powered, with the LED blinking slowly?\n"
                 "  - Is another program (or an earlier logger run) still connected to it?")
    sock.settimeout(0.5)

    def read():
        try:
            data = sock.recv(4096)
        except socket.timeout:
            return b""
        except OSError:
            return None
        return data if data else None
    return read


def noise(seg, first_gyro_col):
    """Average standard deviation of the three gyro axes, in deg/s."""
    return statistics.mean(
        statistics.pstdev([r[first_gyro_col + k] for r in seg]) for k in range(3))


def accel_mag(seg, first_acc_col):
    return statistics.mean(
        math.sqrt(r[first_acc_col] ** 2 + r[first_acc_col + 1] ** 2 + r[first_acc_col + 2] ** 2)
        for r in seg)


def summarize(rows):
    """Print a health report. Returns (duration_s, rate_hz, missing)."""
    if len(rows) < 2 * HZ:
        print("WARNING: fewer than 2 seconds of data recorded.")
        return 0.0, 0.0, 0
    n = [r[0] for r in rows]
    t = [r[1] for r in rows]
    dur = (t[-1] - t[0]) / 1e6
    rate = (len(rows) - 1) / dur if dur > 0 else 0.0
    missing = int(n[-1] - n[0] + 1) - len(rows)
    max_gap = max((b - a) / 1000.0 for a, b in zip(t, t[1:]))

    print(f"  samples: {len(rows)}   duration: {dur:.1f} s   rate: {rate:.1f} Hz")
    print(f"  missing samples: {missing} ({100.0 * missing / (missing + len(rows)):.1f} %)"
          f"   longest gap: {max_gap:.1f} ms")
    warns = []
    if not 95 <= rate <= 105:
        warns.append(f"sample rate is {rate:.0f} Hz, expected about {HZ}")
    if missing > 0.01 * len(rows):
        warns.append("more than 1 % of samples were dropped (WiFi range? loose shank wire?)")
    if max_gap > 30:
        warns.append(f"a gap of {max_gap:.0f} ms (a loose wire or WiFi stall)")

    for name, seg in (("start", rows[:2 * HZ]), ("end", rows[-2 * HZ:])):
        th_a, sh_a = accel_mag(seg, 2), accel_mag(seg, 8)
        th_n, sh_n = noise(seg, 5), noise(seg, 11)
        print(f"  standing check ({name}, 2 s): |a| thigh {th_a:.3f} g, shank {sh_a:.3f} g;"
              f"  gyro noise thigh {th_n:.2f}, shank {sh_n:.2f} deg/s")
        if max(th_n, sh_n) > 5:
            warns.append(f"not standing still at the {name} of the trial (gyro noise > 5 deg/s)")
        elif not (0.9 <= th_a <= 1.1 and 0.9 <= sh_a <= 1.1):
            warns.append(f"|a| at the {name} is not close to 1 g: sensor scale or wiring problem?")
    if warns:
        print("  WARNINGS:")
        for w in warns:
            print("   -", w)
    else:
        print("  all checks passed")
    return dur, rate, missing


def main():
    ap = argparse.ArgumentParser(description="Record one gait trial from the ESP32.")
    ap.add_argument("--subject", required=True, help="e.g. S01")
    ap.add_argument("--label", default="normal", help="normal / slow / fast / angletest / bench ...")
    ap.add_argument("--trial", type=int, default=1)
    ap.add_argument("--note", default="", help="free text saved in trials_log.csv")
    ap.add_argument("--seconds", type=float, default=0, help="stop automatically after N seconds")
    ap.add_argument("--host", default="192.168.4.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--serial", help="USB mode: COM port, e.g. COM5 or /dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(args.out, f"{args.subject}_{args.label}_T{args.trial:02d}_{stamp}.csv")

    read = open_link(args)

    stop = threading.Event()

    def wait_enter():
        try:
            input()
        except EOFError:
            return
        stop.set()
    threading.Thread(target=wait_enter, daemon=True).start()

    print(f"Recording to {path}")
    print("  1) stand still 3 s   2) 3 hard stomps   3) walk   4) stand still 3 s   5) press Enter")

    rows = []
    buf = b""
    t_start = time.time()
    try:
        with open(path, "w", newline="") as f:
            f.write(",".join(COLS) + "\n")
            while not stop.is_set():
                if args.seconds and time.time() - t_start >= args.seconds:
                    break
                chunk = read()
                if chunk is None:
                    print("The ESP32 closed the connection.")
                    break
                if not chunk:
                    continue
                buf += chunk
                *lines, buf = buf.split(b"\n")
                for ln in lines:
                    parts = ln.decode(errors="ignore").strip().split(",")
                    if len(parts) != len(COLS):
                        continue          # partial first line, boot text, etc.
                    try:
                        vals = [float(x) for x in parts]
                    except ValueError:
                        continue
                    f.write(",".join(parts) + "\n")
                    rows.append(vals)
    except KeyboardInterrupt:
        pass

    print(f"Saved {path}")
    dur, rate, missing = summarize(rows)

    log_path = os.path.join(args.out, "trials_log.csv")
    new = not os.path.exists(log_path)
    with open(log_path, "a") as lf:
        if new:
            lf.write("file,subject,label,trial,time,samples,duration_s,rate_hz,missing,note\n")
        note = args.note.replace(",", ";")
        lf.write(f"{os.path.basename(path)},{args.subject},{args.label},{args.trial},{stamp},"
                 f"{len(rows)},{dur:.1f},{rate:.1f},{missing},{note}\n")


if __name__ == "__main__":
    main()
