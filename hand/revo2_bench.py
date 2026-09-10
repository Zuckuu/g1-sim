"""Bench test for a physical BrainCo Revo 2 hand over USB-RS485 (Modbus RTU).

Modes
  probe   Move one motor at a time so you can confirm which motor index drives which finger.
  grasp   Open -> close on the bottle -> hold (logging positions/currents/states/touch) -> open, N cycles.
  status  Connect, print device info, voltage, motor status, exit.

Examples (from the project root, hand connected and powered):
  work/hand-venv/bin/python hand/revo2_bench.py status
  work/hand-venv/bin/python hand/revo2_bench.py probe
  work/hand-venv/bin/python hand/revo2_bench.py grasp --cycles 5 --hold-seconds 5 --label pepsi-500ml

Motor order used by the SDK (unified 0..1000 position range, 0 = fully open, 1000 = fully closed):
  0 Thumb (flex)   1 ThumbAux (rotation / opposition)   2 Index   3 Middle   4 Ring   5 Pinky
Verify this with `probe` before trusting any grasp numbers.

Safety
  * Keep fingers/cables clear; the hand closes with >=50 N.
  * Ctrl-C at any time sends an open command before exiting.
  * Start with --force small until the bottle grasp is proven.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

try:
    from bc_stark_sdk import main_mod as sdk
except ImportError:  # pragma: no cover
    print("bc_stark_sdk missing. Install with: uv pip install --python work/hand-venv/bin/python bc-stark-sdk", file=sys.stderr)
    raise

FINGERS = ["thumb", "thumb_aux", "index", "middle", "ring", "pinky"]
OPEN = [0] * 6
# Default bottle power grasp: thumb rotated into opposition, thumb flexed partway, four fingers mostly closed.
# The fingers will stall on the bottle before reaching these targets; that is expected.
DEFAULT_GRASP = [650, 900, 950, 950, 950, 950]
FORCE_PROTECTED_CURRENT = {"small": 300, "normal": 600, "full": 1000}


def parse_positions(text):
    vals = [int(v) for v in text.split(",")]
    if len(vals) != 6 or any(not 0 <= v <= 1000 for v in vals):
        raise argparse.ArgumentTypeError("need 6 comma-separated integers in 0..1000")
    return vals


async def connect(args):
    if args.port and args.baud and args.slave_id is not None:
        baud = getattr(sdk.Baudrate, f"Baud{args.baud}", None)
        if baud is None:
            baud = sdk.Baudrate.Baud460800
        ctx = await sdk.modbus_open(args.port, baud)
        slave_id = args.slave_id
        print(f"Opened {args.port} @ {args.baud} slave_id={slave_id}")
    else:
        print("Auto-detecting Revo 2 over Modbus/RS-485 ..." + (f" (port {args.port})" if args.port else ""))
        devices = await sdk.auto_detect(scan_all=False, port=args.port, protocol="Modbus")
        if not devices:
            print("No hand detected. Check: hand powered (12-28 V), USB-RS485 adapter enumerated (ls /dev/ttyUSB*),"
                  " user in `dialout` group, A/B lines not swapped.", file=sys.stderr)
            sys.exit(2)
        dev = devices[0]
        print(f"Detected: port={dev.port_name} protocol={dev.protocol_type} baud={dev.baudrate} slave_id={dev.slave_id} "
              f"hw={dev.hardware_type} sku={dev.sku_type} fw={dev.firmware_version} sn={dev.serial_number}")
        ctx = await sdk.init_from_detected(dev)
        slave_id = dev.slave_id
    info = await ctx.get_device_info(slave_id)
    print(f"Device: {info.description}")
    try:
        mv = await ctx.get_voltage(slave_id)
        print(f"Supply voltage: {mv/1000:.2f} V")
    except Exception as exc:  # some firmware may not expose it over Modbus
        print(f"Voltage read failed: {exc}")
    await ctx.set_finger_unit_mode(slave_id, sdk.FingerUnitMode.Normalized)
    return ctx, slave_id, info


async def read_status(ctx, slave_id, touch):
    st = await ctx.get_motor_status(slave_id)
    row = {
        "t": time.time(),
        "positions": list(st.positions),
        "currents": list(st.currents),
        "speeds": list(st.speeds),
        "states": [str(s).split(".")[-1] for s in st.states],
    }
    if touch:
        try:
            summ = await ctx.get_modulus_touch_summary(slave_id)
            row["touch_summary"] = [getattr(s, "description", str(s)) for s in summ]
        except Exception as exc:
            row["touch_error"] = str(exc)
    return row


def fmt(row):
    pos = " ".join(f"{p:4d}" for p in row["positions"])
    cur = " ".join(f"{c:5d}" for c in row["currents"])
    stt = " ".join(s[:4] for s in row["states"])
    return f"pos[{pos}] cur[{cur}] st[{stt}]"


async def mode_status(ctx, slave_id, info, args):
    touch = await ctx.is_touch_hand(slave_id) if hasattr(ctx, "is_touch_hand") else False
    print(f"Touch hand: {touch}")
    prot = await ctx.get_finger_protected_currents(slave_id)
    print(f"Protected currents: {list(prot)}")
    row = await read_status(ctx, slave_id, touch)
    print(fmt(row))


async def mode_probe(ctx, slave_id, info, args):
    print("Probe: each motor moves to 400 and back. Watch which finger moves and note the index.")
    await ctx.set_finger_positions_and_durations(slave_id, OPEN, [800] * 6)
    await asyncio.sleep(1.0)
    for i, name in enumerate(FINGERS):
        target = OPEN.copy()
        target[i] = 400
        print(f"  motor {i} (expected: {name}) -> 400")
        await ctx.set_finger_positions_and_durations(slave_id, target, [600] * 6)
        await asyncio.sleep(1.2)
        st = await ctx.get_motor_status(slave_id)
        print(f"     positions now: {list(st.positions)}")
        await ctx.set_finger_positions_and_durations(slave_id, OPEN, [600] * 6)
        await asyncio.sleep(1.0)
    print("Probe done. If the mapping differs from the expected names, record it in docs/ before running grasp.")


async def mode_grasp(ctx, slave_id, info, args):
    touch = await ctx.is_touch_hand(slave_id) if hasattr(ctx, "is_touch_hand") else False
    prot = FORCE_PROTECTED_CURRENT[args.force]
    await ctx.set_finger_protected_currents(slave_id, [prot] * 6)
    print(f"Protected current set to {prot} ({args.force}) for all motors; touch hand: {touch}")
    log = {
        "label": args.label, "device": info.description, "grasp_positions": args.grasp,
        "force": args.force, "protected_current": prot, "cycles": [],
    }
    for cycle in range(args.cycles):
        print(f"\n=== cycle {cycle+1}/{args.cycles} ===")
        await ctx.set_finger_positions_and_durations(slave_id, OPEN, [800] * 6)
        await asyncio.sleep(1.2)
        input("Place the bottle against the palm, then press Enter to close ...") if not args.no_prompt else None
        t_close = time.time()
        await ctx.set_finger_positions_and_durations(slave_id, args.grasp, [args.close_ms] * 6)
        samples = []
        t_end = t_close + args.close_ms / 1000.0 + args.hold_seconds
        while time.time() < t_end:
            row = await read_status(ctx, slave_id, touch)
            row["t"] -= t_close
            samples.append(row)
            print(f"  t={row['t']:5.2f}s {fmt(row)}")
            await asyncio.sleep(max(0.0, 1.0 / args.rate_hz))
        settled = samples[-1] if samples else None
        outcome = input("Result? [h]eld / [s]lipped / [d]ropped / [n]o-contact (Enter=held): ").strip().lower() if not args.no_prompt else ""
        outcome = {"": "held", "h": "held", "s": "slipped", "d": "dropped", "n": "no-contact"}.get(outcome, outcome)
        log["cycles"].append({"cycle": cycle + 1, "outcome": outcome, "settled": settled, "samples": samples})
        await ctx.set_finger_positions_and_durations(slave_id, OPEN, [800] * 6)
        await asyncio.sleep(1.2)
    out = Path(args.log_dir) / f"revo2-bench-{args.label}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(log, indent=2) + "\n")
    outcomes = [c["outcome"] for c in log["cycles"]]
    print(f"\nOutcomes: {outcomes}\nLog: {out}")


async def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["status", "probe", "grasp"])
    parser.add_argument("--port", default=None, help="/dev/ttyUSB0 etc. Default: auto-detect")
    parser.add_argument("--baud", type=int, default=None, help="used only with --port and --slave-id")
    parser.add_argument("--slave-id", type=int, default=None, help="126 = left default, 127 = right default")
    parser.add_argument("--grasp", type=parse_positions, default=DEFAULT_GRASP,
                        help="6 targets 0..1000 in SDK order thumb,thumb_aux,index,middle,ring,pinky")
    parser.add_argument("--force", choices=list(FORCE_PROTECTED_CURRENT), default="small")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--hold-seconds", type=float, default=4.0)
    parser.add_argument("--close-ms", type=int, default=700, help="commanded closing duration (1..2000 ms)")
    parser.add_argument("--rate-hz", type=float, default=10.0, help="status sampling rate during hold")
    parser.add_argument("--label", default="bottle")
    parser.add_argument("--no-prompt", action="store_true", help="do not wait for Enter between phases")
    parser.add_argument("--log-dir", default=str(Path(__file__).resolve().parent / "logs"))
    args = parser.parse_args()

    ctx, slave_id, info = await connect(args)
    try:
        await {"status": mode_status, "probe": mode_probe, "grasp": mode_grasp}[args.mode](ctx, slave_id, info, args)
    except KeyboardInterrupt:
        print("\nInterrupted; opening hand.")
    finally:
        try:
            await ctx.set_finger_positions_and_durations(slave_id, OPEN, [800] * 6)
            await asyncio.sleep(0.5)
        except Exception:
            pass
        try:
            await sdk.close_device_handler(ctx)
        except Exception:
            try:
                sdk.modbus_close(ctx)
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(main())
