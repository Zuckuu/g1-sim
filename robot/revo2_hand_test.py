#!/usr/bin/env python3
"""Staged, recorded Revo 2 grasp test over Unitree's brainco DDS bridge (rt/brainco/<hand>/cmd|state).

Reproduces the protocol that first held a 12 oz can on the right hand (2026-09-10):
  OPEN    all fingers 0; the operator stages the can against the palm, clear of the thumb arc
  OPPOSE  thumb_aux 0 -> 1.0 in 0.2 steps, actual position confirmed after each step
  CLOSE   thumb + 4 fingers advance in 0.1 steps; a finger whose actual position lags its command by more than
          --stall-threshold has hit the can and is frozen ("*"); repeat until every finger has stalled or --max-close
  HOLD    keep the final command --hold seconds (pull on the can), then RELEASE (fingers open, then thumb_aux back)

Facts about the bridge (brainco_hand_service/main.cpp): it re-sends the LAST received command to the hand at 100 Hz
forever, so the pose in the final message of any run (or abort) is what the hand keeps. q and dq are normalized
0..1 (0 open, 1 closed; dq = speed). Index order [thumb, thumb_aux, index, middle, ring, pinky]; tau_est = current (A).

Safety: hard-locked to one hand (--hand; right needs --allow-right), refuses to start if anything else publishes on
that hand's cmd topic or if the hand is not open (oppose/all), aborts to OPEN on over-current, Ctrl-C or watchdog.
Every state sample, command and event is written to --out (JSON + CSV).

Run on the Jetson (python 3.8, unitree_sdk2py):
  PY=~/miniforge3/envs/g1brainco/bin/python
  $PY revo2_hand_test.py --stage oppose                 # thumb across; leaves it there
  $PY revo2_hand_test.py --stage close --hold 10        # incremental close, hold, release
  $PY revo2_hand_test.py --stage release                # open everything (also the panic button)
  $PY revo2_hand_test.py --stage all --countdown 15     # whole protocol, staging countdown first
"""
import argparse
import csv
import json
import os
import signal
import sys
import threading
import time
from datetime import datetime

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmd_, MotorCmds_, MotorStates_

FINGERS = ["thumb", "thumb_aux", "index", "middle", "ring", "pinky"]
CLOSERS = [0, 2, 3, 4, 5]            # everything except thumb_aux takes part in CLOSE
SIGNATURE = 0x5150                   # stamped into MotorCmd_.reserve[0]; the bridge ignores reserve/mode/kp/kd/tau

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--hand", choices=["left", "right"], default="left")
parser.add_argument("--allow-right", action="store_true", help="required to touch the right hand (other tests run there)")
parser.add_argument("--stage", choices=["check", "oppose", "close", "release", "all"], default="check")
parser.add_argument("--iface", default="eth0")
parser.add_argument("--domain", type=int, default=0)
parser.add_argument("--speed", type=float, default=0.6, help="normalized finger speed sent as dq (bridge default/recommended 1.0)")
parser.add_argument("--mode", choices=["step", "ramp"], default="ramp",
                    help="step: 0.10 increments with dwells (the first protocol); ramp: one continuous command ramp at 50 Hz, "
                         "each finger frozen at actual+squeeze the moment it stops tracking (contact)")
parser.add_argument("--ramp-rate", type=float, default=0.4, help="ramp mode: command slope in normalized units/s (finger max ~1.2 at speed 0.6)")
parser.add_argument("--squeeze", type=float, default=0.10, help="ramp mode: command held this far beyond the contact position")
parser.add_argument("--squeeze-seconds", type=float, default=0.0, help="ramp mode: apply the squeeze as a ramp over this long (0 = step)")
parser.add_argument("--thumb-lag", type=float, default=0.0, help="ramp mode: thumb flex starts this many s after the fingers")
parser.add_argument("--aux-target", type=float, default=1.0, help="thumb_aux opposition target")
parser.add_argument("--aux-step", type=float, default=0.2, help="step mode oppose/release increments")
parser.add_argument("--aux-seconds", type=float, default=1.2, help="ramp mode: thumb_aux ramp duration (oppose and release)")
parser.add_argument("--release-seconds", type=float, default=0.8, help="ramp mode: closers open over this long")
parser.add_argument("--keep-aux", action="store_true", help="release leaves thumb_aux opposed (hand pre-shaped for the next run)")
parser.add_argument("--close-step", type=float, default=0.1)
parser.add_argument("--max-close", type=float, default=0.8, help="never command a closer beyond this")
parser.add_argument("--stall-threshold", type=float, default=0.07, help="cmd - actual above this = finger stopped by the can")
parser.add_argument("--stopped-threshold", type=float, default=0.04,
                    help="ramp mode: also call contact when the finger has not moved for ~40 ms while cmd - actual exceeds this")
parser.add_argument("--step-dwell", type=float, default=0.8, help="s to settle after each step before reading positions")
parser.add_argument("--hold", type=float, default=10.0, help="s to hold the grasp before releasing")
parser.add_argument("--keep", action="store_true", help="close stage: leave the hand closed (no release)")
parser.add_argument("--label", default="", help="free text stored with the recording (what was different about this run)")
parser.add_argument("--countdown", type=float, default=0.0, help="s of OPEN before OPPOSE so the operator can stage the can")
parser.add_argument("--max-current", type=float, default=1.2,
                    help="A; abort to OPEN if any finger stays above this for --over-current-seconds (step start/brake "
                         "transients reach 0.4-1.0 A / -2 A for 20-80 ms; steady hold currents are < 0.2 A)")
parser.add_argument("--over-current-seconds", type=float, default=0.25)
parser.add_argument("--max-seconds", type=float, default=90.0, help="hard watchdog")
parser.add_argument("--force", action="store_true", help="skip the 'hand must be open / thumb must be opposed' preconditions")
parser.add_argument("--out", default="")
args = parser.parse_args()

if args.hand == "right" and not args.allow_right:
    sys.exit("refusing to touch the right hand without --allow-right")
NS = "rt/brainco/%s" % args.hand
T0 = time.monotonic()
LOCK = threading.Lock()
STATE_ROWS = []          # [t, q0..q5, dq0..dq5, cur0..cur5]
CMD_ROWS = []            # [t, q0..q5, dq]
EVENTS = []
FOREIGN_CMDS = []        # cmd messages on our topic that we did not send
LATEST = {"q": None, "dq": None, "cur": None, "t": None}
ABORTED = {"why": None}


def now():
    return time.monotonic() - T0


def log(msg, **kw):
    EVENTS.append(dict(t=round(now(), 3), msg=msg, **kw))
    print("[%6.2f] %s" % (now(), msg), flush=True)


def on_state(msg):
    st = msg.states
    if len(st) < 6:
        return
    t = now()
    q = [float(st[i].q) for i in range(6)]
    dq = [float(st[i].dq) for i in range(6)]
    cur = [float(st[i].tau_est) for i in range(6)]
    with LOCK:
        STATE_ROWS.append([round(t, 4)] + q + dq + cur)
        LATEST.update(q=q, dq=dq, cur=cur, t=t)


def on_cmd(msg):
    cmds = msg.cmds
    if len(cmds) >= 1 and list(cmds[0].reserve)[:1] == [SIGNATURE]:
        return  # ours
    with LOCK:
        FOREIGN_CMDS.append(dict(t=round(now(), 3), q=[float(c.q) for c in cmds], dq=[float(c.dq) for c in cmds]))


def latest():
    with LOCK:
        return (list(LATEST["q"]) if LATEST["q"] else None, list(LATEST["cur"]) if LATEST["cur"] else None, LATEST["t"])


def fmt_q(v, marks=None):
    return " ".join("%.2f%s" % (x, "*" if marks and marks[i] else "") for i, x in enumerate(v))


def fmt_named(v, scale=1.0, unit="", fmtstr="%.2f"):
    return " ".join("%s=%s%s" % (FINGERS[i], fmtstr % (x * scale), unit) for i, x in enumerate(v))


ChannelFactoryInitialize(args.domain, args.iface)
pub = ChannelPublisher(NS + "/cmd", MotorCmds_)
pub.Init()
sub_state = ChannelSubscriber(NS + "/state", MotorStates_)
sub_state.Init(on_state, 0)
sub_cmd = ChannelSubscriber(NS + "/cmd", MotorCmds_)
sub_cmd.Init(on_cmd, 0)


def publish(q, speed=None):
    speed = args.speed if speed is None else speed
    cmds = [MotorCmd_(mode=0, q=float(min(max(q[i], 0.0), 1.0)), dq=float(speed), tau=0.0, kp=0.0, kd=0.0,
                      reserve=[SIGNATURE, 0, 0]) for i in range(6)]
    pub.Write(MotorCmds_(cmds=cmds))
    with LOCK:
        CMD_ROWS.append([round(now(), 4)] + [float(x) for x in q] + [float(speed)])


def dwell(seconds, q=None):
    """wait, re-publishing q at 20 Hz (keep-alive; the bridge would hold it anyway) and checking the guards."""
    end = now() + seconds
    while now() < end:
        if q is not None:
            publish(q)
        guard()
        time.sleep(0.05)


OVER_SINCE = [None] * 6   # per finger: time the current first exceeded max-current in the current excursion


def guard():
    if ABORTED["why"]:
        return
    qq, cur, t = latest()
    if cur:
        for i, c in enumerate(cur):
            if abs(c) > args.max_current:
                if OVER_SINCE[i] is None:
                    OVER_SINCE[i] = t
                elif t - OVER_SINCE[i] >= args.over_current_seconds:
                    abort("sustained over-current on %s (%.0f mA for %.2f s) %s" % (
                        FINGERS[i], c * 1000.0, t - OVER_SINCE[i], fmt_named(cur, 1000.0, "mA", "%.0f")))
            else:
                OVER_SINCE[i] = None
    if t is not None and now() - t > 1.0:
        abort("state stream stalled (%.1fs without rt/brainco/%s/state)" % (now() - t, args.hand))
    with LOCK:
        nf = len(FOREIGN_CMDS)
    if nf:
        abort("another publisher is commanding this hand (%d foreign cmd msgs)" % nf)


def abort(why):
    if ABORTED["why"]:
        return
    ABORTED["why"] = why
    log("ABORT: %s -> opening" % why)
    open_all(fast=True)
    save()
    os._exit(2)


def ramp_to(q, goals, seconds, label=None, lag=None):
    """50 Hz linear ramp of q toward goals (dict index->value) over `seconds`; lag (dict index->s) delays a finger's start."""
    q = list(q)
    start = {i: q[i] for i in goals}
    lag = lag or {}
    t0 = now()
    total = seconds + max([0.0] + list(lag.values()))
    while True:
        el = now() - t0
        for i, g in goals.items():
            a = min(1.0, max(0.0, (el - lag.get(i, 0.0)) / seconds)) if seconds > 0 else 1.0
            q[i] = start[i] + (g - start[i]) * a
        publish(q)
        guard()
        if el >= total:
            break
        time.sleep(0.02)
    if label:
        qq, _, _ = latest()
        log("%s -> %s | act %s" % (label, fmt_q(q), fmt_named(qq)))
    return q


def open_all(fast=False):
    """fingers open first (so the thumb does not sweep into them), then thumb_aux back to 0 unless --keep-aux."""
    qq, _, _ = latest()
    q = list(qq) if qq else [0.0] * 6
    if fast:
        for i in CLOSERS:
            q[i] = 0.0
        publish(q, speed=1.0)
        time.sleep(0.8)
        publish([0.0] * 6, speed=1.0)
        time.sleep(0.5)
        return
    if args.mode == "ramp":
        q = ramp_to(q, {i: 0.0 for i in CLOSERS}, args.release_seconds, label="RELEASE fingers")
        dwell(0.3, q)
        if args.keep_aux:
            log("RELEASE keeps thumb_aux at %.2f (--keep-aux)" % q[1])
            dwell(0.2, q)
            return
        q = ramp_to(q, {1: 0.0}, args.aux_seconds, label="RELEASE aux")
        dwell(0.3, q)
        return
    for i in CLOSERS:
        q[i] = 0.0
    publish(q)
    log("RELEASE fingers -> 0")
    dwell(1.2, q)
    if args.keep_aux:
        log("RELEASE keeps thumb_aux at %.2f (--keep-aux)" % q[1])
        return
    aux = q[1]
    while aux > 1e-3:
        aux = max(0.0, aux - args.aux_step)
        q[1] = aux
        publish(q)
        dwell(0.5, q)
        qq, _, _ = latest()
        log("RELEASE aux=%.2f act=%.2f" % (aux, qq[1] if qq else float("nan")))
    publish([0.0] * 6)
    dwell(0.5, [0.0] * 6)


def watchdog():
    time.sleep(args.max_seconds)
    abort("watchdog %.0fs" % args.max_seconds)


def on_sigint(sig, frame):
    abort("SIGINT")


def save():
    out = args.out or "/tmp/revo2-%s-%s-%s.json" % (args.hand, args.stage, datetime.now().strftime("%Y%m%d-%H%M%S"))
    with LOCK:
        rows = list(STATE_ROWS)
        cmd_rows = list(CMD_ROWS)
        foreign = list(FOREIGN_CMDS)
    span = (rows[-1][0] - rows[0][0]) if len(rows) > 1 else 0.0
    doc = dict(
        hand=args.hand, stage=args.stage, iface=args.iface, started=datetime.now().isoformat(timespec="seconds"),
        args=vars(args), fingers=FINGERS, signature=SIGNATURE, aborted=ABORTED["why"],
        state_rate_hz=(len(rows) - 1) / span if span > 0 else None,
        state_columns=["t"] + ["q_" + f for f in FINGERS] + ["dq_" + f for f in FINGERS] + ["cur_A_" + f for f in FINGERS],
        cmd_columns=["t"] + ["q_" + f for f in FINGERS] + ["dq"],
        events=EVENTS, foreign_cmds=foreign, summary=SUMMARY, states=rows, cmds=cmd_rows,
    )
    with open(out, "w") as f:
        json.dump(doc, f)
    csv_path = out[:-5] + ".csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(doc["state_columns"])
        w.writerows(rows)
    print("REVO2_TEST_OUT %s (+ .csv) states=%d cmds=%d rate=%.1f Hz" % (out, len(rows), len(cmd_rows), doc["state_rate_hz"] or 0.0), flush=True)


SUMMARY = {}


def precheck(require_open, require_opposed):
    log("CHECK listening 2.0 s on %s/state and %s/cmd" % (NS, NS))
    time.sleep(2.0)
    qq, cur, t = latest()
    with LOCK:
        n = len(STATE_ROWS)
        nf = len(FOREIGN_CMDS)
    if qq is None:
        log("no state from the %s hand: is brainco_hand.service running / the hand bound?" % args.hand)
        save()
        os._exit(1)
    if nf:
        log("someone else publishes on %s/cmd (%d msgs in 2 s): %s -> not touching the hand" % (NS, nf, FOREIGN_CMDS[0]))
        save()
        os._exit(1)
    log("state %.0f Hz | POS %s | CUR %s" % (n / 2.0, fmt_named(qq), fmt_named(cur, 1000.0, "mA", "%.0f")))
    SUMMARY["precheck"] = dict(state_hz=n / 2.0, q=qq, cur_A=cur)
    if require_open and not args.force and max(qq) > 0.15:
        log("hand is not open (max q %.2f): run --stage release first, or --force" % max(qq))
        save()
        os._exit(1)
    if require_opposed and not args.force and qq[1] < 0.5:
        log("thumb_aux is %.2f: run --stage oppose first, or --force" % qq[1])
        save()
        os._exit(1)
    return qq


def stage_oppose(q):
    q = list(q)
    for i in CLOSERS:
        q[i] = 0.0  # fingers stay open (precheck verified they are)
    if args.countdown > 0:
        log("OPEN - stage the can against the palm, keep it clear of the thumb arc (%.0f s)" % args.countdown)
        publish([0.0] * 6)
        remaining = args.countdown
        while remaining > 0:
            dwell(min(1.0, remaining), [0.0] * 6)
            remaining -= 1.0
            if remaining > 0 and int(remaining) % 5 == 0:
                print("        ... %d s" % int(remaining), flush=True)
    t_start = now()
    if args.mode == "ramp":
        q = ramp_to(q, {1: args.aux_target}, args.aux_seconds, label="OPPOSE ramp %.1fs" % args.aux_seconds)
        dwell(0.4, q)
    else:
        aux = q[1]
        while aux < args.aux_target - 1e-6:
            aux = min(args.aux_target, aux + args.aux_step)
            q[1] = aux
            publish(q)
            dwell(args.step_dwell, q)
            qq, cur, _ = latest()
            log("OPPOSE aux=%.2f act=%.2f cur=%.0f mA" % (aux, qq[1], cur[1] * 1000.0))
    qq, cur, _ = latest()
    SUMMARY["oppose"] = dict(aux_cmd=args.aux_target, aux_act=qq[1], seconds=round(now() - t_start, 2), cur_A=cur[1])
    log("OPPOSE done act=%.2f cur=%.0f mA" % (qq[1], cur[1] * 1000.0))
    return q


def stage_close(q):
    q = list(q)
    for i in CLOSERS:
        q[i] = 0.0                          # closers start from exactly open -> steps 0.10, 0.20, ...
    if abs(q[1] - args.aux_target) < 0.15:
        q[1] = args.aux_target              # re-command the opposition target, not the slightly-off reading
    stalled = [False] * 6
    stalled[1] = True                       # thumb_aux is fixed during CLOSE
    steps = []
    if args.mode == "ramp":
        return close_ramp(q, stalled)
    for step in range(40):
        active = [i for i in CLOSERS if not stalled[i] and q[i] < args.max_close - 1e-6]
        if not active:
            break
        for i in active:
            q[i] = round(min(args.max_close, q[i] + args.close_step), 3)
        publish(q)
        dwell(args.step_dwell, q)
        act, cur, _ = latest()
        newly = []
        for i in active:
            if q[i] - act[i] >= args.stall_threshold:
                stalled[i] = True
                newly.append(FINGERS[i])
        log("CMD %s POS %s" % (fmt_q(q, stalled), fmt_named(act)))
        log("CMD %s CUR %s%s" % (fmt_q(q, stalled), fmt_named(cur, 1000.0, "mA", "%.0f"),
                                 ("  stalled: " + ", ".join(newly)) if newly else ""))
        steps.append(dict(t=round(now(), 3), cmd=list(q), act=act, cur_A=cur, stalled=list(stalled)))
    act, cur, _ = latest()
    SUMMARY["close"] = dict(mode="step", steps=steps, final_cmd=list(q), final_act=act, final_cur_A=cur, stalled=stalled,
                            all_stalled=all(stalled[i] for i in CLOSERS))
    return hold_phase(q)


def close_ramp(q, stalled):
    """one continuous close: every closer's command ramps at --ramp-rate; the moment a finger stops tracking (contact)
    its command is frozen at actual + --squeeze. Thumb flex may start --thumb-lag later than the fingers."""
    t0 = now()
    start_q = list(q)
    contact = {}
    recent = {i: [] for i in CLOSERS}       # last few actual positions per finger, for the 'stopped' test
    last_print = 0.0
    log("CLOSE ramp %.2f/s, squeeze +%.2f, thumb lag %.2fs, speed %.2f" % (args.ramp_rate, args.squeeze, args.thumb_lag, args.speed))
    squeeze_ramp = {}                       # finger -> (t_contact, from, to) when --squeeze-seconds > 0
    while True:
        el = now() - t0
        act, cur, _ = latest()
        for i, (tc, q_from, q_to) in list(squeeze_ramp.items()):
            a = min(1.0, (el - tc) / args.squeeze_seconds)
            q[i] = q_from + (q_to - q_from) * a
            if a >= 1.0:
                del squeeze_ramp[i]
        for i in CLOSERS:
            if stalled[i]:
                continue
            lag = args.thumb_lag if i == 0 else 0.0
            q[i] = min(args.max_close, start_q[i] + args.ramp_rate * max(0.0, el - lag))
            recent[i].append(act[i])
            recent[i] = recent[i][-4:]
            err = q[i] - act[i]
            stopped = len(recent[i]) == 4 and max(recent[i]) - min(recent[i]) < 0.005 and q[i] >= start_q[i] + 0.10
            if err >= args.stall_threshold or (stopped and err >= args.stopped_threshold):
                stalled[i] = True
                q_hold = min(args.max_close, act[i] + args.squeeze)
                if args.squeeze_seconds > 0:
                    squeeze_ramp[i] = (el, act[i], q_hold)
                    q[i] = act[i]
                else:
                    q[i] = q_hold
                contact[FINGERS[i]] = dict(t=round(el, 3), act=act[i], cmd=q_hold, cur_A=cur[i],
                                           why="lag %.3f" % err if err >= args.stall_threshold else "stopped, lag %.3f" % err)
                log("CONTACT %-6s at %.2fs: act %.2f -> hold cmd %.2f (%s)" % (FINGERS[i], el, act[i], q_hold, contact[FINGERS[i]]["why"]))
            elif q[i] >= args.max_close - 1e-9:
                stalled[i] = True
                contact[FINGERS[i]] = dict(t=round(el, 3), act=act[i], cmd=q[i], cur_A=cur[i], why="max-close, no contact")
                log("MAX    %-6s at %.2fs: act %.2f, no contact" % (FINGERS[i], el, act[i]))
        publish(q)
        guard()
        if el - last_print >= 0.25:
            last_print = el
            print("        %.2fs CMD %s POS %s" % (el, fmt_q(q, stalled), fmt_named(act)), flush=True)
        if all(stalled[i] for i in CLOSERS) and not squeeze_ramp:
            break
        time.sleep(0.02)
    act, cur, _ = latest()
    SUMMARY["close"] = dict(mode="ramp", ramp_rate=args.ramp_rate, squeeze=args.squeeze, squeeze_seconds=args.squeeze_seconds, thumb_lag=args.thumb_lag,
                            contact=contact, seconds=round(now() - t0, 3), final_cmd=list(q), final_act=act, final_cur_A=cur,
                            stalled=stalled, all_stalled=all(k in contact and "no contact" not in contact[k]["why"] for k in
                                                              [FINGERS[i] for i in CLOSERS]))
    log("CLOSE done in %.2fs: contact %s" % (SUMMARY["close"]["seconds"],
                                             " ".join("%s@%.2f" % (k, v["act"]) for k, v in contact.items())))
    return hold_phase(q)


def hold_phase(q):
    log("HOLD %s for %.0f s%s" % (fmt_q(q), args.hold, "" if SUMMARY["close"]["all_stalled"] else "  (not all fingers stalled)"))
    hold_rows0 = len(STATE_ROWS)
    remaining = args.hold
    while remaining > 0:
        dwell(min(1.0, remaining), q)
        remaining -= 1.0
        act, cur, _ = latest()
        if int(round(remaining)) % 3 == 0:
            print("        hold %2d s | POS %s | CUR %s" % (int(round(remaining)), fmt_named(act), fmt_named(cur, 1000.0, "mA", "%.0f")), flush=True)
    with LOCK:
        hold_rows = STATE_ROWS[hold_rows0:]
    if hold_rows:
        SUMMARY["hold"] = dict(
            seconds=args.hold, samples=len(hold_rows),
            q_mean=[sum(r[1 + i] for r in hold_rows) / len(hold_rows) for i in range(6)],
            q_drift=[hold_rows[-1][1 + i] - hold_rows[0][1 + i] for i in range(6)],
            cur_A_mean=[sum(r[13 + i] for r in hold_rows) / len(hold_rows) for i in range(6)],
            cur_A_peak=[max(abs(r[13 + i]) for r in hold_rows) for i in range(6)],
        )
        log("HOLD drift %s | mean current %s" % (fmt_named(SUMMARY["hold"]["q_drift"], fmtstr="%+.3f"),
                                                  fmt_named(SUMMARY["hold"]["cur_A_mean"], 1000.0, "mA", "%.0f")))
    return q


signal.signal(signal.SIGINT, on_sigint)
threading.Thread(target=watchdog, daemon=True).start()
log("REVO2 %s hand test, stage=%s, speed=%.2f, max-close=%.2f, stall>=%.2f" % (args.hand, args.stage, args.speed, args.max_close, args.stall_threshold))

if args.stage == "check":
    precheck(False, False)
elif args.stage == "oppose":
    q0 = precheck(True, False)
    stage_oppose(q0)
    log("thumb opposed; hand left in this pose (bridge holds it). Next: --stage close")
elif args.stage == "close":
    q0 = precheck(False, True)
    if not args.force and max(q0[i] for i in CLOSERS) > 0.15:
        log("closers are not open (%s): --stage release first, or --force" % fmt_named(q0))
        save()
        os._exit(1)
    q1 = stage_close(q0)
    if args.keep:
        log("--keep: hand left closed. Open with --stage release")
    else:
        open_all()
elif args.stage == "release":
    precheck(False, False)
    open_all()
elif args.stage == "all":
    q0 = precheck(False, False)
    if not args.force and max(q0[i] for i in CLOSERS) > 0.15:
        log("closers are not open (%s): --stage release first, or --force" % fmt_named(q0))
        save()
        os._exit(1)
    if q0[1] >= 0.5:
        log("thumb already opposed (aux %.2f): skipping OPPOSE" % q0[1])
        q1 = list(q0)
        q1[1] = args.aux_target if abs(q0[1] - args.aux_target) < 0.15 else q0[1]
        if args.countdown > 0:
            log("stage the can against the palm (%.0f s)" % args.countdown)
            dwell(args.countdown, q1)
    else:
        q1 = stage_oppose(q0)
    dwell(0.6, q1)
    q2 = stage_close(q1)
    if args.keep:
        log("--keep: hand left closed. Open with --stage release")
    else:
        open_all()

qq, cur, _ = latest()
log("DONE POS %s | CUR %s" % (fmt_named(qq), fmt_named(cur, 1000.0, "mA", "%.0f")))
save()
os._exit(0)
