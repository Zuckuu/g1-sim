# Physical Revo 2 bench test (no robot needed)

Goal: prove, on the real BrainCo Revo 2 hand, that it closes on the actual demo bottle and holds it —
before the G1 arrives. The hand only needs power and a USB-RS485 adapter.

## What you need on the bench

| Item | Notes |
| --- | --- |
| Revo 2 hand | Ours: **Pro and Touch** (`XEL/XER`, `XTL/XTR`). Touch = piezoresistive pads, 9 pressure points per finger, readable via registers/SDK. |
| Power | Pro/Touch: **12–64 V DC**. The Pro/Touch box ships a power cable; BrainCo's optional adapter is a **24 V supply with an XT30 connector**. Any 24 V bench supply with an XT30 pigtail works — confirm polarity against the cable before plugging in. Basic hands are 12–28 V. [Product manual](https://brainco-common-public.oss-cn-hangzhou.aliyuncs.com/web-config/docs-sdk/Revo-2-Product-Manual-V2.1EN.pdf), §6 packing list. |
| RS-485 → USB | Optional BrainCo kit = dual-port 485-to-USB converter + two 1.15 m 485 cables + USB-C. A generic USB-RS485 dongle (CH340/FTDI) also works with the supplied 485/CAN FD cable (A, B, GND). Shows up as `/dev/ttyUSB0`. A/B swapped = no detection, no damage. |
| Mechanical | Wrist flange: 38 mm diameter, 13.5 mm tall, 4× M3 on the circumference. Pro/Touch flange threads are only 3.5 mm deep — screws must not penetrate more than 3 mm (PCB behind it). CAD of the flange: [BrainCo download page](https://www.brainco-hz.com/docs/revolimb-hand/revo2/download.html). |
| A clamp or vise | The hand must be fixed to the table for a hold test; clamp the wrist flange or bolt it to a plate. |
| Bottles | Standard **20 oz Pepsi** (222 mm, 72.8 mm max diameter, ~0.64 kg full). Measure the grip-zone diameter and mass, note them in `docs/`. |

Default Modbus IDs: left hand `126` (0x7E), right hand `127` (0x7F). Baud for Revo 2 is normally 460800; the SDK auto-detects both.

## One-time setup

```bash
sudo usermod -aG dialout $USER        # then log out / in once
ls /dev/ttyUSB*                       # adapter present?
work/hand-venv/bin/python -c "from bc_stark_sdk import main_mod as s; print(s.get_sdk_version())"   # 2.0.5
```

The venv `work/hand-venv` already has `bc-stark-sdk` (BrainCo's official PyPI package, from their
[brainco-hand-sdk](https://github.com/BrainCoTech/brainco-hand-sdk) repo).

## Procedure

```bash
# 1. connect + print device info, voltage, motor status
work/hand-venv/bin/python hand/revo2_bench.py status

# 2. confirm motor index -> finger mapping (each motor moves alone to 40%)
work/hand-venv/bin/python hand/revo2_bench.py probe

# 3. bottle grasp cycles: opens, waits for Enter, closes, logs positions/currents for 4 s, asks you the outcome
work/hand-venv/bin/python hand/revo2_bench.py grasp --cycles 5 --label pepsi-500ml --force small
```

Start with `--force small` (protected current 300). Move to `normal`, then `full` only if the bottle slips.
Adjust the grasp shape with `--grasp thumb,thumb_aux,index,middle,ring,pinky` (0 open … 1000 closed), e.g.
`--grasp 650,900,950,950,950,950` (default) or a gentler `--grasp 550,850,850,850,850,850`.

Logs go to `hand/logs/*.json` (positions, currents, motor states per sample; touch summaries on a Touch hand).

## What we learn from it

* Whether the 100 mm max opening and 50 N grip handle the real bottle diameter/mass (500 mL ≈ 65 mm, 20 oz ≈ 73 mm; a 2 L bottle at ~110 mm will not fit).
* Which motor currents correspond to a firm hold — this becomes the grasp-verification signal in the demo
  (a Touch hand gives pressure directly).
* The finger positions where the fingers stall on the bottle — used to validate the simulation model.

## Safety

* Keep fingers, cables and the USB adapter out of the closing envelope; the hand closes with ≥50 N in ≤0.65 s.
* The script opens the hand on Ctrl-C and on exit.
* If anything looks wrong, cut power at the supply.
