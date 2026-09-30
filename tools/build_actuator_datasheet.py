"""Build ``data/robot/actuators_datasheet_v1.json``: Dropbear's actuator model from the MyActuator datasheets.

Phase 1a of making the simulated robot match the real one. PROPOSAL ONLY: no training code imports this file, and
no training default changes. A later step may add an ``hw_datasheet_v1`` actuator profile once the user agrees
(docs/ACTUATORS.md).

What it does (CPU only, numpy; the USD itself is NOT opened, the repo's read-only dumps of it are used):

1. Datasheet facts per motor model, each with its source file, page and SHA-256 (transcribed from the PDFs/CSVs,
   see ``MODELS``; the characteristic-curve tables are copied row by row).
2. The joint -> motor-model map with the evidence of every source and the conflict resolution (``JOINT_EVIDENCE``).
   The CAD evidence comes from ``tools/probe_usd_motor_prims.py`` (``logs/actuators/usd_motor_prims.log``).
3. Derived values: armature = rotor inertia x ratio^2, a DC-motor torque-speed line fitted to the 48 V curve,
   joint-side torque/speed/armature through the knee four-bar, the parallel ankle and the elbow linkage (calibration
   LUT slopes / Jacobian at the standing pose), the link inertia each motor sees at the authored rest pose, and
   physically motivated (TODO) gain suggestions.
4. The current sim values (``dropbear_names.ACTUATOR_PARAMS``, ``flat_env_cfg.ACTUATOR_PROFILES``, the USD 10 rad/s
   motor velocity cap) and the sim/datasheet ratios.
5. CAN-bus timing for the documented bus layouts.
6. ``--motions``: motor speeds of the accepted reference clips and of the recorded policy rollouts against the
   proposed per-motor speed gates.

Run::

    python tools/build_actuator_datasheet.py --motions > logs/actuators/build_actuator_datasheet_v1.log
"""
from __future__ import annotations

import argparse
import ast
import glob
import hashlib
import json
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
H = REPO.parent
sys.path.insert(0, str(REPO / "source"))

from dropbear_wbc.robots import dropbear_names as dn  # noqa: E402  (pure Python)

MYA = H / "myactuator-can"
DCV = H / "dropbear_control" / "assets" / "vendor" / "myactuator" / "docs" / "RMD-X"
RPM = 2.0 * math.pi / 60.0
G_CM2 = 1e-7  # g*cm^2 -> kg*m^2

# ------------------------------------------------------------------------------------------------------------------
# 1. Sources (read-only). Page numbers are PDF page indices (1-based); "printed" gives the page label when different.
# ------------------------------------------------------------------------------------------------------------------
SOURCES: dict[str, dict] = {
    "X10S2_PROFILE": {"path": MYA / "RMD-X10-S2 V3 - Actuator Profile.pdf",
                      "what": "RMD-X10-S2 V3 1:35 product parameters (catalog 240220, printed page 31/32); byte-identical "
                              "to X10-100/RMD-X10-S2 V3.pdf and the 240220 'Product parameters' copy"},
    "X10_PROFILE": {"path": MYA / "RMD-X10 V3 - Actuator Profile.pdf",
                    "what": "RMD-X10 V3 1:7 product parameters (catalog 240220, printed 29/30); byte-identical to "
                            "X10-40/RMD-X10 1:7 V3.pdf"},
    "X8V2_PROFILE": {"path": MYA / "RMD-X8 V2 - Actuator Profile.pdf",
                     "what": "RMD-X8 V2 1:9 (item A00083, 568 g, NOT the Pro; catalog 240220, printed 11/12); "
                             "byte-identical to X8-25/RMD-X8 1:9 V2.pdf"},
    "SIMPLE_PROFILE": {"path": MYA / "Actuator Profile (Simple).pdf",
                       "what": "user's summary tables: p1 RMD-X8 PRO, p2 RMD-X10-S2 1:35, p3 RMD-X10 1:7 (the 'Reducer "
                               "Ratio' cells are Excel time values: 0.3757 d = 9:01 -> 9:1, 1.4590 d = 35:01, "
                               "0.2924 d = 7:01)"},
    "X10_100": {"path": MYA / "X10-100" / "X10-100 (RMD-X10-S2 V3)" / "Product parameters" / "X10-100(RMD-X10-S2 V3).pdf",
                "what": "X10-100 = RMD-X10-P35-100-C-N (previous name RMD-X10 S2 1:35 V3), catalog 240403 printed "
                        "X-21/22; same content as X Series Product Manual240403.pdf page 13"},
    "X10_40": {"path": MYA / "X10-40" / "X10-40 Product information 240403" / "Product parameters"
                             / "X10-40(RMD-X10 1ú║7 V3).pdf",
               "what": "X10-40 = RMD-X10-P7-40-C-N (previous name RMD-X10 1:7 V3), catalog 240403 printed X-19/20; "
                       "same content as X Series Product Manual240403.pdf page 12"},
    "X8_25": {"path": MYA / "X8-25" / "X8-25 Product information 240403" / "Product parameters"
                            / "X8-25 (RMD-X8 Pro1ú║9 V2).pdf",
              "what": "X8-25 = RMD-X8-P9-25-C-N (previous name RMD-X8-Pro 1:9 V2), catalog 240403 printed X-11/12; "
                      "same content as X Series Product Manual240403.pdf page 8"},
    "XMAN_240403": {"path": MYA / "Protocol and Manual" / "X Series Product Manual240403.pdf",
                    "what": "X-series catalog 240403: p2 lineup, p8 X8-25, p10 X8-20 (RMD-X8-Pro-H 1:6 V3), p12 X10-40, "
                            "p13 X10-100"},
    "XV3MAN_241227": {"path": DCV / "X-V3-protocol-manual" / "vendor" / "(X-V3) Protocol and manual of V3-250213"
                                    / "X Series-V3 Product Manual-241227.pdf",
                      "what": "X-series V3 catalog 241227 (dropbear_control vendor copy): p5 X10-40, p6 X10-100; "
                              "same numbers as 240403 except encoders listed 14/14 bit"},
    "XV2MAN_241227": {"path": DCV / "X-V2-protocol-manual" / "vendor" / "(X-V2) Protocol and manual of V2-250213"
                                    / "X Series-V2 Product Manual-241227.pdf",
                      "what": "X-series V2 catalog 241227: p5 X8-25 (encoder listed 18 bit)"},
    "PROTO_V39": {"path": MYA / "Protocol and Manual" / "RMD-X Motor Motion Protocol V3.9-240415.pdf",
                  "what": "RMD-X CAN/RS485 protocol V3.9: p7 bus parameters; p45-49 0xA1 torque; p49-52 0xA2 speed; "
                          "p52-54 0xA4 absolute position; p74-75 0xB3 comm-loss protection; p77 0xB4 baud rate; "
                          "p80-82 0xB6 active reply; p85-86 0x280 multi-motor; p89-92 motion mode 0x400+ID. The "
                          "'RMD-L ... V3.9' PDF in the same folder is byte-identical."},
    "PROTO_V42": {"path": DCV / "X-V3-protocol-manual" / "vendor" / "(X-V3) Protocol and manual of V3-250213"
                                / "Motor Motion Protocol V4.2-250208.pdf",
                  "what": "protocol V4.2 shipped for X-V2 and X-V3 (byte-identical copies): p78 0xB3, p92-93 motion "
                          "mode (printed 83-84/95)"},
    "PROTO_V44": {"path": DCV / "X-V4-protocol-manual" / "vendor" / "CAN BUS Motor Motion Protocol V4.4 260425.pdf",
                  "what": "protocol V4.4 for the X-V4 generation (NOT Dropbear's V2/V3 motors): p161-163 motion mode "
                          "(printed 153-155)"},
    "MC_DRIVER": {"path": MYA / "Protocol and Manual" / "MC Series Brushless Servo Driver Manual-24.2.1.pdf",
                  "what": "MC driver manual: p7 drive table, loop rates (current 15 kHz, speed 5 kHz, position 1 kHz), "
                          "CAN 500K/1M, protections"},
    "DBG_SW": {"path": MYA / "Protocol and Manual" / "V3.0 Debugging Software Manual -latest version.pdf",
               "what": "setup software manual: p5-6 motion-mode panel (TorqueRef formula; says v_des/v_fb are the "
                       "MOTOR-end speed, i.e. before the reducer)"},
    "ACT_CONFIG_PNG": {"path": MYA / "Actuator Config.png",
                       "what": "'HyperBot (Dropbear)' actuator callout render (file date 2024-11-12): the origin of the "
                               "CSV1 joint table. Its pointer lines put 'Waist Pivot (X10-S2 1:35)' on the two large "
                               "discs at the top of the thighs (hip pitch), 'Hip Spreader (X10 1:7 V3)' on the motor "
                               "behind the pelvis (hip roll), 'Leg Rotator (X10 1:7 V3)' on the horizontal discs under "
                               "the pelvis (hip yaw), 'Knee Bender (X10-S2)' on the lower-thigh discs"},
    "CEM_MANUAL": {"path": MYA / "(CEM) Protocol and manual - 260520" / "CEM Series Product Manual 250731.pdf",
                   "what": "CEM-15 / CEM-25 / CEM-45 only; no CEM-60 datasheet exists in the corpus"},
    "CSV": {"path": MYA / "Actuators.csv", "what": "rows 2-4: X10-P35-100, X10-P7-40, X8-P6-20 spec rows"},
    "CSV1": {"path": MYA / "Actuators1.csv",
             "what": "rows 2-4 spec rows (as Actuators.csv); rows 8-18 joint -> actuator table; rows 20-28 X-V4 "
                     "candidate list (EtherCAT, not on the robot)"},
    "EMBODIMENT": {"path": H / "dropbear_control" / "integrations" / "gr00t_wbc" / "config" / "dropbear_embodiment.json",
                   "what": "Codex GR00T overlay: 'motor' per action (lines 59-296), CAN ids 0x141-0x14C for the 12 leg "
                           "motors, arms canId null; runtimeParameters all 'unverified'"},
    "BOM": {"path": H / "dropbear_bom" / "BOM.md",
            "what": "Aug 2025 BOM: RMD X10 x2 (torso), RMD 8 x4 (arms), RMD 10 x2 + CEM 60 x2 (pelvic), RMD X8 x4 + "
                    "CEM 60 x4 (legs); summary 'RMD X10 S2 4, RMD X8 Pro 12, CEM 60 6'; 5 ESP32, 4 MCP2515"},
    "MYA_README": {"path": MYA / "README.md",
                   "what": "user's integration guide (Sep 2025): 'RMD-X8-PRO-1:9 x 12, RMD-X10-1:7-V3 x 4, EPS-CEM-60 "
                           "x 6'; 5 ESP32+MCP2515 (8 MHz crystal) nodes, 1 Mbit/s, linear bus; arm IDs 21-25/31-35"},
    "DOCS_LEGS": {"path": H / "dropbear_docs" / "docs" / "03-assembly" / "legs" / "actuators.md",
                  "what": "older docs: hip pitch and knee = CEM-60 (60 N*m, 60 rpm)"},
    "DOCS_ARMS": {"path": H / "dropbear_docs" / "docs" / "03-assembly" / "arms" / "parts.md",
                  "what": "older docs: shoulder rot = RMD-X10 S2, other arm joints RMD-X8 Pro"},
    "USD_PRIMS": {"path": REPO / "logs" / "actuators" / "usd_motor_prims.log",
                  "what": "tools/probe_usd_motor_prims.py on the contract USD (sha 45586414): the MyActuator CAD "
                          "components merged into each rigid body"},
    "CALIB": {"path": REPO / "data" / "calibration" / "dropbear_semantic_calibration.json",
              "what": "semantic calibration v3: knee/elbow lut1d, ankle lut2d, standing_motor_pos, knee pivot fit"},
    "BODY_PROPS": {"path": REPO / "data" / "robot" / "usd_body_properties_45586414.json",
                   "what": "per-body mass/CoM/inertia of the contract USD (tools/extract_usd_body_properties.py)"},
    "JOINT_FRAMES": {"path": REPO / "logs" / "calibrate_settle" / "usd_joint_frames.json",
                     "what": "joint anchors/axes in the world frame at the authored rest pose (tools/dump_usd_joint_frames.py)"},
    "USD_TREE": {"path": REPO / "docs" / "usd_tree_45586414.json", "what": "joint/body dump of the contract USD"},
    "FLAT_ENV_CFG": {"path": REPO / "source" / "dropbear_wbc" / "tasks" / "locomotion" / "config" / "dropbear"
                     / "flat_env_cfg.py", "what": "ACTUATOR_PROFILES (velocity task)"},
    "DROPBEAR_NAMES": {"path": REPO / "source" / "dropbear_wbc" / "robots" / "dropbear_names.py",
                       "what": "ACTUATOR_PARAMS (legacy groups), USD_MOTOR_MAX_JOINT_VELOCITY_RAD_S"},
}


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fact(value, unit: str, src: str, page=None, note: str | None = None) -> dict:
    d = {"value": value, "unit": unit, "source": src}
    if page is not None:
        d["page"] = page
    if note:
        d["note"] = note
    return d


# ------------------------------------------------------------------------------------------------------------------
# 2. Motor models (transcribed). "value" is the adopted number; "alternatives" lists other published numbers.
# ------------------------------------------------------------------------------------------------------------------
# Characteristic curves at 48 V: rows (T [N*m], N [rpm], U [V], I [A]); I is the DC input current (U*I = Pin).
CURVE_X10_100 = [
    (0.24, 53.6, 47.99, 0.433), (0.35, 53.6, 47.99, 0.445), (0.64, 53.5, 47.99, 0.482), (1.16, 53.3, 47.99, 0.551),
    (1.93, 53.1, 47.98, 0.653), (2.94, 52.9, 47.98, 0.777), (4.17, 52.7, 47.98, 0.917), (5.57, 52.5, 47.97, 1.074),
    (7.11, 52.5, 47.96, 1.256), (8.76, 52.3, 47.95, 1.452), (10.49, 52.2, 47.95, 1.658), (12.30, 52.1, 47.94, 1.873),
    (14.14, 52.0, 47.93, 2.095), (16.02, 51.9, 47.92, 2.332), (17.93, 51.8, 47.91, 2.568), (19.87, 51.7, 47.91, 2.806),
    (21.88, 51.6, 47.90, 3.056), (23.93, 51.6, 47.89, 3.310), (25.94, 51.4, 47.88, 3.571), (27.98, 51.3, 47.87, 3.829),
    (29.99, 51.2, 47.86, 4.081), (32.00, 51.1, 47.85, 4.337), (34.02, 51.0, 47.85, 4.601), (36.00, 50.9, 47.83, 4.864),
    (37.93, 50.8, 47.83, 5.116), (39.84, 50.7, 47.82, 5.361), (41.76, 50.6, 47.81, 5.612), (43.61, 50.5, 47.80, 5.865),
    (45.40, 50.3, 47.79, 6.107), (47.12, 50.2, 47.79, 6.333), (48.76, 50.1, 47.78, 6.549), (50.39, 50.1, 47.77, 6.759),
]
CURVE_X10_40 = [
    (0.13, 186.4, 48.09, 0.515), (0.17, 186.3, 48.09, 0.530), (0.23, 186.2, 48.09, 0.555), (0.34, 186.0, 48.09, 0.598),
    (0.50, 185.6, 48.09, 0.664), (0.75, 185.1, 48.08, 0.756), (1.06, 184.4, 48.08, 0.877), (1.46, 183.6, 48.07, 1.030),
    (1.93, 182.7, 48.07, 1.216), (2.47, 181.8, 48.06, 1.434), (3.08, 180.8, 48.05, 1.680), (3.74, 179.8, 48.04, 1.952),
    (4.47, 178.8, 48.03, 2.247), (5.25, 177.9, 48.02, 2.564), (6.07, 176.9, 48.01, 2.903), (6.94, 175.9, 48.00, 3.260),
    (7.84, 174.8, 47.98, 3.631), (8.76, 173.8, 47.97, 4.012), (9.69, 172.7, 47.96, 4.399), (10.63, 171.5, 47.94, 4.792),
    (11.59, 170.4, 47.93, 5.192), (12.55, 169.2, 47.91, 5.600), (13.51, 167.9, 47.90, 6.015), (14.49, 166.7, 47.89, 6.438),
    (15.46, 165.4, 47.87, 6.863), (16.44, 164.0, 47.86, 7.286), (17.42, 162.6, 47.84, 7.702), (18.40, 161.1, 47.83, 8.113),
    (19.38, 159.5, 47.81, 8.519), (20.36, 158.0, 47.80, 8.923),
]
CURVE_X8_25 = [
    (0.04, 254.3, 48.14, 1.011), (0.10, 254.2, 48.14, 1.013), (0.19, 254.0, 48.14, 1.020), (0.31, 253.8, 48.14, 1.035),
    (0.48, 253.5, 48.14, 1.069), (0.72, 253.1, 48.13, 1.136), (1.01, 252.7, 48.12, 1.248), (1.35, 252.4, 48.11, 1.411),
    (1.75, 252.3, 48.11, 1.622), (2.19, 252.2, 48.10, 1.867), (2.68, 251.9, 48.09, 2.139), (3.21, 251.5, 48.07, 2.432),
    (3.79, 250.8, 48.06, 2.748), (4.41, 249.9, 48.05, 3.091), (5.07, 248.7, 48.03, 3.460), (5.75, 247.1, 48.02, 3.850),
    (6.46, 244.9, 48.00, 4.257), (7.19, 242.0, 47.99, 4.676), (7.93, 238.6, 47.97, 5.104), (8.69, 234.6, 47.96, 5.542),
    (9.47, 230.3, 47.94, 5.992), (10.26, 225.7, 47.92, 6.455), (11.07, 220.9, 47.90, 6.933), (11.88, 216.0, 47.89, 7.427),
    (12.69, 210.9, 47.87, 7.941), (13.52, 205.7, 47.85, 8.480), (14.35, 200.4, 47.83, 9.046), (15.19, 195.1, 47.80, 9.643),
    (16.03, 190.0, 47.78, 10.267), (16.88, 185.0, 47.76, 10.915), (17.73, 180.1, 47.73, 11.579), (18.59, 175.1, 47.71, 12.252),
]

MODELS: dict[str, dict] = {
    "RMD-X10-S2-V3-1:35": {
        "aliases": ["X10-100", "RMD-X10-P35-100-C-N (CAN)", "RMD-X10-P35-100-R-N (RS485)", "RMD-X10 S2 1:35 V3"],
        "series": "RMD-X V3 (dual encoder)",
        "gear_ratio": fact(35, "-", "X10_100", 1, "also X10S2_PROFILE p1 '35:1', CSV1 row 2"),
        "input_voltage": fact(48, "V", "X10_100", 1),
        "rated_torque": fact(50, "N*m", "X10_100", 1, "X10S2_PROFILE p1 'nominal torque 50'"),
        "peak_torque": fact(100, "N*m", "X10_100", 1, "X10S2_PROFILE p1: overload coefficient 2 x 50"),
        "rated_speed_output": fact(50, "rpm", "X10_100", 1, "X10S2_PROFILE p1 'nominal output speed 50'"),
        "no_load_speed_output": fact(53.6, "rpm", "X10_100", 1,
                                     "characteristic-curve 'No_Load' row at 47.99 V; X10S2_PROFILE p1 lists 55 rpm"),
        "rated_current": fact(6.7, "A", "X10_100", 1, "= the DC input current at 50.39 N*m in the curve table"),
        "peak_current": fact(13.5, "A", "X10_100", 1),
        "rated_power": fact(265, "W", "X10_100", 1, "thermal balance point: 24 C ambient, 60 K rise, no extra cooling"),
        "efficiency": fact(82, "%", "X10_100", 1),
        "motor_torque_constant": fact(0.32, "N*m/A (motor side)", "X10S2_PROFILE", 1,
                                      "SUSPECT: R, L, Kt, Kv and rotor inertia are identical to the X10 1:7 sheet; the "
                                      "no-load speed (53.6 rpm x 35 = 1876 motor rpm at 48 V, ~39 rpm/V) contradicts "
                                      "Kv 30 rpm/V"),
        "motor_speed_constant": fact(30, "rpm/V (motor side)", "X10S2_PROFILE", 1, "see motor_torque_constant note"),
        "wire_resistance": fact(0.3, "ohm", "X10S2_PROFILE", 1),
        "wire_inductance": fact(0.13, "mH", "X10S2_PROFILE", 1),
        "output_torque_per_input_amp": fact(round(50.39 / 6.759, 3), "N*m/A (output, DC input current)", "X10_100", 1,
                                            "derived from the curve's max-torque row"),
        "rotor_inertia": fact(5675, "g*cm^2 (motor side)", "X10S2_PROFILE", 1,
                              "catalog 'Inertia 198.6 kg*cm^2' (X10_100 p1, CSV1 row 2) = 5675 g*cm^2 x 35 / 1000, i.e. "
                              "the vendor multiplied the rotor inertia by the ratio once, not squared"),
        "catalog_inertia_field": fact(198.6, "kg*cm^2 (vendor field; = rotor inertia x ratio)", "X10_100", 1),
        "back_drive_torque": fact(2.88, "N*m (output)", "X10_100", 1),
        "backlash": fact(15, "arcmin", "X10_100", 1, "X10S2_PROFILE p1: 15"),
        "weight": fact(1700, "g", "X10_100", 1),
        "pole_pairs": fact(21, "-", "X10_100", 1),
        "encoder": fact("dual: input 16 bit / output 14 bit", "-", "X10_100", 1,
                        "XV3MAN_241227 p6 and CSV1 row 2 list 14/14 bit"),
        "axial_radial_payload": fact([1625, 2250], "N", "X10_100", 1),
        "communication": fact("CAN 1 Mbit/s (500 kbit/s selectable); RS485 115200/500K/1M/2.5M", "-", "X10_100", 1,
                              "X10S2_PROFILE p1: CAN 500 kbit/s / 1 Mbit/s"),
        "control_modes": fact("servo mode (torque/velocity/position); motion mode (feedforward torque/velocity/"
                              "position)", "-", "X10S2_PROFILE", 1),
        "characteristic_curve_48V": {"source": "X10_100", "page": 1, "columns": ["T_Nm", "N_rpm", "U_V", "I_A"],
                                     "rows": CURVE_X10_100},
    },
    "RMD-X10-V3-1:7": {
        "aliases": ["X10-40", "RMD-X10-P7-40-C-N", "RMD-X10 1:7 V3"],
        "series": "RMD-X V3 (dual encoder)",
        "gear_ratio": fact(7, "-", "X10_40", 1, "X10_PROFILE p1 '7:1', CSV1 row 3"),
        "input_voltage": fact(48, "V", "X10_40", 1),
        "rated_torque": fact(15, "N*m", "X10_40", 1, "older X10_PROFILE p1: 12 N*m"),
        "peak_torque": fact(40, "N*m", "X10_40", 1, "older X10_PROFILE p1: overload coefficient 3 x 12 = 36 N*m"),
        "rated_speed_output": fact(165, "rpm", "X10_40", 1, "older X10_PROFILE p1: 170 rpm"),
        "no_load_speed_output": fact(186.4, "rpm", "X10_40", 1, "curve 'No_Load' row at 48.09 V; X10_PROFILE p1: 190"),
        "rated_current": fact(6.5, "A", "X10_40", 1, "older X10_PROFILE p1: 5.3 A"),
        "peak_current": fact(15, "A", "X10_40", 1),
        "rated_power": fact(265, "W", "X10_40", 1, "older X10_PROFILE p1: 215 W"),
        "efficiency": fact(82, "%", "X10_40", 1, "X10_PROFILE p1: 82.5"),
        "motor_torque_constant": fact(0.32, "N*m/A (motor side)", "X10_PROFILE", 1),
        "motor_speed_constant": fact(30, "rpm/V (motor side)", "X10_PROFILE", 1,
                                     "consistent with the curve: 186.4 rpm x 7 = 1305 motor rpm at 48 V (27 rpm/V)"),
        "wire_resistance": fact(0.3, "ohm", "X10_PROFILE", 1),
        "wire_inductance": fact(0.13, "mH", "X10_PROFILE", 1),
        "output_torque_per_input_amp": fact(round(20.36 / 8.923, 3), "N*m/A (output, DC input current)", "X10_40", 1),
        "rotor_inertia": fact(5675, "g*cm^2 (motor side)", "X10_PROFILE", 1,
                              "catalog 'Inertia 39.7 kg*cm^2' (X10_40 p1) = 5675 x 7 / 1000"),
        "catalog_inertia_field": fact(39.7, "kg*cm^2 (vendor field; = rotor inertia x ratio)", "X10_40", 1),
        "back_drive_torque": fact(0.62, "N*m (output)", "X10_40", 1),
        "backlash": fact(10, "arcmin", "X10_40", 1, "X10_PROFILE p1: 7; SIMPLE_PROFILE p3: 8"),
        "weight": fact(1150, "g", "X10_40", 1),
        "pole_pairs": fact(21, "-", "X10_40", 1),
        "encoder": fact("dual: input 16 bit / output 14 bit", "-", "X10_40", 1, "XV3MAN_241227 p5: 14/14 bit"),
        "axial_radial_payload": fact([1625, 2250], "N", "X10_40", 1),
        "communication": fact("CAN 1 Mbit/s; RS485 115200/500K/1M/2.5M", "-", "X10_40", 1),
        "control_modes": fact("servo mode (torque/velocity/position); motion mode", "-", "X10_PROFILE", 1),
        "characteristic_curve_48V": {"source": "X10_40", "page": 1, "columns": ["T_Nm", "N_rpm", "U_V", "I_A"],
                                     "rows": CURVE_X10_40},
    },
    "RMD-X8-Pro-V2-1:9": {
        "aliases": ["X8-25", "RMD-X8-P9-25-C-N", "RMD-X8-Pro 1:9 V2", "RMD-X8 PRO 1:9 (CSV1 joint table)"],
        "series": "RMD-X V2 (single encoder)",
        "gear_ratio": fact(9, "-", "X8_25", 1, "CSV1 rows 9-13, 18; SIMPLE_PROFILE p1 (9:01 time cell)"),
        "input_voltage": fact(48, "V", "X8_25", 1, "SIMPLE_PROFILE p1: 24-48 V"),
        "rated_torque": fact(10, "N*m", "X8_25", 1, "SIMPLE_PROFILE p1 'RMD-X8 PRO': nominal torque 13 N*m"),
        "peak_torque": fact(25, "N*m", "X8_25", 1, "the only 1:9 Pro source with a peak value"),
        "rated_speed_output": fact(110, "rpm", "X8_25", 1, "SIMPLE_PROFILE p1: nominal speed 122 rpm"),
        "no_load_speed_output": fact(160, "rpm", "SIMPLE_PROFILE", 1,
                                     "CONSERVATIVE choice. X8_25 p1 curve: 254.3 rpm at 48.14 V. Same weight (710 g) "
                                     "and rotor inertia (3400 g*cm^2), different winding (Kv 30 vs ~47 rpm/V). Which "
                                     "generation is on the robot is an open question (docs/ACTUATORS.md Q3)"),
        "no_load_speed_output_x8_25_curve": fact(254.3, "rpm", "X8_25", 1),
        "rated_current": fact(3.2, "A", "X8_25", 1,
                              "NOT the DC current (the curve draws ~6.5 A at 10 N*m); SIMPLE_PROFILE p1: 5 A"),
        "peak_current": fact(8, "A", "X8_25", 1),
        "rated_power": fact(125, "W", "X8_25", 1, "SIMPLE_PROFILE p1: 166 W"),
        "efficiency": fact(80, "%", "X8_25", 1, "SIMPLE_PROFILE p1: 0.77"),
        "output_torque_constant": fact(2.6, "N*m/A (output)", "SIMPLE_PROFILE", 1),
        "motor_speed_constant": fact(30, "rpm/V (motor side)", "SIMPLE_PROFILE", 1),
        "wire_resistance": fact(0.54, "ohm", "SIMPLE_PROFILE", 1),
        "phase_inductance": fact(0.28, "mH (phase to phase)", "SIMPLE_PROFILE", 1),
        "rotor_inertia": fact(3400, "g*cm^2 (motor side)", "SIMPLE_PROFILE", 1,
                              "catalog 'Inertia 30.6 kg*cm^2' (X8_25 p1) = 3400 x 9 / 1000"),
        "catalog_inertia_field": fact(30.6, "kg*cm^2 (vendor field; = rotor inertia x ratio)", "X8_25", 1),
        "back_drive_torque": fact(0.61, "N*m (output)", "X8_25", 1),
        "backlash": fact(10, "arcmin", "X8_25", 1),
        "weight": fact(710, "g", "X8_25", 1, "SIMPLE_PROFILE p1: 710 g"),
        "pole_pairs": fact(21, "-", "X8_25", 1, "SIMPLE_PROFILE p1: 20"),
        "encoder": fact("single, motor side only: 16 bit (240403) / 18 bit (241227); NO output encoder", "-", "X8_25", 1,
                        "XV2MAN_241227 p5 lists 18 bit; the output angle is motor counts / 9"),
        "axial_radial_payload": fact([985, 1250], "N", "X8_25", 1),
        "communication": fact("CAN 1 Mbit/s; RS485 115200/500K/1M/2.5M", "-", "X8_25", 1),
        "characteristic_curve_48V": {"source": "X8_25", "page": 1, "columns": ["T_Nm", "N_rpm", "U_V", "I_A"],
                                     "rows": CURVE_X8_25,
                                     "note": "concave: 1.7 rpm/(N*m) near 0, 6 rpm/(N*m) near 18 N*m"},
        "siblings_not_used": {
            "RMD-X8 V2 1:9 (non-Pro, A00083)": {
                "source": "X8V2_PROFILE", "page": 1,
                "values": {"rated_torque_Nm": 9, "no_load_rpm_48V": 215, "nominal_rpm": 170, "nominal_current_A": 4.3,
                           "power_W": 160, "efficiency_pct": 74, "back_drive_Nm": 0.31, "pole_pairs": 21,
                           "resistance_ohm": 0.4, "inductance_mH": 0.22, "Kv_rpm_per_V": 40,
                           "Kt_output_Nm_per_A": 2.09, "rotor_inertia_gcm2": 2600, "weight_g": 568,
                           "backlash_arcmin": 8},
                "why_not": "568 g and no 'Pro' in the name; the USD components and CSV1 say X8 Pro"},
            "RMD-X8-Pro-H 1:6 V3 (X8-20, RMD-X8-P6-20-C-N)": {
                "source": "XMAN_240403", "page": 10,
                "values": {"gear_ratio": 6, "rated_torque_Nm": 10, "peak_torque_Nm": 20, "rated_rpm": 190,
                           "rated_current_A": 5.2, "peak_current_A": 10.5, "power_W": 200, "efficiency_pct": 80,
                           "catalog_inertia_kgcm2": 20, "rotor_inertia_gcm2_derived": round(20 * 1000 / 6),
                           "back_drive_Nm": 0.4, "backlash_arcmin": 10, "pole_pairs": 20, "weight_g": 780,
                           "encoder": "dual 16/14 bit"},
                "why_not": "it is the spec row of Actuators.csv/Actuators1.csv row 4, but the joint table (rows 9-18), "
                           "the model folder X8-25 and DECISIONS all say 1:9 (docs/ACTUATORS.md Q2)"},
        },
    },
}

# Conservative X8 torque-speed line from the SIMPLE_PROFILE nominal point (13 N*m at 122 rpm, 160 rpm no-load).
X8_CONSERVATIVE_LINE = {"no_load_rpm": 160.0, "point": (13.0, 122.0), "source": "SIMPLE_PROFILE", "page": 1}

# ------------------------------------------------------------------------------------------------------------------
# 3. Joint -> model map with evidence.
# ------------------------------------------------------------------------------------------------------------------
ROLE = {  # docs/CONTRACTS.md section 1 + calibration
    "PG_{s}_leg_pitch": "hip roll", "PG_{s}_leg_roll": "hip yaw", "{S}L_hip_joint": "hip pitch",
    "{S}L_knee_actuator_joint": "knee crank (four-bar input)", "{S}L_Revolute67": "calf motor A (parallel ankle)",
    "{S}L_Revolute81": "calf motor B (parallel ankle)", "{S}H_yaw": "shoulder pitch", "{S}H_pitch": "shoulder abduction",
    "{S}H_roll": "upper-arm roll", "{S}H_elbow_joint": "elbow (linkage input)", "{S}H_wrist_roll": "wrist roll",
}

JOINT_EVIDENCE: dict[str, dict] = {
    "hip roll": {
        "model": "RMD-X10-S2-V3-1:35", "confidence": "high",
        "evidence": [
            {"source": "USD_PRIMS", "says": "RMD-X10-S2", "detail": "pelvis 'world' holds PG_RMD_X10_S2_MIR4__2_Stator_1 "
             "(right __1); the child body IS PG_RMD_X10_S2_MIR4__2_Rotor_1", "agrees": True},
            {"source": "EMBODIMENT", "says": "RMD-X10-S2", "detail": "lines 162/174 (left/right_hip_roll)", "agrees": True},
            {"source": "CSV1", "says": "RMD-X10 1:7 V3", "detail": "row 14 'Hip Spreader'", "agrees": False},
            {"source": "ACT_CONFIG_PNG", "says": "RMD-X10 1:7 V3", "detail": "'Hip Spreader: actuator on the back butt "
             "area' (the render the CSV was typed from; file date 2024-11)", "agrees": False},
            {"source": "BOM", "says": "RMD 10 / CEM 60 (pelvic)", "detail": "Aug 2025, pre-CAD plan", "agrees": False},
        ],
        "resolution": "CAD (USD of 2025-10) wins over the 2024 render/CSV: the hip-roll stator/rotor pair is an X10-S2, "
                      "and the embodiment file agrees. Physics agrees too: single-support hip-roll moment ~557 N x "
                      "~0.1 m ~ 56 N*m exceeds the X10 1:7 peak (40). OPEN (Q5): if the robot really has an X10 1:7 "
                      "there, use peak 40 N*m, armature 0.0278, no-load 19.5 rad/s for these two joints.",
    },
    "hip yaw": {
        "model": "RMD-X10-V3-1:7", "confidence": "high",
        "evidence": [
            {"source": "USD_PRIMS", "says": "RMD-X10 V3", "detail": "hip-roll output body holds PG_RMD_X10_V3Stator_1; "
             "the child body IS PG_RMD_X10_V3Rotor_1", "agrees": True},
            {"source": "CSV1", "says": "RMD-X10 1:7 V3", "detail": "row 15 'Leg Rotator'", "agrees": True},
            {"source": "ACT_CONFIG_PNG", "says": "RMD-X10 1:7 V3", "detail": "'Leg Rotator' on the horizontal discs "
             "under the pelvis", "agrees": True},
            {"source": "MYA_README", "says": "RMD-X10-1:7-V3 x 4", "detail": "= 2 shoulders + 2 hip yaws", "agrees": True},
            {"source": "EMBODIMENT", "says": "RMD-X10-S2", "detail": "lines 150/186 (hip yaw and hip pitch look swapped "
             "there)", "agrees": False},
        ],
        "resolution": "X10 1:7 V3 (CAD + CSV + README count).",
    },
    "hip pitch": {
        "model": "RMD-X10-S2-V3-1:35", "confidence": "high (CAD); the CSV row name is the open point",
        "evidence": [
            {"source": "USD_PRIMS", "says": "RMD-X10-S2", "detail": "the hip-yaw output body holds "
             "LL_RMD_X10_S2_MIR4__3_Rotor_1 and the child (thigh) IS LL_RMD_X10_S2_MIR4__3_Stator_1, a second X10-S2 "
             "besides the knee's", "agrees": True},
            {"source": "CSV1", "says": "RMD-X10-S2 1:35", "detail": "row 17 'Waist Pivot: torso forward/back bending' is "
             "the only other X10-S2 row; the USD has no waist joint, and hip pitch is what bends the torso vs. the "
             "legs", "agrees": True},
            {"source": "ACT_CONFIG_PNG", "says": "RMD-X10-S2 1:35", "detail": "the 'Waist Pivot' pointer lines end on the "
             "two large discs at the top of the thighs, i.e. the hip-pitch motors", "agrees": True},
            {"source": "EMBODIMENT", "says": "RMD-X10 (ratio unspecified)", "detail": "lines 116/128", "agrees": False},
            {"source": "DOCS_LEGS", "says": "CEM-60", "detail": "older design; no CEM-60 datasheet in the corpus",
             "agrees": False},
        ],
        "resolution": "X10-S2 1:35 (CAD). Confirm that CSV 'Waist Pivot' means the two hip-pitch motors (Q1).",
    },
    "knee crank (four-bar input)": {
        "model": "RMD-X10-S2-V3-1:35", "confidence": "high",
        "evidence": [
            {"source": "USD_PRIMS", "says": "RMD-X10-S2", "detail": "thigh holds LL_RMD_X10_S2_MIR4Stator_1; child IS "
             "LL_RMD_X10_S2_MIR4Rotor_1", "agrees": True},
            {"source": "CSV1", "says": "RMD-X10-S2 1:35", "detail": "row 16 'Knee Bender, via 4-bar'", "agrees": True},
            {"source": "EMBODIMENT", "says": "RMD-X10-S2", "detail": "lines 104/138", "agrees": True},
            {"source": "ACT_CONFIG_PNG", "says": "RMD-X10-S2 1:35", "detail": "'Knee Bender: actuator on the back thigh "
             "triggering the 4-bar linkage'", "agrees": True},
            {"source": "DOCS_LEGS", "says": "CEM-60", "detail": "older design; CEM_MANUAL has no CEM-60", "agrees": False},
        ],
        "resolution": "X10-S2 1:35 driving the crank directly (as DECISIONS 2026-09-24 assumed).",
    },
    "calf motor": {
        "model": "RMD-X8-Pro-V2-1:9", "confidence": "high for X8 Pro; medium for the 1:9 variant",
        "evidence": [
            {"source": "USD_PRIMS", "says": "RMD-X8 Pro", "detail": "shank holds LL_RMD_X8_Pro_MIR8_MIR1_1 and _2",
             "agrees": True},
            {"source": "CSV1", "says": "RMD-X8 PRO 1:9", "detail": "row 18 'Calf-Foot Flexor, via tie rods'",
             "agrees": True},
            {"source": "ACT_CONFIG_PNG", "says": "RMD-X8 PRO 1:9", "detail": "'Calf-foot Flexor'", "agrees": True},
            {"source": "EMBODIMENT", "says": "RMD-X8", "detail": "lines 59-95", "agrees": True},
            {"source": "BOM", "says": "RMD X8 x4 (legs)", "detail": "", "agrees": True},
            {"source": "CSV1", "says": "RMD-X8 Pro 1:6 (X8-20)", "detail": "row 4 spec row", "agrees": False},
        ],
        "resolution": "X8 Pro 1:9 (X8-25 generation name). Q2/Q3 ask for the label.",
    },
    "shoulder pitch": {
        "model": "RMD-X10-V3-1:7", "confidence": "medium-high",
        "evidence": [
            {"source": "USD_PRIMS", "says": "RMD-X10 (no S2/V3 in the component name)", "detail": "world holds "
             "torso_RMD_X10Stator_1; child IS torso_RMD_X10__1_Rotor_1", "agrees": True},
            {"source": "CSV1", "says": "RMD-X10 1:7 V3", "detail": "row 12 'Shoulder Rotator'", "agrees": True},
            {"source": "ACT_CONFIG_PNG", "says": "RMD-X10 1:7 V3", "detail": "'Shoulder Rotator'", "agrees": True},
            {"source": "BOM", "says": "RMD X10 x2 (torso)", "detail": "", "agrees": True},
            {"source": "MYA_README", "says": "RMD-X10-1:7-V3 x 4", "detail": "", "agrees": True},
            {"source": "EMBODIMENT", "says": "RMD-X10", "detail": "lines 194/250", "agrees": True},
            {"source": "DOCS_ARMS", "says": "RMD-X10 S2", "detail": "older docs, 'Shoulder Rot'", "agrees": False},
        ],
        "resolution": "X10 1:7 V3 (Q4 asks to confirm it is not an X10-S2).",
    },
    "arm X8": {
        "model": "RMD-X8-Pro-V2-1:9", "confidence": "high for X8 Pro; medium for 1:9",
        "evidence": [
            {"source": "USD_PRIMS", "says": "RMD-X8 Pro", "detail": "4 per arm: LH_RMD_X8_Pro_MIR8_MIR1__1__1 (shoulder "
             "abduction child), __3__1 (roll child), __2__1 inside the upper arm (elbow), __4__1 inside the forearm "
             "(wrist)", "agrees": True},
            {"source": "CSV1", "says": "RMD-X8 PRO 1:9", "detail": "rows 9-11, 13", "agrees": True},
            {"source": "ACT_CONFIG_PNG", "says": "RMD-X8 PRO 1:9", "detail": "Arm Twister, Shoulder Extender, Elbow "
             "Bender, Wrist Articulator", "agrees": True},
            {"source": "EMBODIMENT", "says": "RMD-X8", "detail": "", "agrees": True},
        ],
        "resolution": "X8 Pro 1:9.",
    },
}


def joint_role(name: str) -> tuple[str, str]:
    side = "left" if name.startswith(("PG_left", "LL_", "LH_")) else "right"
    for pat, role in ROLE.items():
        s = "left" if side == "left" else "right"
        S = "L" if side == "left" else "R"
        if pat.format(s=s, S=S) == name:
            return side, role
    raise KeyError(name)


def evidence_key(role: str) -> str:
    if role.startswith("calf motor"):
        return "calf motor"
    if role in ("shoulder abduction", "upper-arm roll", "elbow (linkage input)", "wrist roll"):
        return "arm X8"
    return role


# ------------------------------------------------------------------------------------------------------------------
# 4. Geometry helpers.
# ------------------------------------------------------------------------------------------------------------------
def quat_to_mat(q_wxyz) -> np.ndarray:
    w, x, y, z = q_wxyz
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class Plant:
    def __init__(self) -> None:
        self.tree = json.loads(SOURCES["USD_TREE"]["path"].read_text(encoding="utf-8"))
        self.frames = json.loads(SOURCES["JOINT_FRAMES"]["path"].read_text(encoding="utf-8"))
        props = json.loads(SOURCES["BODY_PROPS"]["path"].read_text(encoding="utf-8"))
        self.bodies = {}
        for name, b in props["bodies"].items():
            R = quat_to_mat(b["rest_quat_w_wxyz"])
            com_w = np.array(b["rest_pos_w"]) + R @ np.array(b["com_b"])
            I_w = R @ np.array(b["inertia_b"]) @ R.T
            self.bodies[name] = (float(b["mass"]), com_w, I_w)
        adj: dict[str, list[tuple[str, str]]] = {}
        for j in self.tree["joints"]:
            if j["excl"] or not j["b0"] or not j["b1"]:
                continue
            a, b = j["b0"][0], j["b1"][0]
            adj.setdefault(a, []).append((b, j["name"]))
            adj.setdefault(b, []).append((a, j["name"]))
        self.adj = adj
        depth = {"world": 0}
        dq = deque(["world"])
        while dq:
            u = dq.popleft()
            for v, _ in adj.get(u, []):
                if v not in depth:
                    depth[v] = depth[u] + 1
                    dq.append(v)
        self.depth = depth

    def joint_axis(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        j = self.frames["joints"][name]
        a = np.array(j["axis0_w"], float)
        return np.array(j["anchor0_w"], float), a / np.linalg.norm(a)

    def subtree(self, joint: str) -> list[str]:
        j = next(x for x in self.tree["joints"] if x["name"] == joint)
        a, b = j["b0"][0], j["b1"][0]
        child, parent = (b, a) if self.depth.get(b, 1e9) > self.depth.get(a, 1e9) else (a, b)
        seen = {parent, child}
        dq = deque([child])
        while dq:
            u = dq.popleft()
            for v, _ in self.adj.get(u, []):
                if v not in seen:
                    seen.add(v)
                    dq.append(v)
        seen.discard(parent)
        return sorted(seen)

    def inertia_about(self, bodies, point, axis) -> float:
        total = 0.0
        for n in bodies:
            m, c, I = self.bodies[n]
            r = c - point
            r_perp = r - (r @ axis) * axis
            total += float(axis @ I @ axis) + m * float(r_perp @ r_perp)
        return total


# ------------------------------------------------------------------------------------------------------------------
# 5. Calibration transmissions.
# ------------------------------------------------------------------------------------------------------------------
def load_calibration() -> dict:
    return json.loads(SOURCES["CALIB"]["path"].read_text(encoding="utf-8"))


def lut1d_slope(cal: dict, dof: str, q: float) -> dict:
    d = cal["dofs"][dof]
    g = np.array(d["motor_grid"], float)
    s = np.array(d["semantic_values"], float)
    sl = np.gradient(s, g)
    return {"motor": d["motors"][0], "at_stand": float(np.interp(q, g, sl)), "min": float(sl.min()),
            "max": float(sl.max()), "mean": float(d["info"].get("mean_gain", sl.mean())),
            "abs_min": float(np.abs(sl).min()), "abs_max": float(np.abs(sl).max())}


def ankle_jacobian(cal: dict, side: str, stand: dict) -> dict:
    ap = cal["ankle_pairs"][side]
    a = np.array(ap["a_grid"], float)
    b = np.array(ap["b_grid"], float)
    P = np.array(ap["pitch"], float)
    R = np.array(ap["roll"], float)
    ma, mb = ap["motors"]
    qa, qb = stand[ma], stand[mb]
    grads = [np.gradient(P, a, b), np.gradient(R, a, b)]

    def bil(M):
        i = int(np.clip(np.searchsorted(a, qa) - 1, 0, len(a) - 2))
        k = int(np.clip(np.searchsorted(b, qb) - 1, 0, len(b) - 2))
        ta = (qa - a[i]) / (a[i + 1] - a[i])
        tb = (qb - b[k]) / (b[k + 1] - b[k])
        return float((1 - ta) * (1 - tb) * M[i, k] + ta * (1 - tb) * M[i + 1, k] + (1 - ta) * tb * M[i, k + 1]
                     + ta * tb * M[i + 1, k + 1])

    J = np.array([[bil(grads[0][0]), bil(grads[0][1])], [bil(grads[1][0]), bil(grads[1][1])]])
    return {"motors": [ma, mb], "J_pitchroll_wrt_motors": J.round(4).tolist(), "J": J}


# ------------------------------------------------------------------------------------------------------------------
# 6. Sim parameters.
# ------------------------------------------------------------------------------------------------------------------
def load_profiles() -> dict:
    tree = ast.parse(SOURCES["FLAT_ENV_CFG"]["path"].read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "ACTUATOR_PROFILES":
            return ast.literal_eval(node.value)
    raise RuntimeError("ACTUATOR_PROFILES not found")


# ------------------------------------------------------------------------------------------------------------------
# 7. Derived per-model quantities.
# ------------------------------------------------------------------------------------------------------------------
def torque_speed_line(model_id: str, conservative_x8: bool = True) -> dict:
    """Straight DC-motor line through the curve's no-load row and its max-torque row (48 V)."""
    m = MODELS[model_id]
    if model_id.startswith("RMD-X8") and conservative_x8:
        n0 = X8_CONSERVATIVE_LINE["no_load_rpm"]
        t1, n1 = X8_CONSERVATIVE_LINE["point"]
        slope = (n0 - n1) / t1
        basis = "SIMPLE_PROFILE p1: 160 rpm no-load, 13 N*m at 122 rpm (conservative X8 PRO line)"
    else:
        rows = m["characteristic_curve_48V"]["rows"]
        (ta, na, _, _), (tb, nb, _, _) = rows[0], rows[-1]
        slope = (na - nb) / (tb - ta)
        n0 = na + ta * slope
        basis = f"{m['characteristic_curve_48V']['source']} p1 curve: chord from the No_Load row to the Max_Torque row"
    tau_sat = n0 / slope
    w_nl = n0 * RPM
    peak = m["peak_torque"]["value"]
    rated = m["rated_torque"]["value"]
    w_at_peak = (n0 - slope * peak) * RPM
    w_at_rated = (n0 - slope * rated) * RPM
    pts = [[0.0, peak], [round(w_at_peak, 4), peak], [round(w_nl, 4), 0.0]]
    return {"basis": basis, "no_load_speed_rad_s": round(w_nl, 4), "slope_rpm_per_Nm": round(slope, 5),
            "saturation_effort_Nm": round(tau_sat, 2), "speed_at_peak_torque_rad_s": round(w_at_peak, 4),
            "speed_at_rated_torque_rad_s": round(w_at_rated, 4),
            "envelope_points_rad_s_Nm": pts,
            "formula": "tau_max(w) = min(peak, saturation_effort * (1 - |w| / no_load_speed)) in the driving quadrant "
                       "(Isaac Lab DCMotorCfg: effort_limit=peak, saturation_effort, velocity_limit=no_load_speed)"}


def model_derived(model_id: str) -> dict:
    m = MODELS[model_id]
    N = m["gear_ratio"]["value"]
    J_rot = m["rotor_inertia"]["value"] * G_CM2
    out = {"armature_kgm2": round(J_rot * N * N, 6),
           "armature_formula": f"rotor inertia {m['rotor_inertia']['value']} g*cm^2 x {N}^2 (gear-stage inertias "
                               "not published; add ~10-30 % in randomization)",
           "armature_if_catalog_field_were_output_inertia_kgm2": round(m["catalog_inertia_field"]["value"] * 1e-4, 6),
           "torque_speed_line": torque_speed_line(model_id)}
    if model_id.startswith("RMD-X8"):
        out["torque_speed_line_x8_25_curve"] = torque_speed_line(model_id, conservative_x8=False)
    return out


# ------------------------------------------------------------------------------------------------------------------
# 8. CAN timing.
# ------------------------------------------------------------------------------------------------------------------
BITS_8B_MIN = 111   # standard 11-bit ID, DLC 8, no stuff bits, incl. 3-bit intermission
BITS_8B_TYP = 120   # ~ random payload
BITS_8B_MAX = 135   # worst-case stuffing (24 stuff bits)

BUS_LAYOUTS = {
    "L1_per_limb_5_buses": {
        "basis": "MYA_README 'Distributed control architecture': 5 ESP32+MCP2515 nodes (left arm, right arm, pelvis, "
                 "right leg, left leg), 1 Mbit/s linear bus. Mapped onto the 22 USD motors; hip pitch assumed on the "
                 "leg bus because its stator rides on the thigh (INFERRED)",
        "buses": {"left_arm": 5, "right_arm": 5, "pelvis (hip roll+yaw x2)": 4,
                  "left_leg (hip pitch, knee, 2 calf)": 4, "right_leg": 4}},
    "L2_bom_4_buses": {
        "basis": "BOM: 4 MCP2515 (2 arms, 1 pelvic, 1 legs), 5 ESP32 (the 5th drives the neck steppers)",
        "buses": {"left_arm": 5, "right_arm": 5, "pelvis": 4, "legs": 8}},
    "L3_codex_ids": {
        "basis": "EMBODIMENT: leg motors on CAN ids 0x141-0x14C (unique 1-12, one bus); MYA_README sketch: the 10 arm "
                 "motors (ids 21-25, 31-35) on one ESP32",
        "buses": {"legs": 12, "arms": 10}},
    "L4_single_bus": {"basis": "worst case", "buses": {"all": 22}},
}


def can_timing() -> dict:
    res = {}
    for key, lay in BUS_LAYOUTS.items():
        buses = {}
        for bus, n in lay["buses"].items():
            frames = 2 * n  # one command + one reply per motor per cycle (0x140/0x240 or 0x400/0x500)
            t = {k: frames * bits * 1e-6 for k, bits in (("min", BITS_8B_MIN), ("typ", BITS_8B_TYP), ("max", BITS_8B_MAX))}
            buses[bus] = {
                "motors": n, "frames_per_cycle": frames,
                "cycle_time_ms_min_typ_max": [round(t["min"] * 1e3, 3), round(t["typ"] * 1e3, 3), round(t["max"] * 1e3, 3)],
                "max_rate_hz_at_100pct_load_worstcase": round(1.0 / t["max"], 1),
                "max_rate_hz_at_70pct_load_worstcase": round(0.7 / t["max"], 1),
                "bus_load_pct_worstcase": {f"{f}Hz": round(100 * t["max"] * f, 1) for f in (50, 100, 200, 500, 1000)},
            }
        res[key] = {"basis": lay["basis"], "buses": buses,
                    "limiting_bus_max_rate_hz_70pct": min(b["max_rate_hz_at_70pct_load_worstcase"] for b in buses.values())}
    return res


# ------------------------------------------------------------------------------------------------------------------
# 9. Motion analysis.
# ------------------------------------------------------------------------------------------------------------------
def motion_analysis(gates: dict[str, float], caps: dict[str, float]) -> dict:
    motors = list(dn.MOTOR_NAMES)
    out = {"gates_rad_s": gates, "hard_caps_rad_s": caps}
    cat = json.loads((REPO / "data" / "motions" / "catalog.json").read_text(encoding="utf-8"))
    clips = [c for c in cat["clips"] if c.get("status") == "accepted" and c.get("npz")]
    per_clip = []
    for c in clips:
        p = REPO / c["npz"]
        if not p.exists():
            continue
        z = np.load(p, allow_pickle=True)
        names = [str(x) for x in z["joint_names"]]
        idx = [names.index(m) for m in motors]
        v = np.abs(np.asarray(z["joint_vel"])[:, idx])
        mx = v.max(axis=0)
        over = [m for i, m in enumerate(motors) if mx[i] > gates[m]]
        over_cap = [m for i, m in enumerate(motors) if mx[i] > caps[m]]
        frac = float(np.mean(np.any(v > np.array([gates[m] for m in motors]), axis=1)))
        per_clip.append({"clip": c["clip"], "source": c.get("source"), "frames": int(v.shape[0]),
                         "max_abs_motor_vel": {m: round(float(mx[i]), 2) for i, m in enumerate(motors)},
                         "motors_over_gate": over, "motors_over_no_load_speed": over_cap,
                         "frac_frames_any_motor_over_gate": round(frac, 3)})
    n = len(per_clip)
    by_motor = {m: sum(1 for pc in per_clip if m in pc["motors_over_gate"]) for m in motors}
    out["reference_clips"] = {
        "catalog": "data/motions/catalog.json (status accepted, v4 NPZs)", "n_clips": n,
        "clips_passing_all_gates": sum(1 for pc in per_clip if not pc["motors_over_gate"]),
        "clips_over_gate_per_motor": by_motor,
        "clips_over_no_load_speed_any_motor": sum(1 for pc in per_clip if pc["motors_over_no_load_speed"]),
        "max_over_clips_per_motor": {m: round(max(pc["max_abs_motor_vel"][m] for pc in per_clip), 2) for m in motors}
        if per_clip else {},
        "per_clip": per_clip,
    }
    rolls = []
    for f in sorted(glob.glob(str(REPO / "logs" / "brev" / "remote" / "*" / "logs" / "brev" / "eval" / "rollout_*.npz"))):
        z = np.load(f, allow_pickle=True)
        names = [str(x) for x in z["joint_names"]]
        idx = [names.index(m) for m in motors]
        v = np.abs(np.asarray(z["joint_vel"])[:, idx])
        meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
        g = np.array([gates[m] for m in motors])
        rolls.append({"file": str(Path(f).relative_to(REPO)).replace("\\", "/"),
                      "checkpoint": meta.get("checkpoint", ""), "frames": int(v.shape[0]),
                      "falls_env0": len(meta.get("falls_env0", [])),
                      "p99_abs_motor_vel": {m: round(float(np.percentile(v[:, i], 99)), 2) for i, m in enumerate(motors)},
                      "max_abs_motor_vel": {m: round(float(v[:, i].max()), 2) for i, m in enumerate(motors)},
                      "frac_frames_over_gate": {m: round(float(np.mean(v[:, i] > g[i])), 3) for i, m in enumerate(motors)}})
    out["policy_rollouts"] = rolls
    return out


# ------------------------------------------------------------------------------------------------------------------
# 10. Main.
# ------------------------------------------------------------------------------------------------------------------
MIT_LIMITS_V39 = {"p_des_rad": [-12.5, 12.5], "v_des_rad_s": [-45, 45], "t_ff_Nm": [-24, 24], "kp": [0, 500],
                  "kd": [0, 5], "bits": {"p": 16, "v": 12, "kp": 12, "kd": 12, "t": 12}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(REPO / "data" / "robot" / "actuators_datasheet_v1.json"))
    ap.add_argument("--motions", action="store_true", help="also analyse reference clips and policy rollouts")
    args = ap.parse_args()

    plant = Plant()
    cal = load_calibration()
    stand = dict(zip(cal["motor_names"], cal["standing_motor_pos"]))
    profiles = load_profiles()
    usd_vcap = dn.USD_MOTOR_MAX_JOINT_VELOCITY_RAD_S

    derived = {mid: model_derived(mid) for mid in MODELS}

    # transmissions at the standing pose
    knee = {s: lut1d_slope(cal, f"{s}_knee", stand[f"{'L' if s == 'left' else 'R'}L_knee_actuator_joint"])
            for s in ("left", "right")}
    elbow = {s: lut1d_slope(cal, f"{s}_elbow", stand[f"{'L' if s == 'left' else 'R'}H_elbow_joint"])
             for s in ("left", "right")}
    ankle = {s: ankle_jacobian(cal, s, stand) for s in ("left", "right")}

    joints_out = {}
    gates, caps = {}, {}
    for name in dn.MOTOR_NAMES:
        side, role = joint_role(name)
        ev = JOINT_EVIDENCE[evidence_key(role)]
        mid = ev["model"]
        m = MODELS[mid]
        d = derived[mid]
        N = m["gear_ratio"]["value"]
        peak = m["peak_torque"]["value"]
        rated = m["rated_torque"]["value"]
        tsl = d["torque_speed_line"]
        arm = d["armature_kgm2"]
        S = "L" if side == "left" else "R"

        # --- link inertia seen by the motor at the authored rest pose -----------------------------------------------
        anchor, axis = plant.joint_axis(name)
        transmission = {"type": "serial (direct drive of the joint)", "ratio_joint_per_motor": 1.0}
        if "knee" in role:
            pivot = np.array(cal["geometry"]["per_side"][side]["knee_pivot_fit"]["free_pivot_root_at_zero"], float)
            lower = [f"{S}L_double_bracket_10deg_MIR_MIR_MIR_1", f"{S}L_calf_motor_driver_ext_1",
                     f"{S}L_calf_motor_driver_ext_2", f"{S}L_290mm_tie_rod_v1_1", f"{S}L_210mm_tie_rod_v1_1",
                     f"{S}L_basis_left_1", f"{S}L_skateboard_bearing_left_2"]
            J_lower = plant.inertia_about(lower, pivot, axis)
            J_crank = plant.inertia_about([f"{S}L_RMD_X10_S2_MIR4Rotor_1"], anchor, axis)
            G = knee[side]
            J_link = J_crank + G["at_stand"] ** 2 * J_lower
            transmission = {
                "type": "knee four-bar (polycentric); motor drives the crank",
                "ratio_joint_per_motor": {"at_stand": round(G["at_stand"], 3), "min": round(G["min"], 3),
                                          "max": round(G["max"], 3), "mean": round(G["mean"], 3),
                                          "source": f"CALIB dofs.{side}_knee lut1d slope (d knee / d crank)"},
                "joint_side": {
                    "peak_torque_Nm_at_stand": round(peak / G["at_stand"], 1),
                    "peak_torque_Nm_range": [round(peak / G["max"], 1), round(peak / G["min"], 1)],
                    "rated_torque_Nm_at_stand": round(rated / G["at_stand"], 1),
                    "no_load_speed_rad_s_at_stand": round(tsl["no_load_speed_rad_s"] * G["at_stand"], 2),
                    "speed_gate_rad_s_at_stand": round(tsl["speed_at_rated_torque_rad_s"] * G["at_stand"], 2),
                    "armature_reflected_to_knee_kgm2_at_stand": round(arm / G["at_stand"] ** 2, 4),
                    "note": "tau_knee = tau_crank / G, w_knee = G * w_crank, J_knee = J_crank / G^2 (virtual work). "
                            "The four-bar is a speed-up: the knee spans -4.9..47.2 deg for 0-30 deg of crank."},
                "link_inertia_model": "crank body about the motor axis + G^2 x lower leg (shank, calf motors, rods, foot) "
                                      "about the calibrated free knee pivot (rockers ignored, a few %)",
                "lower_leg_inertia_about_knee_kgm2": round(J_lower, 4),
            }
        elif "calf motor" in role:
            A = ankle[side]
            J = A["J"]
            col = 0 if name.endswith("67") else 1
            p_anchor, p_axis = plant.joint_axis(f"{S}L_Revolute87")
            r_anchor, r_axis = plant.joint_axis(f"{S}L_Revolute88")
            I_p = plant.inertia_about([f"{S}L_basis_left_1", f"{S}L_skateboard_bearing_left_2"], p_anchor, p_axis)
            I_r = plant.inertia_about([f"{S}L_skateboard_bearing_left_2"], r_anchor, r_axis)
            ext = f"{S}L_calf_motor_driver_ext_{1 if col == 0 else 2}"
            J_link = plant.inertia_about([ext], anchor, axis) + J[0, col] ** 2 * I_p + J[1, col] ** 2 * I_r
            # joint-side capability of the PAIR (both motors at the same limit)
            tp = peak / np.max(np.abs(J[0, :]))
            tr = peak / np.max(np.abs(J[1, :]))
            Jinv = np.linalg.inv(J)
            wp = tsl["no_load_speed_rad_s"] / np.max(np.abs(Jinv[:, 0]))
            wr = tsl["no_load_speed_rad_s"] / np.max(np.abs(Jinv[:, 1]))
            gp = tsl["speed_at_rated_torque_rad_s"] / np.max(np.abs(Jinv[:, 0]))
            transmission = {
                "type": "parallel ankle: two calf motors -> tie rods -> foot (pitch + roll)",
                "jacobian_d_pitchroll_d_motors_at_stand": A["J_pitchroll_wrt_motors"],
                "jacobian_motor_order": A["motors"], "source": f"CALIB ankle_pairs.{side} lut2d, standing_motor_pos",
                "joint_side_pair": {
                    "pure_pitch_peak_torque_Nm": round(float(tp), 1), "pure_roll_peak_torque_Nm": round(float(tr), 1),
                    "pure_pitch_rated_torque_Nm": round(float(rated / np.max(np.abs(J[0, :]))), 1),
                    "pure_roll_rated_torque_Nm": round(float(rated / np.max(np.abs(J[1, :]))), 1),
                    "pure_pitch_no_load_speed_rad_s": round(float(wp), 2),
                    "pure_roll_no_load_speed_rad_s": round(float(wr), 2),
                    "pure_pitch_speed_gate_rad_s": round(float(gp), 2),
                    "armature_reflected_to_pitch_roll_kgm2": [round(float(v), 4) for v in
                                                              np.diag(arm * Jinv.T @ Jinv)],
                    "note": "tau_joint = J^-T tau_motor, w_joint = J w_motor; limits for one DOF with the other held; "
                            "reflected armature = diag(armature * J^-T J^-1)"},
                "link_inertia_model": "driver crank about the motor axis + J^T diag(I_pitch, I_roll) J diagonal, foot "
                                      "about the U-joint axes *_Revolute87/88 (tie rods ignored)",
                "foot_inertia_pitch_roll_kgm2": [round(I_p, 5), round(I_r, 5)],
            }
        elif "elbow" in role:
            G = elbow[side]
            f_anchor, f_axis = plant.joint_axis(f"{S}H_Revolute44")
            J_fore = plant.inertia_about([f"{S}H_6mm_bearing__4__1", f"{S}H_shoulder_ex_al_interface_1",
                                          f"{S}H_6mm_bearing__9__1"], f_anchor, f_axis)
            J_link = plant.inertia_about([f"{S}H_bicep_motor_mate_1"], anchor, axis) + G["at_stand"] ** 2 * J_fore
            transmission = {
                "type": "elbow linkage (motor -> bicep link -> forearm), speed-up",
                "ratio_joint_per_motor": {"at_stand": round(G["at_stand"], 3), "abs_min": round(G["abs_min"], 3),
                                          "abs_max": round(G["abs_max"], 3), "mean": round(G["mean"], 3),
                                          "source": f"CALIB dofs.{side}_elbow lut1d slope (G1 elbow convention)"},
                "joint_side": {
                    "peak_torque_Nm_at_stand": round(peak / abs(G["at_stand"]), 2),
                    "peak_torque_Nm_range": [round(peak / G["abs_max"], 2), round(peak / G["abs_min"], 2)],
                    "rated_torque_Nm_at_stand": round(rated / abs(G["at_stand"]), 2),
                    "no_load_speed_rad_s_at_stand": round(tsl["no_load_speed_rad_s"] * abs(G["at_stand"]), 1),
                    "note": "the 0-30 deg motor range spans ~122 deg of elbow: strong speed-up, weak joint torque"},
                "link_inertia_model": "bicep link about the motor axis + G^2 x forearm+hand about *_Revolute44",
                "forearm_hand_inertia_about_elbow_kgm2": round(J_fore, 5),
            }
        else:
            sub = plant.subtree(name)
            J_link = plant.inertia_about(sub, anchor, axis)
            transmission["link_inertia_model"] = f"spanning-tree subtree ({len(sub)} bodies) about the joint axis"

        J_eff = arm + J_link
        w8, w10 = 2 * math.pi * 8, 2 * math.pi * 10
        kp8 = J_eff * w8 ** 2
        kp_hw = min(kp8, MIT_LIMITS_V39["kp"][1])
        kd_crit = 2.0 * math.sqrt(kp_hw * J_eff)
        # Starting point: 8 Hz where reachable, capped by the motion-mode kp ceiling and by torque authority (the PD
        # saturates the PEAK torque at >= 0.25 rad of error, cf. G1/H1 where it saturates at > 1 rad).
        kp_start = min(kp8, MIT_LIMITS_V39["kp"][1], peak / 0.25)
        kd_start = 2.0 * 1.0 * math.sqrt(kp_start * J_eff)
        grp = dn.motor_group(name)
        effort, kp, kd, sim_arm = dn.ACTUATOR_PARAMS[grp]
        prof = {pn: dict(p.get(grp, {})) for pn, p in profiles.items() if p.get(grp)}

        gate = tsl["speed_at_rated_torque_rad_s"]
        cap = tsl["no_load_speed_rad_s"]
        gates[name], caps[name] = round(gate, 2), round(cap, 2)

        joints_out[name] = {
            "sdk_slot": dn.MOTOR_NAMES.index(name), "side": side, "role": role,
            "model": mid, "gear_ratio": N, "mapping": {k: ev[k] for k in ("confidence", "evidence", "resolution")},
            "datasheet": {"peak_torque_Nm": peak, "rated_torque_Nm": rated,
                          "no_load_speed_rad_s_48V": tsl["no_load_speed_rad_s"],
                          "rated_speed_rad_s": round(m["rated_speed_output"]["value"] * RPM, 4),
                          "back_drive_torque_Nm": m["back_drive_torque"]["value"],
                          "backlash_arcmin": m["backlash"]["value"],
                          "rotor_inertia_gcm2": m["rotor_inertia"]["value"]},
            "transmission": transmission,
            "inertia": {"armature_kgm2": arm, "link_inertia_at_rest_kgm2": round(J_link, 5),
                        "J_eff_kgm2": round(J_eff, 5),
                        "armature_share_of_J_eff": round(arm / J_eff, 3)},
            "isaac_lab_proposal": {
                "status": "PROPOSAL (not applied; see docs/ACTUATORS.md section 6)",
                "actuator_class": "DCMotorCfg (explicit) + Coulomb friction term (custom subclass, TODO)",
                "effort_limit": peak, "saturation_effort": tsl["saturation_effort_Nm"],
                "velocity_limit": tsl["no_load_speed_rad_s"],
                "velocity_limit_sim": round(2.0 * tsl["no_load_speed_rad_s"], 2),
                "velocity_limit_sim_note": "PhysX hard cap only as a safety net (2x no-load; the torque-speed line does "
                                           "the physics; external loads can back-drive a joint past no-load speed)",
                "armature": arm,
                "friction_coulomb_Nm": m["back_drive_torque"]["value"],
                "stiffness": "TODO", "damping": "TODO",
            },
            "gains_suggestion_TODO": {
                "rule_8Hz_zeta1": {"kp": round(kp8, 1), "kd": round(2 * J_eff * w8, 2),
                                   "note": "J_eff (armature + link at rest) x (2 pi 8 Hz)^2; kd for zeta = 1"},
                "rule_10Hz_zeta1": {"kp": round(J_eff * w10 ** 2, 1), "kd": round(2 * J_eff * w10, 2)},
                "beyondmimic_rule_armature_only": {"kp": round(arm * w10 ** 2, 1), "kd": round(4 * arm * w10, 2),
                                                   "note": "G1 rule (armature x (2 pi 10)^2, zeta 2): built for "
                                                           "low-ratio QDD motors; not transferable here"},
                "motion_mode_realizable": {
                    "kp": round(kp_hw, 1), "natural_frequency_hz": round(math.sqrt(kp_hw / J_eff) / (2 * math.pi), 2),
                    "kd_for_zeta1_output_side": round(kd_crit, 2),
                    "kd_command_if_protocol_kd_is_output_side": round(kd_crit, 2),
                    "kd_command_if_protocol_kd_is_motor_side": round(kd_crit / N, 3),
                    "kd_protocol_max": MIT_LIMITS_V39["kd"][1],
                    "zeta_at_kd_5_if_output_side": round(5.0 / (2 * math.sqrt(kp_hw * J_eff)), 3),
                    "error_to_saturate_rad": round(peak / kp_hw, 4),
                    "action_scale_rule_0.25_effort_over_kp_rad": round(0.25 * peak / kp_hw, 4),
                    "note": "kp capped at the motion-mode ceiling 500 (PROTO_V39 p90, PROTO_V42 p92). Whether the "
                            "protocol kd multiplies output or motor-side speed is unresolved (DBG_SW p5 says motor "
                            "end): bench test B1 in docs/ACTUATORS.md"},
                "suggested_start": {
                    "kp": round(kp_start, 1), "kd_zeta1": round(kd_start, 2),
                    "natural_frequency_hz": round(math.sqrt(kp_start / J_eff) / (2 * math.pi), 2),
                    "error_to_saturate_peak_rad": round(peak / kp_start, 3),
                    "kd_fits_motion_mode_if_output_side": kd_start <= MIT_LIMITS_V39["kd"][1],
                    "kd_fits_motion_mode_if_motor_side": kd_start / N <= MIT_LIMITS_V39["kd"][1],
                    "rule": "kp = min(J_eff (2 pi 8 Hz)^2, 500 [motion mode], peak / 0.25 rad); kd = 2 sqrt(kp J_eff) "
                            "(zeta 1). TODO: tune in sim with the DCMotor + friction model, then sysid on hardware"},
            },
            "sim_current": {
                "group": grp, "effort_limit_Nm": effort, "kp": kp, "kd": kd, "armature_kgm2": sim_arm,
                "velocity_cap_rad_s": usd_vcap, "velocity_cap_source": "USD physxJoint:maxJointVelocity 572.96 deg/s",
                "velocity_task_profiles": prof,
            },
            "sim_over_datasheet": {
                "effort_legacy": round(effort / peak, 2),
                **({f"effort_{pn}": round(float(p.get("effort_limit_sim", effort)) / peak, 2)
                    for pn, p in prof.items()} if prof else {}),
                "armature": round(sim_arm / arm, 4),
                "velocity_cap_vs_no_load": round(usd_vcap / tsl["no_load_speed_rad_s"], 2),
            },
            "motion_gate_proposal_rad_s": {
                "max_abs_motor_vel": round(gate, 2),
                "max_step_per_50Hz_frame_rad": round(gate / 50.0, 3),
                "hard_reject_above_no_load_rad_s": round(cap, 2),
                "definition": "speed at which the 48 V drive still delivers the RATED torque (torque-speed line)",
            },
        }

    doc = {
        "schema": "dropbear-actuators-datasheet-v1",
        "status": "PROPOSAL - no training default uses this file (docs/ACTUATORS.md)",
        "created_by": "tools/build_actuator_datasheet.py",
        "usd_sha256": dn.USD_SHA256,
        "calibration_sha256": _sha256(SOURCES["CALIB"]["path"]),
        "conventions": {
            "units": "SI; speeds at the actuator OUTPUT shaft unless stated; rpm -> rad/s = x 2 pi / 60",
            "armature": "rotor inertia x gear_ratio^2, set on the motor joint itself (the crank / calf driver / elbow "
                        "motor joint), so PhysX reflects it through the linkages",
            "effort_limit": "datasheet PEAK torque (short duty); rated = thermal continuous value",
            "velocity": "no_load_speed at 48 V from the characteristic curve (X8: conservative SIMPLE_PROFILE value)",
        },
        "sources": {k: {"path": str(v["path"]).replace("\\", "/"), "sha256": _sha256(v["path"]), "what": v["what"]}
                    for k, v in SOURCES.items()},
        "models": {mid: {**m, "derived": derived[mid]} for mid, m in MODELS.items()},
        "joints": joints_out,
        "neck": {
            "joints": list(dn.NECK_NAMES),
            "hardware": "NOT MyActuator: 6 NEMA 17 steppers + A4988 drivers + 4-start lead-screw nuts, one ESP32 "
                        "(BOM Head section); not on the RMD CAN buses",
            "datasheet": None,
            "sim_current": {"effort_N": dn.ACTUATOR_PARAMS["neck"][0], "kp_N_per_m": dn.ACTUATOR_PARAMS["neck"][1],
                            "kd": dn.ACTUATOR_PARAMS["neck"][2]},
            "proposal": "leave unchanged (held by PD, never an action); motor model/lead unknown (Q9)",
        },
        "control_interface": {
            "bus": fact("CAN 2.0A standard frames, DLC 8, 1 Mbit/s (500 kbit/s selectable via 0xB4)", "-", "PROTO_V39",
                        7, "PROTO_V39 p77 (0xB4)"),
            "ids": fact("command 0x140+ID, reply 0x240+ID, ID 1-32; broadcast 0x280 (same payload to every motor); "
                        "motion mode 0x400+ID -> reply 0x500+ID", "-", "PROTO_V39", [7, 85, 89]),
            "servo_commands": {
                "0xA1_torque": fact("iq in 0.01 A/LSB (int16); host needs Kt per model", "-", "PROTO_V39", [45, 46]),
                "0xA2_speed": fact("0.01 dps/LSB (int32) at the output; torque limited by MaxTorqueCurrent", "-",
                                   "PROTO_V39", [49, 50]),
                "0xA4_abs_position": fact("0.01 deg/LSB (int32) + max speed 1 dps/LSB; drive PI position loop", "-",
                                          "PROTO_V39", [52, 53]),
                "servo_reply": fact("temperature 1 C, iq 0.01 A, speed 1 dps (0.0175 rad/s), angle 1 deg/LSB (!) - "
                                    "use 0x92 (0.01 deg) or motion mode for usable position feedback", "-", "PROTO_V39",
                                    [46, 47]),
            },
            "motion_mode_MIT": {
                "exists": True,
                "command": fact("0x400+ID: p_des 16 bit [-12.5, 12.5] rad, v_des 12 bit [-45, 45] rad/s, kp 12 bit "
                                "[0, 500], kd 12 bit [0, 5], t_ff 12 bit [-24, 24] N*m", "-", "PROTO_V39", [89, 90]),
                "law": fact("IqRef = [kp (p_des - p) + kd (v_des - v) + t_ff] * KT", "-", "PROTO_V39", 90),
                "reply": fact("0x500+ID: p 16 bit, v 12 bit, t 12 bit (same ranges; torque feedback saturates at 24 "
                              "N*m)", "-", "PROTO_V39", [90, 91]),
                "same_in_V4.2": fact("identical ranges and law", "-", "PROTO_V42", [92, 93]),
                "V4.4_differences": fact("t_ff range = +/- the motor's max torque; kp 0-500 (Chinese) vs 0-1000 "
                                         "(English); law divides by KT_OUT (output torque constant). V4.4 is the X-V4 "
                                         "generation, not Dropbear's V2/V3 motors", "-", "PROTO_V44", [161, 162]),
                "unit_ambiguity": fact("setup-software manual: v_des and the speed feedback are the MOTOR-end speed "
                                       "(before the reducer) and TorqueRef = (p_des-p) KP + (v_des-v) KD + t_ff", "-",
                                       "DBG_SW", [5, 6]),
                "consequences": [
                    "feed-forward torque is capped at 24 N*m (X10-S2 peak 100, rated 50): gravity compensation of the "
                    "knees/hips must come partly from the kp term",
                    "kp <= 500 N*m/rad: the velocity task's knee kp 600 (stiff_knee_hw) and the teleop elbow kp 600 "
                    "are not representable",
                    "kd <= 5: if kd acts on output speed, the knee kd 12 (legacy) / 20 (stiff_knee_hw) are not "
                    "representable; if it acts on motor speed the effective output kd is up to 5 x ratio",
                ],
            },
            "drive_loops": fact("current loop 15 kHz, speed loop 5 kHz, position loop 1 kHz", "-", "MC_DRIVER", 7),
            "comm_loss_protection": fact("0xB3: timeout in ms in DATA[4..7] (0 = disabled)", "-", "PROTO_V39", [74, 75],
                                         "MYA_README setWatchdog() writes the value into DATA[2..3], which the protocol "
                                         "reads as 0 -> protection DISABLED on the current sketch"),
            "active_reply": fact("0xB6: periodic replies of read commands, interval unit 10 ms (<= 100 Hz)", "-",
                                 "PROTO_V39", [80, 81]),
            "max_command_rate": "not specified by the vendor; bus-limited (see can_timing) and bounded above by the "
                                "1 kHz position loop",
        },
        "can_timing": {"frame_bits_8_byte_standard": {"min": BITS_8B_MIN, "typ": BITS_8B_TYP, "max_stuffed": BITS_8B_MAX},
                       "assumption": "one command + one reply frame per motor per control cycle, 1 Mbit/s, no other "
                                     "traffic; excludes the motor's reply turnaround time (not documented) and "
                                     "ESP32/MCP2515 SPI overhead",
                       "layouts": can_timing()},
    }
    if args.motions:
        doc["motion_speed_analysis"] = motion_analysis(gates, caps)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1), encoding="utf-8")

    # ---------------------------------------------------------------- human summary (for the log) ----------------
    print(f"wrote {out}")
    print("\n== models ==")
    for mid in MODELS:
        dd = derived[mid]
        t = dd["torque_speed_line"]
        print(f"{mid:22s} armature {dd['armature_kgm2']:.5f} kg*m^2  no-load {t['no_load_speed_rad_s']:.2f} rad/s  "
              f"sat {t['saturation_effort_Nm']:.1f} N*m  w@peak {t['speed_at_peak_torque_rad_s']:.2f}  "
              f"w@rated {t['speed_at_rated_torque_rad_s']:.2f}")
        if "torque_speed_line_x8_25_curve" in dd:
            t2 = dd["torque_speed_line_x8_25_curve"]
            print(f"{'  (X8-25 curve)':22s} no-load {t2['no_load_speed_rad_s']:.2f}  sat {t2['saturation_effort_Nm']:.1f}"
                  f"  w@peak {t2['speed_at_peak_torque_rad_s']:.2f}  w@rated {t2['speed_at_rated_torque_rad_s']:.2f}")
    print("\n== joints ==")
    hdr = (f"{'joint':24s} {'model':20s} {'N':>3s} {'peak':>5s} {'rated':>5s} {'w_nl':>6s} {'gate':>6s} "
           f"{'armature':>9s} {'J_link':>8s} {'sim_eff':>7s} {'eff_x':>5s} {'kp8Hz':>8s} {'kp_st':>6s} {'kd_st':>6s} "
           f"{'f_st':>5s} {'sim_kp/kd':>10s}")
    print(hdr)
    for name, j in joints_out.items():
        g = j["gains_suggestion_TODO"]
        st = g["suggested_start"]
        print(f"{name:24s} {j['model']:20s} {j['gear_ratio']:3d} {j['datasheet']['peak_torque_Nm']:5.0f} "
              f"{j['datasheet']['rated_torque_Nm']:5.0f} {j['datasheet']['no_load_speed_rad_s_48V']:6.2f} "
              f"{j['motion_gate_proposal_rad_s']['max_abs_motor_vel']:6.2f} {j['inertia']['armature_kgm2']:9.5f} "
              f"{j['inertia']['link_inertia_at_rest_kgm2']:8.4f} {j['sim_current']['effort_limit_Nm']:7.0f} "
              f"{j['sim_over_datasheet']['effort_legacy']:5.2f} {g['rule_8Hz_zeta1']['kp']:8.1f} "
              f"{st['kp']:6.1f} {st['kd_zeta1']:6.2f} {st['natural_frequency_hz']:5.2f} "
              f"{j['sim_current']['kp']:>5.0f}/{j['sim_current']['kd']:<4.0f}")
    print("\n== transmissions (left) ==")
    for name in ("LL_knee_actuator_joint", "LL_Revolute67", "LL_Revolute81", "LH_elbow_joint"):
        print(name, json.dumps(joints_out[name]["transmission"], default=float)[:900])
    print("\n== CAN timing (worst-case stuffing) ==")
    for key, lay in doc["can_timing"]["layouts"].items():
        print(key, "limiting bus max rate @70% load:", lay["limiting_bus_max_rate_hz_70pct"], "Hz")
        for bus, b in lay["buses"].items():
            print(f"   {bus:36s} n={b['motors']:2d} cycle {b['cycle_time_ms_min_typ_max']} ms  "
                  f"max {b['max_rate_hz_at_100pct_load_worstcase']} Hz (100%)  load {b['bus_load_pct_worstcase']}")
    if args.motions:
        ma = doc["motion_speed_analysis"]
        rc = ma["reference_clips"]
        print(f"\n== reference clips: {rc['n_clips']} accepted, {rc['clips_passing_all_gates']} pass all proposed gates, "
              f"{rc['clips_over_no_load_speed_any_motor']} exceed a no-load speed ==")
        print("clips over gate per motor:", json.dumps(rc["clips_over_gate_per_motor"]))
        print("max |v| over clips per motor:", json.dumps(rc["max_over_clips_per_motor"]))
        for r in ma["policy_rollouts"]:
            worst = sorted(r["frac_frames_over_gate"].items(), key=lambda kv: -kv[1])[:6]
            print(f"\nrollout {r['file']} frames={r['frames']} falls={r['falls_env0']}")
            print("  max |v|:", json.dumps(r["max_abs_motor_vel"]))
            print("  frac frames over gate (worst 6):", worst)


if __name__ == "__main__":
    main()
