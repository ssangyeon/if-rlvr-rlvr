#!/usr/bin/env bash
# Arm B1 (Stage 2): calibrated loosened floor, k=1 (preserves the single-generation claim).
# Fixed rule (abstain flips + hard zero) with the floor shifted down by c nats:
# zero only when P < A - c; the band's effective lower edge moves down with it.
# c defaults to the offline calibration (tools/calibrate_floor_margin.py,
# pre-registered pseudo-wipe target 2%); override with IF_PPL_ANCHOR_FLOOR_MARGIN.
set -euo pipefail
HERE=$(dirname -- "${BASH_SOURCE[0]}")
if [ -z "${IF_PPL_ANCHOR_FLOOR_MARGIN:-}" ]; then
    CAL="${HERE}/calibration_floor_margin.json"
    [ -s "$CAL" ] || { echo "[b1] no IF_PPL_ANCHOR_FLOOR_MARGIN and no ${CAL}; run tools/calibrate_floor_margin.py first" >&2; exit 1; }
    IF_PPL_ANCHOR_FLOOR_MARGIN=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['recommended_c'])" "$CAL")
fi
export IF_PPL_ANCHOR_FLOOR_MARGIN
export IF_PPL_ANCHOR_FLIP_HANDLING=abstain
export IF_PPL_ANCHOR_FLOOR_ACTION=zero
ARM_TAG=b1_floormargin
source "${HERE}/_subset4k_common.sh"
