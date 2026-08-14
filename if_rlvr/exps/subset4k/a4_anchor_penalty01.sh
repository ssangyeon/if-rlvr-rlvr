#!/usr/bin/env bash
# Arm A4 (priority 4): bounded penalty. Identical to A3 except the floor action:
# below-floor rows get IF - 0.1 (clamped at 0) instead of the hard zero.
set -euo pipefail
export IF_PPL_ANCHOR_FLIP_HANDLING=strict
export IF_PPL_ANCHOR_FLOOR_ACTION=penalty
export IF_PPL_ANCHOR_FLOOR_PENALTY=0.1
ARM_TAG=a4_floorpenalty01
source "$(dirname -- "${BASH_SOURCE[0]}")/_subset4k_common.sh"
