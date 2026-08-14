#!/usr/bin/env bash
# Arm A3 (priority 2): STRICT baseline = the July reward semantics on v2 data.
# Floor (hard zero) fires before the band-validity check, so flipped rows with
# P < A are zeroed too. Every other arm changes exactly one component vs this.
set -euo pipefail
export IF_PPL_ANCHOR_FLIP_HANDLING=strict
export IF_PPL_ANCHOR_FLOOR_ACTION=zero
ARM_TAG=a3_baseline_floorzero_flipszeroed
source "$(dirname -- "${BASH_SOURCE[0]}")/_subset4k_common.sh"
