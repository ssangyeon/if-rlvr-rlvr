#!/usr/bin/env bash
# Arm B3 (Stage 2, contingent): the one non-degenerate Stage-1 composition —
# flip-abstain + bounded penalty. Run only if BOTH components showed individual
# gains in Stage 1 (A1 - A3 and A4 - A3).
set -euo pipefail
export IF_PPL_ANCHOR_FLIP_HANDLING=abstain
export IF_PPL_ANCHOR_FLOOR_ACTION=penalty
export IF_PPL_ANCHOR_FLOOR_PENALTY=0.1
ARM_TAG=b3_flipabstain_penalty01
source "$(dirname -- "${BASH_SOURCE[0]}")/_subset4k_common.sh"
