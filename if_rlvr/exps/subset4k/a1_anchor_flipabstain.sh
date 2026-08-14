#!/usr/bin/env bash
# Arm A1 (priority 6): flip-abstain. Identical to A3 except flipped (B < A) rows
# are left untouched (keep IF); the hard-zero floor stays on valid-band rows.
# (This cell equals the post-2026-08-09 shipped rule.)
set -euo pipefail
export IF_PPL_ANCHOR_FLIP_HANDLING=abstain
export IF_PPL_ANCHOR_FLOOR_ACTION=zero
ARM_TAG=a1_flipabstain_floorzero
source "$(dirname -- "${BASH_SOURCE[0]}")/_subset4k_common.sh"
