#!/usr/bin/env bash
# Arm A5 (priority 5): no lower bound. Identical to A3 except the floor is
# removed entirely (mode no_lower_zero): bonus for every P <= B, nothing zeroed.
set -euo pipefail
export IF_PPL_ANCHOR_REWARD_MODE=no_lower_zero
export IF_PPL_ANCHOR_FLIP_HANDLING=strict   # outcome-identical either way (flips can never bonus); kept for literal one-var-diff vs A3
ARM_TAG=a5_nofloor_upperonly
source "$(dirname -- "${BASH_SOURCE[0]}")/_subset4k_common.sh"
