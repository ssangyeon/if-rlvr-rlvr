#!/usr/bin/env bash
# Arm A2 (priority 3): soft floor. Identical to A3 except the floor action:
# below-floor rows keep IF (no zero) and forfeit the bonus. Bonus only in band.
set -euo pipefail
export IF_PPL_ANCHOR_FLIP_HANDLING=strict
export IF_PPL_ANCHOR_FLOOR_ACTION=ignore
ARM_TAG=a2_softfloor_keepif
source "$(dirname -- "${BASH_SOURCE[0]}")/_subset4k_common.sh"
