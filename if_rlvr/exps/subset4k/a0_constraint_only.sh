#!/usr/bin/env bash
# Arm A0 (priority 1): constraint-only reference. Same 4,096 rows (cache used
# purely as the row filter), anchor reward disabled, no per-rollout PPL scoring.
set -euo pipefail
export PY_GIVEN_X_REWARD_COEFF=0.0
export IF_REF_POLICY_ANCHOR_PPL=false
ARM_TAG=a0_constraint_only
source "$(dirname -- "${BASH_SOURCE[0]}")/_subset4k_common.sh"
