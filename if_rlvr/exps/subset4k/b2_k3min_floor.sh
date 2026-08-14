#!/usr/bin/env bash
# Arm B2 (Stage 2): k=3 loosened floor (the anchor-robustness experiment).
# Fixed rule (abstain flips + hard zero) on a DERIVED cache whose ref0 encodes
# the chosen k-draw floor form (default min-of-3). Build the cache first:
#   tools/make_k3_floor_cache.py --base <SUBSET4096.json> --draws d2.json d3.json \
#       --form min3 --out <SUBSET4096.k3min.json>
set -euo pipefail
HERE=$(dirname -- "${BASH_SOURCE[0]}")
VERL_DIR=${VERL_DIR:-$(cd -- "${HERE}/../../.." && pwd)}
export SUBSET_CACHE=${SUBSET_CACHE:-${VERL_DIR}/.cache/if_ref_anchor_teacher4b_reasoning_train_seed1_val512_t1_p095_k20_pp0_r8192_scored_by_qwen3_4b.SUBSET4096.k3min.json}
[ -s "$SUBSET_CACHE" ] || { echo "[b2] derived k3-floor cache missing: $SUBSET_CACHE (build with tools/make_k3_floor_cache.py)" >&2; exit 1; }
export IF_PPL_ANCHOR_FLIP_HANDLING=abstain
export IF_PPL_ANCHOR_FLOOR_ACTION=zero
ARM_TAG=b2_k3minfloor
source "${HERE}/_subset4k_common.sh"
