#!/usr/bin/env bash
# Create an isolated Transformers 4.57.1 overlay for OLMo3 without modifying
# the existing Qwen/SGLang verl environment.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=_olmo3_7b_common.sh
source "${SCRIPT_DIR}/_olmo3_7b_common.sh"

if [[ ! -x "${OLMO3_BASE_PYTHON_BIN}" ]]; then
    echo "Base verl Python not found: ${OLMO3_BASE_PYTHON_BIN}" >&2
    exit 1
fi

if pgrep -f "${OLMO3_VENV_DIR}/bin/python.*verl\.trainer" >/dev/null 2>&1; then
    echo "An OLMo3 verl job is using ${OLMO3_VENV_DIR}; refusing to modify its runtime." >&2
    exit 1
fi

mkdir -p "$(dirname -- "${OLMO3_VENV_DIR}")"
if [[ ! -x "${OLMO3_VENV_DIR}/bin/python" ]]; then
    "${OLMO3_BASE_PYTHON_BIN}" -m venv --system-site-packages "${OLMO3_VENV_DIR}"
fi

if "${OLMO3_VENV_DIR}/bin/python" - <<'PY'
import transformers
from transformers.models.olmo3 import Olmo3ForCausalLM  # noqa: F401

raise SystemExit(0 if transformers.__version__ == "4.57.1" else 1)
PY
then
    echo "OLMo3 Transformers overlay is already installed; skipping pip."
else
    "${OLMO3_VENV_DIR}/bin/python" -m pip install --upgrade "transformers==4.57.1"
fi

"${OLMO3_VENV_DIR}/bin/python" - <<'PY'
from transformers import AutoConfig, AutoTokenizer
from transformers.models.olmo3 import Olmo3ForCausalLM  # noqa: F401
from vllm.model_executor.models.registry import ModelRegistry

model_id = "allenai/Olmo-3-7B-Instruct-DPO"
config = AutoConfig.from_pretrained(model_id)
assert config.model_type == "olmo3", config.model_type
assert config.architectures == ["Olmo3ForCausalLM"], config.architectures
assert "Olmo3ForCausalLM" in ModelRegistry.get_supported_archs()

tokenizer = AutoTokenizer.from_pretrained(model_id)
rendered = tokenizer.apply_chat_template(
    [{"role": "user", "content": "runtime check"}],
    tokenize=False,
    add_generation_prompt=True,
)
assert "<|im_start|>assistant" in rendered
print(f"[OLMo3 setup] model_type={config.model_type} architecture={config.architectures[0]}")
print(f"[OLMo3 setup] tokenizer={tokenizer.__class__.__name__} chat_template=ok")
PY

echo "OLMo3 runtime ready: ${OLMO3_VENV_DIR}/bin/python"
