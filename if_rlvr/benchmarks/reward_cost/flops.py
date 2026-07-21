"""Architecture-aware logical FLOPs estimates for reward-model inference.

The estimates count one fused multiply-add as two FLOPs and cover the large
matrix multiplications in attention projections, dense/active-MoE MLPs,
attention score/value products, and the language-model head.  They deliberately
exclude normalization, rotary embeddings, activations, softmax/top-k, sampling,
memory traffic, quantization/dequantization, and kernel/CUDA-graph padding.

MoE estimates count only the routed top-k experts, plus the router projection.
They are therefore *logical model FLOPs*, not a hardware instruction counter.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any


FMA_FLOPS = 2


def _required_positive_int(config: dict[str, Any], key: str) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"config[{key!r}] must be a positive integer, got {value!r}")
    return int(value)


def _optional_nonnegative_int(config: dict[str, Any], key: str, default: int = 0) -> int:
    value = config.get(key, default)
    if value is None:
        value = default
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError(f"config[{key!r}] must be a non-negative integer, got {value!r}")
    return int(value)


def _nonnegative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return int(value)


def _triangular(length: int) -> int:
    """Number of causal query-key pairs for a full-attention sequence."""

    return length * (length + 1) // 2


def _sliding_pairs(length: int, window: int) -> int:
    """Sum ``min(position, window)`` over one-indexed causal positions."""

    capped = min(length, window)
    return capped * (capped + 1) // 2 + (length - capped) * window


@dataclass(frozen=True)
class ModelSpec:
    """Minimal architecture description needed by the FLOPs estimator."""

    model_type: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    dense_intermediate_size: int
    expert_intermediate_size: int
    num_experts: int
    experts_per_token: int
    num_dense_layers: int
    num_moe_layers: int
    num_full_attention_layers: int
    num_sliding_attention_layers: int
    sliding_window: int | None
    tie_word_embeddings: bool

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ModelSpec":
        """Build a spec from a Hugging Face-style config dictionary.

        Supported target layouts are dense Qwen3, Qwen3MoE, and GptOss.  The
        field aliases used by their published configs are handled directly:
        ``num_experts``/``num_local_experts`` and
        ``num_experts_per_tok``/``experts_per_token``.
        """

        if not isinstance(config, dict):
            raise TypeError(f"config must be a dict, got {type(config).__name__}")

        # Some composite configs place the language model under text_config.
        raw = dict(config)
        text_config = raw.get("text_config")
        if "hidden_size" not in raw and isinstance(text_config, dict):
            raw = {**raw, **text_config}

        model_type = str(raw.get("model_type") or "unknown").lower()
        hidden_size = _required_positive_int(raw, "hidden_size")
        num_hidden_layers = _required_positive_int(raw, "num_hidden_layers")
        num_attention_heads = _required_positive_int(raw, "num_attention_heads")
        num_key_value_heads = int(raw.get("num_key_value_heads") or num_attention_heads)
        if num_key_value_heads <= 0:
            raise ValueError("num_key_value_heads must be positive")

        raw_head_dim = raw.get("head_dim")
        if raw_head_dim is None:
            if hidden_size % num_attention_heads:
                raise ValueError(
                    "head_dim is absent and hidden_size is not divisible by num_attention_heads"
                )
            head_dim = hidden_size // num_attention_heads
        else:
            if isinstance(raw_head_dim, bool) or not isinstance(raw_head_dim, Integral):
                raise ValueError(f"head_dim must be a positive integer, got {raw_head_dim!r}")
            head_dim = int(raw_head_dim)
            if head_dim <= 0:
                raise ValueError(f"head_dim must be positive, got {head_dim}")

        vocab_size = _required_positive_int(raw, "vocab_size")
        dense_intermediate_size = _required_positive_int(raw, "intermediate_size")

        num_experts = _optional_nonnegative_int(raw, "num_experts")
        if num_experts == 0:
            num_experts = _optional_nonnegative_int(raw, "num_local_experts")

        if num_experts:
            raw_top_k = raw.get("num_experts_per_tok", raw.get("experts_per_token"))
            if isinstance(raw_top_k, bool) or not isinstance(raw_top_k, Integral):
                raise ValueError(f"MoE config requires integer experts-per-token, got {raw_top_k!r}")
            experts_per_token = int(raw_top_k)
            if not 0 < experts_per_token <= num_experts:
                raise ValueError(
                    f"experts_per_token must be in [1, {num_experts}], got {experts_per_token}"
                )
            expert_intermediate_size = int(
                raw.get("moe_intermediate_size") or dense_intermediate_size
            )
            if expert_intermediate_size <= 0:
                raise ValueError("expert intermediate size must be positive")
        else:
            experts_per_token = 0
            expert_intermediate_size = 0

        # Qwen3MoE can mix dense and sparse layers.  GptOss uses MoE in every
        # decoder layer.  The published Qwen3-30B-A3B config has step=1 and an
        # empty mlp_only_layers list, so all 48 layers are sparse.
        if num_experts == 0:
            num_moe_layers = 0
        elif model_type == "qwen3_moe":
            decoder_sparse_step = _optional_nonnegative_int(raw, "decoder_sparse_step", 1)
            if decoder_sparse_step == 0:
                raise ValueError("decoder_sparse_step must be positive")
            mlp_only_layers = raw.get("mlp_only_layers") or []
            if not isinstance(mlp_only_layers, (list, tuple)):
                raise ValueError("mlp_only_layers must be a list or tuple")
            dense_layer_indices = {int(index) for index in mlp_only_layers}
            invalid_indices = [
                index for index in dense_layer_indices if index < 0 or index >= num_hidden_layers
            ]
            if invalid_indices:
                raise ValueError(f"mlp_only_layers contains invalid indices: {invalid_indices}")
            num_moe_layers = sum(
                index not in dense_layer_indices and (index + 1) % decoder_sparse_step == 0
                for index in range(num_hidden_layers)
            )
        else:
            num_moe_layers = num_hidden_layers
        num_dense_layers = num_hidden_layers - num_moe_layers

        layer_types = raw.get("layer_types")
        sliding_window_value = raw.get("sliding_window")
        sliding_window = None if sliding_window_value is None else int(sliding_window_value)
        if layer_types is not None:
            if not isinstance(layer_types, (list, tuple)):
                raise ValueError("layer_types must be a list or tuple")
            if len(layer_types) != num_hidden_layers:
                raise ValueError(
                    f"layer_types has {len(layer_types)} entries; expected {num_hidden_layers}"
                )
            num_sliding_attention_layers = sum(
                "sliding" in str(layer_type).lower() for layer_type in layer_types
            )
        elif model_type == "gpt_oss" and sliding_window is not None:
            # The vLLM/HF GptOss implementations alternate sliding attention
            # on zero-based even layers and full attention on odd layers.
            num_sliding_attention_layers = (num_hidden_layers + 1) // 2
        else:
            num_sliding_attention_layers = 0

        num_full_attention_layers = num_hidden_layers - num_sliding_attention_layers
        if num_sliding_attention_layers:
            if sliding_window is None or sliding_window <= 0:
                raise ValueError("sliding-attention layers require a positive sliding_window")
        elif sliding_window is not None and sliding_window <= 0:
            raise ValueError("sliding_window must be positive when provided")

        return cls(
            model_type=model_type,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            vocab_size=vocab_size,
            dense_intermediate_size=dense_intermediate_size,
            expert_intermediate_size=expert_intermediate_size,
            num_experts=num_experts,
            experts_per_token=experts_per_token,
            num_dense_layers=num_dense_layers,
            num_moe_layers=num_moe_layers,
            num_full_attention_layers=num_full_attention_layers,
            num_sliding_attention_layers=num_sliding_attention_layers,
            sliding_window=sliding_window,
            tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
        )

    @property
    def attention_projection_flops_per_token_per_layer(self) -> int:
        # Q and O use num_attention_heads; K and V use num_key_value_heads.
        return (
            FMA_FLOPS
            * self.hidden_size
            * 2
            * self.head_dim
            * (self.num_attention_heads + self.num_key_value_heads)
        )

    @property
    def dense_mlp_flops_per_token_per_layer(self) -> int:
        # SwiGLU gate/up/down projections: 3 matrices, each multiply-add = 2.
        return 3 * FMA_FLOPS * self.hidden_size * self.dense_intermediate_size

    @property
    def moe_mlp_flops_per_token_per_layer(self) -> int:
        if self.num_experts == 0:
            return 0
        active_experts = (
            3
            * FMA_FLOPS
            * self.hidden_size
            * self.expert_intermediate_size
            * self.experts_per_token
        )
        router = FMA_FLOPS * self.hidden_size * self.num_experts
        return active_experts + router

    @property
    def linear_flops_per_processed_token(self) -> int:
        attention_projections = (
            self.num_hidden_layers * self.attention_projection_flops_per_token_per_layer
        )
        mlps = (
            self.num_dense_layers * self.dense_mlp_flops_per_token_per_layer
            + self.num_moe_layers * self.moe_mlp_flops_per_token_per_layer
        )
        return attention_projections + mlps

    @property
    def attention_flops_per_pair_per_layer(self) -> int:
        # QK^T and attention-probability @ V, each one multiply-add.
        return 2 * FMA_FLOPS * self.num_attention_heads * self.head_dim

    @property
    def lm_head_flops_per_position(self) -> int:
        # Tied embeddings save parameters, but the output projection still runs.
        return FMA_FLOPS * self.hidden_size * self.vocab_size

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable architecture and FLOPs summary."""

        return {
            "model_type": self.model_type,
            "fma_flops": FMA_FLOPS,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "vocab_size": self.vocab_size,
            "dense_intermediate_size": self.dense_intermediate_size,
            "expert_intermediate_size": self.expert_intermediate_size,
            "num_experts": self.num_experts,
            "experts_per_token": self.experts_per_token,
            "num_dense_layers": self.num_dense_layers,
            "num_moe_layers": self.num_moe_layers,
            "num_full_attention_layers": self.num_full_attention_layers,
            "num_sliding_attention_layers": self.num_sliding_attention_layers,
            "sliding_window": self.sliding_window,
            "tie_word_embeddings": self.tie_word_embeddings,
            "attention_projection_flops_per_token_per_layer": (
                self.attention_projection_flops_per_token_per_layer
            ),
            "dense_mlp_flops_per_token_per_layer": self.dense_mlp_flops_per_token_per_layer,
            "moe_mlp_flops_per_token_per_layer": self.moe_mlp_flops_per_token_per_layer,
            "linear_flops_per_processed_token": self.linear_flops_per_processed_token,
            "attention_flops_per_pair_per_layer": self.attention_flops_per_pair_per_layer,
            "lm_head_flops_per_position": self.lm_head_flops_per_position,
        }


def _attention_flops_between(spec: ModelSpec, start_length: int, end_length: int) -> int:
    """Attention FLOPs for newly processed positions ``(start, end]``."""

    if end_length < start_length:
        raise ValueError("end_length must be greater than or equal to start_length")

    full_pairs = _triangular(end_length) - _triangular(start_length)
    sliding_pairs = 0
    if spec.num_sliding_attention_layers:
        assert spec.sliding_window is not None
        sliding_pairs = _sliding_pairs(end_length, spec.sliding_window) - _sliding_pairs(
            start_length, spec.sliding_window
        )

    pair_layers = (
        spec.num_full_attention_layers * full_pairs
        + spec.num_sliding_attention_layers * sliding_pairs
    )
    return spec.attention_flops_per_pair_per_layer * pair_layers


def _result(linear: int, attention: int, lm_head: int) -> dict[str, int]:
    return {
        "linear": linear,
        "attention": attention,
        "lm_head": lm_head,
        "total": linear + attention + lm_head,
    }


def estimate_ppl(spec: ModelSpec, sequence_length: int) -> dict[str, int]:
    """Estimate a full prompt-logprob/PPL forward over ``sequence_length``.

    The language-model head is evaluated at every position.  This matches the
    vLLM prompt-logprobs path used by the PPL reward: it computes prompt-token
    logits plus the final sampling-position logits.
    """

    sequence_length = _nonnegative_int("sequence_length", sequence_length)
    linear = sequence_length * spec.linear_flops_per_processed_token
    attention = _attention_flops_between(spec, 0, sequence_length)
    lm_head = sequence_length * spec.lm_head_flops_per_position
    return _result(linear, attention, lm_head)


def estimate_generation(
    spec: ModelSpec,
    prompt_tokens: int,
    completion_tokens: int,
    cached_prompt_tokens: int = 0,
) -> dict[str, int]:
    """Estimate autoregressive verifier generation FLOPs.

    ``completion_tokens`` is the API usage count.  Prefill produces the first
    completion token, so only ``completion_tokens - 1`` completion tokens are
    subsequently fed through the transformer.  A zero-token completion is
    treated as a prompt-only forward with no language-model-head evaluation.

    ``cached_prompt_tokens`` denotes a contiguous prefix whose KV state is
    reused.  Linear work is charged only for uncached/new tokens; attention is
    the causal-pair difference between the cached prefix and the final processed
    length.  vLLM normally caps cache hits below the full prompt length when it
    needs sampling logits, but accepting the full inclusive range keeps this
    estimator useful for trace-derived and hypothetical inputs.
    """

    prompt_tokens = _nonnegative_int("prompt_tokens", prompt_tokens)
    completion_tokens = _nonnegative_int("completion_tokens", completion_tokens)
    cached_prompt_tokens = _nonnegative_int("cached_prompt_tokens", cached_prompt_tokens)
    if cached_prompt_tokens > prompt_tokens:
        raise ValueError(
            "cached_prompt_tokens cannot exceed prompt_tokens: "
            f"{cached_prompt_tokens} > {prompt_tokens}"
        )

    decode_input_tokens = max(completion_tokens - 1, 0)
    newly_processed_tokens = prompt_tokens - cached_prompt_tokens + decode_input_tokens
    final_processed_length = prompt_tokens + decode_input_tokens

    linear = newly_processed_tokens * spec.linear_flops_per_processed_token
    attention = _attention_flops_between(
        spec, cached_prompt_tokens, final_processed_length
    )
    lm_head = completion_tokens * spec.lm_head_flops_per_position
    return _result(linear, attention, lm_head)

