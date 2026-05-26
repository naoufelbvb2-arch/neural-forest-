"""Shared Spine: multi-layer causal attention backbone, always resident in VRAM (HOT tier).

Architecture per forward pass:
    tokens → embedding → [SpineBlock × L] → final_norm → hidden_states (→ Router)

No FFN lives here — domain-specific FFN computation happens in the Zones after routing.
KV cache is maintained in the Spine so context is never fragmented across zone switches.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from forest.config import ForestConfig

# True on PyTorch 2.0+; enables Flash Attention on supported GPUs automatically.
_SDPA_AVAILABLE = hasattr(F, "scaled_dot_product_attention")


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Faster than LayerNorm: normalizes with RMS only (no mean centering, no bias).
    Used in Llama, Mistral, Gemma.

    Formula: x / sqrt(mean(x²) + eps) × weight

    Args:
        dim: Feature dimension to normalize over.
        eps: Numerical stability constant.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # torch.rsqrt is numerically stable and fused on GPU
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


# ---------------------------------------------------------------------------
# Rotary Position Embedding (RoPE)
# ---------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split last dim in half and swap with negation: [a, b] → [-b, a].

    Used to implement the 2-D rotation in RoPE without a trig loop.
    """
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def _apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply Rotary Position Embedding to a query or key tensor.

    For each adjacent pair (x_i, x_{i+d/2}), apply 2-D rotation by angle θ_i:
        x_i'       =  x_i * cos(θ) - x_{i+d/2} * sin(θ)
        x_{i+d/2}' = x_{i+d/2} * cos(θ) + x_i * sin(θ)

    Args:
        x:   Shape (batch, heads, seq, head_dim).
        cos: Shape (seq, head_dim) — broadcast over batch and heads.
        sin: Shape (seq, head_dim).

    Returns:
        Rotated tensor, same shape as x.
    """
    # Unsqueeze to (1, 1, seq, head_dim) for broadcast
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + _rotate_half(x) * sin


class RotaryEmbedding(nn.Module):
    """Precomputed Rotary Position Embedding tables (no learnable parameters).

    Frequencies follow the Llama convention:
        θ_i = 1 / (base ^ (2i / head_dim))   for i in [0, head_dim/2)

    Tables are stored as buffers: they move with the model to GPU automatically.
    Supports KV-cache decoding via the `offset` argument.

    Args:
        head_dim:    Attention head dimension (must be even).
        max_seq_len: Maximum sequence length to precompute.
        base:        RoPE base frequency (default 10 000, same as Llama).
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        base: float = 10_000.0,
    ) -> None:
        super().__init__()
        # θ_i = base^(-2i/d): shape (head_dim/2,)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq)
        self._precompute(max_seq_len)

    def _precompute(self, max_seq_len: int) -> None:
        """Build cos/sin tables for positions 0 … max_seq_len-1."""
        t = torch.arange(max_seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        # outer product → (max_seq, head_dim/2); cat → (max_seq, head_dim)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, seq_len: int, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cos/sin for positions [offset, offset + seq_len).

        Args:
            seq_len: Number of positions needed (length of current input).
            offset:  Starting position — equals the length of the KV cache.

        Returns:
            cos, sin: Each of shape (seq_len, head_dim).
        """
        return (
            self.cos_cached[offset : offset + seq_len],
            self.sin_cached[offset : offset + seq_len],
        )


# ---------------------------------------------------------------------------
# Spine Attention
# ---------------------------------------------------------------------------

class SpineAttention(nn.Module):
    """Multi-head causal self-attention with RoPE and optional KV cache.

    Uses ``torch.nn.functional.scaled_dot_product_attention`` when available
    (PyTorch 2.0+), which dispatches to Flash Attention on supported GPUs.
    Falls back to a manual implementation on older builds.

    No projection bias (consistent with Llama/Mistral style).

    KV-cache protocol:
        - Training / first inference step: kv_cache=None → full causal pass.
        - Subsequent inference steps: kv_cache=(k_past, v_past) → append and attend.

    Args:
        config: ForestConfig instance.
    """

    def __init__(self, config: ForestConfig) -> None:
        super().__init__()
        self.num_heads = config.spine_heads
        self.head_dim  = config.spine_head_dim
        self.dropout   = config.dropout
        inner_dim = self.num_heads * self.head_dim

        self.q_proj = nn.Linear(config.embed_dim, inner_dim, bias=False)
        self.k_proj = nn.Linear(config.embed_dim, inner_dim, bias=False)
        self.v_proj = nn.Linear(config.embed_dim, inner_dim, bias=False)
        self.o_proj = nn.Linear(inner_dim, config.embed_dim, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            nn.init.xavier_uniform_(proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Causal self-attention with optional KV cache.

        Args:
            x:        Input hidden states, shape (batch, seq, embed_dim).
            cos:      RoPE cosines for current positions, shape (seq, head_dim).
            sin:      RoPE sines for current positions, shape (seq, head_dim).
            kv_cache: Cached (key, value) from prior steps, or None.

        Returns:
            output:      Shape (batch, seq, embed_dim).
            new_kv_cache: Updated (key, value) tensors including the current step.
                          key/value each have shape (batch, heads, total_seq, head_dim).
        """
        B, T, _ = x.shape

        # Project and reshape → (batch, heads, seq, head_dim)
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # RoPE: rotate Q and K (not V) to encode position information
        q = _apply_rotary(q, cos, sin)
        k = _apply_rotary(k, cos, sin)

        # Append new keys/values to cache (grows along the sequence dimension)
        if kv_cache is not None:
            k_past, v_past = kv_cache
            k = torch.cat([k_past, k], dim=2)
            v = torch.cat([v_past, v], dim=2)

        new_kv_cache = (k, v)

        # Attention: causal only when the full sequence is processed at once.
        # When a cache prefix exists, Q is shorter than K/V so is_causal=False
        # is correct — the cache already contains only past tokens.
        if _SDPA_AVAILABLE:
            is_causal = kv_cache is None
            dropout_p = self.dropout if self.training else 0.0
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )
        else:
            attn_out = self._manual_attention(q, k, v, apply_causal=kv_cache is None)

        out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out), new_kv_cache

    def _manual_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        apply_causal: bool,
    ) -> torch.Tensor:
        """Reference attention for environments without scaled_dot_product_attention."""
        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if apply_causal:
            seq_q, seq_k = q.shape[2], k.shape[2]
            # Upper-triangular mask: position i cannot see j > i
            mask = torch.triu(
                torch.ones(seq_q, seq_k, device=q.device, dtype=torch.bool),
                diagonal=1,
            )
            scores = scores.masked_fill(mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        if self.training and self.dropout > 0.0:
            weights = F.dropout(weights, p=self.dropout)
        return torch.matmul(weights, v)


# ---------------------------------------------------------------------------
# Spine Block
# ---------------------------------------------------------------------------

class SpineBlock(nn.Module):
    """Single spine layer: pre-norm + causal self-attention + residual connection.

    No FFN — domain-specific computation is handled by the Zones after routing.

    Architecture: output = x + Attention(RMSNorm(x))

    Args:
        config: ForestConfig instance.
    """

    def __init__(self, config: ForestConfig) -> None:
        super().__init__()
        self.norm = RMSNorm(config.embed_dim)
        self.attn = SpineAttention(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """One spine layer forward pass.

        Args:
            x:        Hidden states, shape (batch, seq, embed_dim).
            cos:      RoPE cosines for current positions.
            sin:      RoPE sines for current positions.
            kv_cache: Cached (K, V) from prior steps, or None.

        Returns:
            Updated hidden states and new KV cache for this layer.
        """
        attn_out, new_kv = self.attn(self.norm(x), cos, sin, kv_cache)
        return x + attn_out, new_kv


# ---------------------------------------------------------------------------
# Shared Spine (main export)
# ---------------------------------------------------------------------------

class SharedSpine(nn.Module):
    """Shared attention backbone: always resident in VRAM (HOT tier).

    Processes all input tokens before they are routed to specialty Zones.
    Maintains a coherent KV cache — attention is never fragmented across zones.

    Components:
        token_embedding: nn.Embedding shared with the LM head (weight tying).
        rotary:          Precomputed RoPE tables (no learnable parameters).
        blocks:          spine_layers × SpineBlock (attention only, no FFN).
        final_norm:      RMSNorm applied to output before passing to Router.

    Args:
        config: ForestConfig instance.
    """

    def __init__(self, config: ForestConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.rotary  = RotaryEmbedding(config.spine_head_dim, config.max_seq_len)
        self.blocks  = nn.ModuleList([SpineBlock(config) for _ in range(config.spine_layers)])
        self.final_norm = RMSNorm(config.embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.ones_(self.final_norm.weight)

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the token embedding for weight tying with the LM head."""
        return self.token_embedding

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Run the full spine: embed → L attention layers → final norm.

        Args:
            input_ids: Token IDs, shape (batch, seq_len).
            kv_cache:  Per-layer (K, V) caches from prior steps.
                       - None: full-sequence pass (training or first inference step).
                       - List of length spine_layers: incremental decoding.

        Returns:
            hidden_states: Shape (batch, seq_len, embed_dim). Ready for Router.
            new_kv_cache:  Updated list of (K, V) tensors, one entry per layer.
                           Each K/V has shape (batch, heads, total_seq, head_dim).
        """
        # Positional offset = number of tokens already in the cache
        offset = kv_cache[0][0].shape[2] if kv_cache is not None else 0
        seq_len = input_ids.shape[1]

        x = self.token_embedding(input_ids)
        cos, sin = self.rotary(seq_len, offset)

        new_kv_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, block in enumerate(self.blocks):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x, new_kv = block(x, cos, sin, layer_cache)
            new_kv_cache.append(new_kv)

        return self.final_norm(x), new_kv_cache
