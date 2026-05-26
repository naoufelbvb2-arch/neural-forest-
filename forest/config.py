"""ForestConfig: central configuration dataclass for Neural Forest models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ForestConfig:
    """Configuration for a Neural Forest model.

    Attributes:
        vocab_size: Vocabulary size.
        embed_dim: Token embedding dimension (also spine hidden size).
        spine_layers: Number of shared attention layers in the spine.
        spine_heads: Number of attention heads per spine layer.
        spine_head_dim: Dimension per attention head.
        num_zones: Total zones including Zone 0 (Skip). Zone 0 is always index 0.
        zone_hidden_dim: Hidden dimension inside each specialty FFN zone.
        zone_ffn_layers: Total number of linear layers per zone (2 = d→h→d,
            3 = d→h→h→d, etc.). Minimum 2.
        max_seq_len: Maximum sequence length (for positional encodings / KV cache).
        router_top_k: How many zones to activate per token during inference (1 or 2).
        dropout: Dropout probability (0.0 = disabled, recommended for inference).
        use_skip_zone: Whether Zone 0 (skip) is enabled during routing.
        tie_word_embeddings: Share the token embedding matrix with the LM head
            output projection. Saves vocab_size × embed_dim parameters with no
            quality loss (used in GPT-2, Llama, Mistral).
    """

    vocab_size: int = 50_000
    embed_dim: int = 512
    spine_layers: int = 6
    spine_heads: int = 8
    spine_head_dim: int = 64
    num_zones: int = 11        # 0 = Skip Zone, 1-10 = specialty zones
    zone_hidden_dim: int = 1_024
    zone_ffn_layers: int = 2
    max_seq_len: int = 2_048
    router_top_k: int = 1
    dropout: float = 0.0
    use_skip_zone: bool = True
    tie_word_embeddings: bool = True

    # ---------------------------------------------------------------------------
    # Presets
    # ---------------------------------------------------------------------------

    @classmethod
    def tiny(cls) -> "ForestConfig":
        """~61M parameter model for smoke testing (Phase 4.1).

        Actual param count with weight tying: ~61.3M.
        spine_heads * spine_head_dim == embed_dim (512 = 8 * 64).
        zone_ffn_layers=1 SwiGLU block per zone (3 matrices: gate, up, down).
        """
        return cls(
            vocab_size=50_000,
            embed_dim=512,
            spine_layers=6,
            spine_heads=8,
            spine_head_dim=64,
            num_zones=11,
            zone_hidden_dim=1_024,
            zone_ffn_layers=1,
            max_seq_len=2_048,
            router_top_k=1,
            tie_word_embeddings=True,
        )

    @classmethod
    def small(cls) -> "ForestConfig":
        """~130M parameter model for proof-of-concept (Phase 4.2).

        Actual param count with weight tying: ~129.7M.
        spine_heads * spine_head_dim == embed_dim (768 = 12 * 64).
        zone_ffn_layers=1 SwiGLU block per zone.
        """
        return cls(
            vocab_size=50_000,
            embed_dim=768,
            spine_layers=6,
            spine_heads=12,
            spine_head_dim=64,
            num_zones=11,
            zone_hidden_dim=2_048,
            zone_ffn_layers=1,
            max_seq_len=2_048,
            router_top_k=1,
            tie_word_embeddings=True,
        )

    @classmethod
    def base(cls) -> "ForestConfig":
        """~493M parameter model for scaling experiments (Phase 4.3).

        Actual param count with weight tying: ~492.7M.
        spine_heads * spine_head_dim == embed_dim (1024 = 16 * 64).
        zone_ffn_layers=3 SwiGLU blocks per zone, zone_hidden_dim=2560.
        """
        return cls(
            vocab_size=50_000,
            embed_dim=1_024,
            spine_layers=16,
            spine_heads=16,
            spine_head_dim=64,
            num_zones=11,
            zone_hidden_dim=2_560,
            zone_ffn_layers=3,
            max_seq_len=4_096,
            router_top_k=1,
            tie_word_embeddings=True,
        )

    @classmethod
    def large(cls) -> "ForestConfig":
        """~989M parameter model for production use (Phase 5.0).

        Actual param count with weight tying: ~989.2M.
        spine_heads * spine_head_dim == embed_dim (1536 = 24 * 64).
        zone_ffn_layers=3 SwiGLU blocks per zone, zone_hidden_dim=2048.
        """
        return cls(
            vocab_size=50_000,
            embed_dim=1_536,
            spine_layers=22,
            spine_heads=24,
            spine_head_dim=64,
            num_zones=11,
            zone_hidden_dim=2_048,
            zone_ffn_layers=3,
            max_seq_len=4_096,
            router_top_k=2,
            tie_word_embeddings=True,
        )

    # ---------------------------------------------------------------------------
    # Estimation utilities
    # ---------------------------------------------------------------------------

    def estimate_params(self) -> dict[str, int]:
        """Estimate parameter counts broken down by component.

        Zone formula (SwiGLU): each zone has zone_ffn_layers SwiGLU blocks.
        Each block has 3 weight matrices (gate d→h, up d→h, down h→d) = 3*d*h params.
        Plus one RMSNorm weight vector (d params) per zone.
        Total per zone: zone_ffn_layers * 3 * d * h + d.

        Returns:
            Dict with keys: embedding, spine, zones, router, lm_head, total.
            When tie_word_embeddings=True, lm_head=0 (shared with embedding).
            The sum of all values (excluding 'total') always equals total.
        """
        d = self.embed_dim

        # Token embedding + learned positional embedding
        embedding = self.vocab_size * d + self.max_seq_len * d

        # Spine: Q, K, V, O projections + MLP (4x expansion) + two LayerNorms per layer
        spine_attn_per_layer = 4 * d * (self.spine_heads * self.spine_head_dim)
        spine_mlp_per_layer = 2 * d * (4 * d)
        spine_ln_per_layer = 2 * 2 * d
        spine = self.spine_layers * (
            spine_attn_per_layer + spine_mlp_per_layer + spine_ln_per_layer
        )

        # Zones: Zone 0 (Skip) has no params; count (num_zones - 1) specialty zones.
        # Each zone: zone_ffn_layers SwiGLU blocks (3 matrices each) + 1 RMSNorm weight.
        h = self.zone_hidden_dim
        zone_params_per_zone = self.zone_ffn_layers * 3 * d * h + d
        zones = (self.num_zones - 1) * zone_params_per_zone

        # Router: single linear from embed_dim to num_zones
        router = d * self.num_zones

        # LM head: vocab_size × embed_dim, or 0 if tied with embedding
        lm_head = 0 if self.tie_word_embeddings else self.vocab_size * d

        total = embedding + spine + zones + router + lm_head

        return {
            "embedding": embedding,
            "spine": spine,
            "zones": zones,
            "router": router,
            "lm_head": lm_head,
            "total": total,
        }

    def estimate_vram_gb(self, batch_size: int = 1, seq_len: int = 512) -> float:
        """Rough VRAM estimate for the HOT tier (spine + embeddings) in bf16.

        When tie_word_embeddings=True the LM head shares the embedding matrix,
        so it does not contribute additional VRAM (lm_head=0 in the param dict).

        Args:
            batch_size: Training or inference batch size.
            seq_len: Sequence length.

        Returns:
            Estimated VRAM usage in GB (bf16 = 2 bytes/param).
        """
        params = self.estimate_params()
        hot_params = (
            params["embedding"]
            + params["spine"]
            + params["router"]
            + params["lm_head"]   # 0 when tied — no double-counting
        )
        param_bytes = hot_params * 2  # bf16

        kv_bytes = (
            2 * self.spine_layers * batch_size * seq_len
            * self.spine_heads * self.spine_head_dim * 2
        )

        act_bytes = batch_size * seq_len * self.embed_dim * self.spine_layers * 2 * 2

        total_bytes = param_bytes + kv_bytes + act_bytes
        return round(total_bytes / (1024 ** 3), 3)

    def __repr__(self) -> str:
        params = self.estimate_params()
        total_m = params["total"] / 1e6
        embed_m = params["embedding"] / 1e6
        spine_m = params["spine"] / 1e6
        zone_m = params["zones"] / 1e6
        vram = self.estimate_vram_gb()

        if self.tie_word_embeddings:
            embed_str = f"embedding={embed_m:.1f}M (tied with LM head)"
        else:
            lm_head_m = params["lm_head"] / 1e6
            embed_str = f"embedding={embed_m:.1f}M, lm_head={lm_head_m:.1f}M"

        return (
            f"ForestConfig("
            f"total={total_m:.1f}M | "
            f"{embed_str} | "
            f"spine={spine_m:.1f}M ({self.spine_layers}L) | "
            f"zones={zone_m:.1f}M ({self.num_zones - 1} specialty) | "
            f"d={self.embed_dim} | "
            f"vram~{vram}GB"
            f")"
        )
