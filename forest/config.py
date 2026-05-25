"""ForestConfig: central configuration dataclass for Neural Forest models."""

from __future__ import annotations

from dataclasses import dataclass, field


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
        zone_ffn_layers: Number of linear layers inside each zone (2 or 3).
        max_seq_len: Maximum sequence length (for positional encodings / KV cache).
        router_top_k: How many zones to activate per token during inference (1 or 2).
        dropout: Dropout probability (0.0 = disabled, recommended for inference).
        use_skip_zone: Whether Zone 0 (skip) is enabled during routing.
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

    # ---------------------------------------------------------------------------
    # Presets
    # ---------------------------------------------------------------------------

    @classmethod
    def tiny(cls) -> "ForestConfig":
        """~50M parameter model for smoke testing (Phase 4.1).

        Tuned so embedding + spine + zones + lm_head sums to ~50M.
        spine_heads * spine_head_dim == embed_dim (320 = 8 * 40).
        """
        return cls(
            vocab_size=50_000,
            embed_dim=320,
            spine_layers=6,
            spine_heads=8,
            spine_head_dim=40,
            num_zones=11,
            zone_hidden_dim=640,
            zone_ffn_layers=2,
            max_seq_len=2_048,
            router_top_k=1,
        )

    @classmethod
    def small(cls) -> "ForestConfig":
        """~125M parameter model for proof-of-concept (Phase 4.2).

        spine_heads * spine_head_dim == embed_dim (576 = 9 * 64).
        """
        return cls(
            vocab_size=50_000,
            embed_dim=576,
            spine_layers=8,
            spine_heads=9,
            spine_head_dim=64,
            num_zones=11,
            zone_hidden_dim=1_152,
            zone_ffn_layers=2,
            max_seq_len=2_048,
            router_top_k=1,
        )

    @classmethod
    def base(cls) -> "ForestConfig":
        """~500M parameter model for scaling experiments (Phase 4.3).

        spine_heads * spine_head_dim == embed_dim (1024 = 16 * 64).
        """
        return cls(
            vocab_size=50_000,
            embed_dim=1_024,
            spine_layers=12,
            spine_heads=16,
            spine_head_dim=64,
            num_zones=11,
            zone_hidden_dim=2_560,
            zone_ffn_layers=3,
            max_seq_len=4_096,
            router_top_k=1,
        )

    @classmethod
    def large(cls) -> "ForestConfig":
        """~1B parameter model for production use (Phase 5.0).

        spine_heads * spine_head_dim == embed_dim (1536 = 12 * 128).
        """
        return cls(
            vocab_size=50_000,
            embed_dim=1_536,
            spine_layers=16,
            spine_heads=12,
            spine_head_dim=128,
            num_zones=11,
            zone_hidden_dim=3_072,
            zone_ffn_layers=3,
            max_seq_len=4_096,
            router_top_k=2,
        )

    # ---------------------------------------------------------------------------
    # Estimation utilities
    # ---------------------------------------------------------------------------

    def estimate_params(self) -> dict[str, int]:
        """Estimate parameter counts for each component.

        Returns:
            Dictionary with keys: embedding, spine, zones, router, lm_head, total.
        """
        d = self.embed_dim

        # Token embedding + positional embedding
        embedding = self.vocab_size * d + self.max_seq_len * d

        # Spine: each layer has Q,K,V,O projections + MLP (2x) + layer norms
        spine_attn_per_layer = 4 * d * (self.spine_heads * self.spine_head_dim)
        spine_mlp_per_layer = 2 * d * (4 * d)        # 4x expansion ratio
        spine_ln_per_layer = 2 * 2 * d               # 2 LN per layer, weight+bias
        spine = self.spine_layers * (
            spine_attn_per_layer + spine_mlp_per_layer + spine_ln_per_layer
        )

        # Zones: num_zones FFN blocks (Zone 0 has near-zero params — skip path)
        h = self.zone_hidden_dim
        zone_params_per_zone = (
            self.zone_ffn_layers * d * h   # linear weights
            + (self.zone_ffn_layers - 1) * h * h   # inter-layer weights
            + d * h                         # output projection back to d
        )
        # Zone 0 (skip) has effectively 0 params; count specialty zones only
        zones = (self.num_zones - 1) * zone_params_per_zone

        # Router: linear projection from embed_dim -> num_zones
        router = d * self.num_zones

        # LM head (tied with embedding, but count separately)
        lm_head = self.vocab_size * d

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

        Args:
            batch_size: Training or inference batch size.
            seq_len: Sequence length.

        Returns:
            Estimated VRAM usage in GB (bf16 = 2 bytes/param).
        """
        params = self.estimate_params()
        # Only HOT params are always in VRAM: embedding + spine + router + lm_head
        hot_params = params["embedding"] + params["spine"] + params["router"] + params["lm_head"]
        param_bytes = hot_params * 2  # bf16

        # KV cache: 2 tensors (K, V) * spine_layers * batch * seq_len * heads * head_dim
        kv_bytes = (
            2 * self.spine_layers * batch_size * seq_len
            * self.spine_heads * self.spine_head_dim * 2  # bf16
        )

        # Activations: rough estimate (batch * seq * embed * spine_layers * 2)
        act_bytes = batch_size * seq_len * self.embed_dim * self.spine_layers * 2 * 2

        total_bytes = param_bytes + kv_bytes + act_bytes
        return round(total_bytes / (1024 ** 3), 3)

    def __repr__(self) -> str:
        params = self.estimate_params()
        total_m = params["total"] / 1e6
        hot_m = (params["embedding"] + params["spine"] + params["router"] + params["lm_head"]) / 1e6
        zone_m = params["zones"] / 1e6
        vram = self.estimate_vram_gb()
        return (
            f"ForestConfig("
            f"total={total_m:.1f}M, "
            f"hot={hot_m:.1f}M, "
            f"zones={zone_m:.1f}M ({self.num_zones - 1} specialty), "
            f"embed={self.embed_dim}, "
            f"spine={self.spine_layers}L, "
            f"vram~{vram}GB"
            f")"
        )
