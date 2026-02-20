"""
FORDE Decoder-Only LLM Model.

A decoder-only autoregressive language model integrating:
1. Mixture of Experts (MoE) - Adaptive expert routing
2. Native Sparse Attention (NSA) - Efficient long-context attention
3. Manifold Constrained Hyper-Connections (mHC) - Enhanced residual connections
4. FORDE StatefulLayer sensing - For adaptive neuron specialization

This module preserves the original FORDE dual-encoder in model.py
and introduces the decoder-only LLM as a separate architecture.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Tuple
from dataclasses import dataclass

# Handle imports for both package and script execution
try:
    from src.forde.moe import MoELayer, MoEStatefulLayer
    from src.forde.sparse_attention import NativeSparseAttention, CausalSelfAttention
    from src.forde.hyper_connections import (
        HyperConnectionStream,
        ManifoldHyperConnection,
        StreamCollapser,
    )
except ModuleNotFoundError:
    from moe import MoELayer, MoEStatefulLayer
    from sparse_attention import NativeSparseAttention, CausalSelfAttention
    from hyper_connections import (
        HyperConnectionStream,
        ManifoldHyperConnection,
        StreamCollapser,
    )


@dataclass
class LLMConfig:
    """Configuration for the decoder-only LLM."""

    vocab_size: int = 32000
    d_model: int = 512
    num_layers: int = 12
    num_heads: int = 8
    head_dim: int = 64
    max_seq_len: int = 2048

    # MoE configuration
    use_moe: bool = True
    num_experts: int = 8
    top_k_experts: int = 2
    expert_hidden_dim: int = 2048
    moe_aux_loss_weight: float = 0.01

    # NSA configuration
    use_sparse_attention: bool = True
    window_size: int = 512
    compression_ratio: int = 8
    top_k_global: int = 64

    # mHC configuration
    use_hyper_connections: bool = True
    num_streams: int = 4
    sinkhorn_iterations: int = 5

    # Dropout
    dropout_rate: float = 0.1


class DecoderBlock(nn.Module):
    """
    Single decoder block with NSA + MoE + mHC integration.

    Architecture:
    1. Pre-norm + Sparse Attention (or standard causal attention)
    2. mHC residual mixing
    3. Pre-norm + MoE FFN (or standard FFN)
    4. mHC residual mixing
    """

    config: LLMConfig

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        streams: Optional[jnp.ndarray] = None,
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Forward pass through decoder block.

        Args:
            x: Input tensor (batch, seq, d_model)
            streams: Optional mHC streams (batch, seq, num_streams, d_model)
            mask: Optional attention mask
            deterministic: If False, apply dropout

        Returns:
            Tuple of (output, updated_streams, moe_aux_loss)
        """
        config = self.config
        batch_size, seq_len, d_model = x.shape

        # Initialize streams if using mHC
        if config.use_hyper_connections:
            if streams is None:
                stream_init = HyperConnectionStream(
                    num_streams=config.num_streams,
                    d_model=config.d_model,
                    name="stream_init",
                )
                streams = stream_init(x)
            working_input = streams[:, :, 0, :]  # Use first stream as main
        else:
            working_input = x

        # ===== ATTENTION SUBLAYER =====
        # Pre-norm
        attn_input = nn.LayerNorm(name="attn_norm")(working_input)

        # Sparse or dense attention
        if config.use_sparse_attention:
            attn_output = NativeSparseAttention(
                num_heads=config.num_heads,
                head_dim=config.head_dim,
                window_size=config.window_size,
                compression_ratio=config.compression_ratio,
                top_k_global=config.top_k_global,
                name="sparse_attention",
            )(attn_input, mask)
        else:
            attn_output = CausalSelfAttention(
                num_heads=config.num_heads,
                head_dim=config.head_dim,
                name="causal_attention",
            )(attn_input, mask)

        # Dropout
        if not deterministic:
            attn_output = nn.Dropout(rate=config.dropout_rate)(
                attn_output, deterministic=deterministic
            )

        # Residual connection (with or without mHC)
        if config.use_hyper_connections:
            mhc_attn = ManifoldHyperConnection(
                num_streams=config.num_streams,
                sinkhorn_iterations=config.sinkhorn_iterations,
                name="mhc_attn",
            )
            streams, working_input = mhc_attn(streams, attn_output, output_stream_idx=0)
        else:
            working_input = working_input + attn_output

        # ===== FFN/MoE SUBLAYER =====
        # Pre-norm
        ffn_input = nn.LayerNorm(name="ffn_norm")(working_input)

        # MoE or standard FFN
        if config.use_moe:
            moe_layer = MoEStatefulLayer(
                num_experts=config.num_experts,
                top_k=config.top_k_experts,
                expert_hidden_dim=config.expert_hidden_dim,
                d_model=config.d_model,
                aux_loss_weight=config.moe_aux_loss_weight,
                name="moe",
            )
            ffn_output, moe_aux_loss = moe_layer(ffn_input)
        else:
            # Standard FFN
            ffn_output = nn.Dense(config.expert_hidden_dim, name="ffn_up")(ffn_input)
            ffn_output = nn.gelu(ffn_output)
            ffn_output = nn.Dense(config.d_model, name="ffn_down")(ffn_output)
            moe_aux_loss = jnp.array(0.0)

        # Dropout
        if not deterministic:
            ffn_output = nn.Dropout(rate=config.dropout_rate)(
                ffn_output, deterministic=deterministic
            )

        # Residual connection (with or without mHC)
        if config.use_hyper_connections:
            mhc_ffn = ManifoldHyperConnection(
                num_streams=config.num_streams,
                sinkhorn_iterations=config.sinkhorn_iterations,
                name="mhc_ffn",
            )
            streams, output = mhc_ffn(streams, ffn_output, output_stream_idx=0)
        else:
            output = working_input + ffn_output
            streams = None

        return output, streams, moe_aux_loss


class FORDEDecoderLM(nn.Module):
    """
    Complete decoder-only language model with FORDE enhancements.

    Integrates MoE, NSA, and mHC for efficient and powerful LLM pretraining.
    """

    config: LLMConfig

    @nn.compact
    def __call__(
        self,
        input_ids: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Forward pass through the decoder LLM.

        Args:
            input_ids: Token indices (batch, seq_len)
            mask: Optional attention mask
            deterministic: If False, apply dropout

        Returns:
            Tuple of (logits, total_aux_loss)
            - logits: (batch, seq_len, vocab_size)
            - total_aux_loss: Scalar sum of MoE auxiliary losses
        """
        config = self.config
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        token_embedding = nn.Embed(
            num_embeddings=config.vocab_size,
            features=config.d_model,
            name="token_embed",
        )(input_ids)

        # Positional embeddings (learned)
        position_ids = jnp.arange(seq_len)[None, :]
        position_embedding = nn.Embed(
            num_embeddings=config.max_seq_len, features=config.d_model, name="pos_embed"
        )(position_ids)

        # Combine embeddings
        x = token_embedding + position_embedding

        # Apply dropout to embeddings
        if not deterministic:
            x = nn.Dropout(rate=config.dropout_rate)(x, deterministic=deterministic)

        # Initialize streams for mHC (if enabled)
        streams = None
        if config.use_hyper_connections:
            stream_init = HyperConnectionStream(
                num_streams=config.num_streams,
                d_model=config.d_model,
                name="initial_streams",
            )
            streams = stream_init(x)

        # Track total auxiliary loss from MoE layers
        total_aux_loss = jnp.array(0.0)

        # Process through decoder blocks
        for layer_idx in range(config.num_layers):
            x, streams, moe_aux_loss = DecoderBlock(
                config=config, name=f"layer_{layer_idx}"
            )(x, streams, mask, deterministic)

            total_aux_loss = total_aux_loss + moe_aux_loss

        # Final layer norm
        x = nn.LayerNorm(name="final_norm")(x)

        # Collapse streams if using mHC
        if config.use_hyper_connections and streams is not None:
            collapser = StreamCollapser(
                d_model=config.d_model,
                collapse_method="weighted_sum",
                name="stream_collapser",
            )
            x = collapser(streams)

        # Language modeling head (project to vocabulary)
        logits = nn.Dense(
            config.vocab_size,
            name="lm_head",
            kernel_init=nn.initializers.normal(stddev=0.02),
        )(x)

        return logits, total_aux_loss


class FORDEDecoderLMWithLoss(nn.Module):
    """
    Wrapper that includes loss computation for training convenience.
    """

    config: LLMConfig

    @nn.compact
    def __call__(
        self,
        input_ids: jnp.ndarray,
        labels: Optional[jnp.ndarray] = None,
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Forward pass with optional loss computation.

        Args:
            input_ids: Token indices (batch, seq_len)
            labels: Target token indices for loss (batch, seq_len), optional
            mask: Optional attention mask
            deterministic: If False, apply dropout

        Returns:
            Tuple of (logits, lm_loss, aux_loss)
            - logits: (batch, seq_len, vocab_size)
            - lm_loss: Scalar language modeling loss (or 0 if no labels)
            - aux_loss: Scalar MoE auxiliary loss
        """
        # Get model predictions
        model = FORDEDecoderLM(config=self.config, name="decoder")
        logits, aux_loss = model(input_ids, mask, deterministic)

        # Compute loss if labels provided
        if labels is not None:
            # Shift for next-token prediction
            # logits[:, :-1] predicts labels[:, 1:]
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]

            # Cross-entropy loss
            lm_loss = optax.softmax_cross_entropy_with_integer_labels(
                shift_logits.reshape(-1, self.config.vocab_size),
                shift_labels.reshape(-1),
            ).mean()
        else:
            lm_loss = jnp.array(0.0)

        return logits, lm_loss, aux_loss


def create_default_config() -> LLMConfig:
    """Create a reasonable default configuration for testing."""
    return LLMConfig(
        vocab_size=50257,
        d_model=256,
        num_layers=4,
        num_heads=4,
        head_dim=64,
        max_seq_len=1024,
        use_moe=True,
        num_experts=4,
        top_k_experts=2,
        expert_hidden_dim=512,
        use_sparse_attention=True,
        window_size=128,
        compression_ratio=4,
        top_k_global=32,
        use_hyper_connections=True,
        num_streams=2,
        sinkhorn_iterations=3,
        dropout_rate=0.0,
    )


if __name__ == "__main__":
    import optax  # Required for loss computation

    print("=" * 60)
    print("Testing FORDE Decoder-Only LLM")
    print("=" * 60)

    key = jax.random.PRNGKey(42)
    config = create_default_config()

    # Test inputs
    batch_size = 2
    seq_len = 64
    input_ids = jax.random.randint(key, (batch_size, seq_len), 0, config.vocab_size)

    print("\nConfiguration:")
    print(f"  - d_model: {config.d_model}")
    print(f"  - num_layers: {config.num_layers}")
    print(f"  - num_heads: {config.num_heads}")
    print(f"  - vocab_size: {config.vocab_size}")
    print(
        f"  - MoE: {config.use_moe} ({config.num_experts} experts, top-{config.top_k_experts})"
    )
    print(f"  - NSA: {config.use_sparse_attention} (window={config.window_size})")
    print(f"  - mHC: {config.use_hyper_connections} ({config.num_streams} streams)")

    # Initialize model
    print("\nInitializing model...")
    model = FORDEDecoderLM(config=config)
    variables = model.init(key, input_ids)

    # Count parameters
    param_count = sum(x.size for x in jax.tree.leaves(variables["params"]))
    print(f"Total parameters: {param_count:,}")

    # Forward pass
    print("\nRunning forward pass...")
    # Fix: stats_buffer is mutable
    (logits, aux_loss), updated_vars = model.apply(variables, input_ids, mutable=["stats_buffer"])

    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected shape: ({batch_size}, {seq_len}, {config.vocab_size})")
    print(f"MoE aux loss: {aux_loss:.6f}")

    # Verify shapes
    assert logits.shape == (batch_size, seq_len, config.vocab_size), (
        "Output shape mismatch!"
    )
    print("\n✓ Forward pass successful!")

    # Test with loss computation
    print("\nTesting loss computation...")
    labels = jax.random.randint(key, (batch_size, seq_len), 0, config.vocab_size)

    # Manual loss computation
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    lm_loss = optax.softmax_cross_entropy_with_integer_labels(
        shift_logits.reshape(-1, config.vocab_size), shift_labels.reshape(-1)
    ).mean()

    print(f"Language modeling loss: {lm_loss:.4f}")
    print(f"Total loss (LM + aux): {lm_loss + aux_loss:.4f}")

    # Test gradient computation
    print("\nTesting gradient computation...")

    def loss_fn(params, input_ids, labels):
        (logits, aux_loss), _ = model.apply(
            {"params": params, "stats_buffer": variables["stats_buffer"]},
            input_ids,
            mutable=["stats_buffer"]
        )
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        lm_loss = optax.softmax_cross_entropy_with_integer_labels(
            shift_logits.reshape(-1, config.vocab_size), shift_labels.reshape(-1)
        ).mean()
        return lm_loss + aux_loss

    grads = jax.grad(loss_fn)(variables["params"], input_ids, labels)
    grad_norm = jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree.leaves(grads)))
    print(f"Gradient norm: {grad_norm:.4f}")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
