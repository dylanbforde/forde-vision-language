"""
Manifold Constrained Hyper-Connections (mHC) for FORDE LLM.

Based on Deepseek's Manifold Constrained Hyper-Connections (arxiv:2512.24880).

This module implements:
1. Hyper-Connections: Expand residual stream into multiple parallel streams
2. Mixing matrices: Cross-stream information flow
3. Manifold constraint: Project mixing matrices to doubly stochastic manifold
   via Sinkhorn-Knopp algorithm to preserve identity mapping property

Key benefits:
- Richer information flow between layers
- Stable training through constrained connectivity
- Identity mapping preservation for gradient flow
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple


def sinkhorn_knopp(
    logits: jnp.ndarray, num_iterations: int = 5, epsilon: float = 1e-8
) -> jnp.ndarray:
    """
    Project logits to a doubly stochastic matrix via Sinkhorn-Knopp algorithm.

    A doubly stochastic matrix has all rows and columns sum to 1.
    This property ensures the mixing preserves information and enables
    stable gradient flow by maintaining the identity mapping property.

    Args:
        logits: Unconstrained matrix of shape (n, n)
        num_iterations: Number of Sinkhorn iterations
        epsilon: Small constant for numerical stability

    Returns:
        Doubly stochastic matrix of shape (n, n)
    """
    # Convert logits to positive values via softplus (smoother than exp)
    M = jax.nn.softplus(logits) + epsilon

    for _ in range(num_iterations):
        # Row normalize
        M = M / (M.sum(axis=1, keepdims=True) + epsilon)
        # Column normalize
        M = M / (M.sum(axis=0, keepdims=True) + epsilon)

    return M


def sinkhorn_knopp_exp(
    logits: jnp.ndarray,
    num_iterations: int = 5,
    temperature: float = 1.0,
    epsilon: float = 1e-8,
) -> jnp.ndarray:
    """
    Sinkhorn-Knopp with exponential (softer) initialization.

    Uses temperature-scaled exponential for smoother optimization landscape.

    Args:
        logits: Unconstrained matrix of shape (n, n)
        num_iterations: Number of Sinkhorn iterations
        temperature: Temperature for softmax-like scaling
        epsilon: Small constant for numerical stability

    Returns:
        Doubly stochastic matrix of shape (n, n)
    """
    M = jnp.exp(logits / temperature) + epsilon

    for _ in range(num_iterations):
        M = M / (M.sum(axis=1, keepdims=True) + epsilon)
        M = M / (M.sum(axis=0, keepdims=True) + epsilon)

    return M


class HyperConnectionStream(nn.Module):
    """
    Manages multiple parallel residual streams with hyper-connectivity.

    Each stream maintains a separate representation that can interact
    with other streams through learned mixing matrices.
    """

    num_streams: int = 4
    d_model: int = 512

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Initialize or expand input into multiple streams.

        Args:
            x: Input tensor (batch, seq, d_model)

        Returns:
            Stream tensor (batch, seq, num_streams, d_model)
        """
        batch_size, seq_len, d_model = x.shape

        # Expand single input to multiple streams
        # First stream is identity, others are learned projections
        streams = []

        # Stream 0: Identity (main residual path)
        streams.append(x)

        # Streams 1 to num_streams-1: Learned projections
        for i in range(1, self.num_streams):
            projection = nn.Dense(
                d_model,
                name=f"stream_init_{i}",
                kernel_init=nn.initializers.normal(stddev=0.02),
            )(x)
            streams.append(projection)

        # Stack into (batch, seq, num_streams, d_model)
        return jnp.stack(streams, axis=2)


class ManifoldHyperConnection(nn.Module):
    """
    Manifold Constrained Hyper-Connection layer.

    Replaces standard residual connection with multiple interacting streams.
    The mixing matrix is constrained to be doubly stochastic to preserve
    the identity mapping property essential for stable deep network training.

    Usage:
        Replace: output = sublayer(x) + x  (standard residual)
        With:    output = mhc(x, sublayer(x))  (hyper-connected)
    """

    num_streams: int = 4
    sinkhorn_iterations: int = 5
    temperature: float = 1.0

    @nn.compact
    def __call__(
        self,
        streams: jnp.ndarray,
        sublayer_output: jnp.ndarray,
        output_stream_idx: int = 0,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Apply manifold-constrained hyper-connection mixing.

        Args:
            streams: Current stream states (batch, seq, num_streams, d_model)
            sublayer_output: Output from attention/FFN sublayer (batch, seq, d_model)
            output_stream_idx: Which stream receives the sublayer output

        Returns:
            Tuple of (mixed_streams, output_for_next_sublayer)
            - mixed_streams: (batch, seq, num_streams, d_model)
            - output_for_next_sublayer: (batch, seq, d_model)
        """
        batch_size, seq_len, num_streams, d_model = streams.shape

        # Learnable mixing matrix (before constraint)
        mixing_logits = self.param(
            "mixing_logits",
            nn.initializers.normal(stddev=0.1),
            (num_streams, num_streams),
        )

        # Project to doubly stochastic matrix via Sinkhorn-Knopp
        mixing_matrix = sinkhorn_knopp_exp(
            mixing_logits,
            num_iterations=self.sinkhorn_iterations,
            temperature=self.temperature,
        )

        # Apply mixing to streams: mix information across streams
        # streams: (batch, seq, num_streams, d_model)
        # mixing_matrix: (num_streams, num_streams)
        # Result: (batch, seq, num_streams, d_model)
        mixed_streams = jnp.einsum("ij,bsjd->bsid", mixing_matrix, streams)

        # Add sublayer output to the designated stream
        output_addition = jnp.zeros_like(mixed_streams)
        output_addition = output_addition.at[:, :, output_stream_idx, :].set(
            sublayer_output
        )
        mixed_streams = mixed_streams + output_addition

        # Extract output for next sublayer (typically stream 0)
        output = mixed_streams[:, :, output_stream_idx, :]

        return mixed_streams, output

    def get_mixing_matrix(self) -> jnp.ndarray:
        """Get the constrained mixing matrix (for visualization/debugging)."""
        mixing_logits = self.get_variable("params", "mixing_logits")
        return sinkhorn_knopp_exp(
            mixing_logits,
            num_iterations=self.sinkhorn_iterations,
            temperature=self.temperature,
        )


class ManifoldHyperConnectionBlock(nn.Module):
    """
    Complete block with mHC-enhanced residual connections.

    This wraps a sublayer (attention or FFN) with manifold-constrained
    hyper-connections instead of standard residual connections.
    """

    num_streams: int = 4
    sinkhorn_iterations: int = 5
    d_model: int = 512

    def setup(self):
        self.mhc = ManifoldHyperConnection(
            num_streams=self.num_streams, sinkhorn_iterations=self.sinkhorn_iterations
        )
        self.layer_norm = nn.LayerNorm()

    def __call__(
        self, streams: jnp.ndarray, sublayer_fn, output_stream_idx: int = 0
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Apply sublayer with mHC residual connection.

        Args:
            streams: Current stream states (batch, seq, num_streams, d_model)
            sublayer_fn: Function to apply (e.g., attention or FFN)
            output_stream_idx: Which stream to use for sublayer input/output

        Returns:
            Tuple of (new_streams, sublayer_output)
        """
        # Get input for sublayer from designated stream
        sublayer_input = streams[:, :, output_stream_idx, :]

        # Pre-norm
        normalized_input = self.layer_norm(sublayer_input)

        # Apply sublayer
        sublayer_output = sublayer_fn(normalized_input)

        # Apply mHC mixing
        new_streams, output = self.mhc(streams, sublayer_output, output_stream_idx)

        return new_streams, output


class StreamCollapser(nn.Module):
    """
    Collapse multiple streams back to single representation.

    Used at the end of the model to combine all streams into
    a single output tensor.
    """

    d_model: int
    collapse_method: str = "weighted_sum"  # "weighted_sum", "concat", "first"

    @nn.compact
    def __call__(self, streams: jnp.ndarray) -> jnp.ndarray:
        """
        Collapse streams to single representation.

        Args:
            streams: (batch, seq, num_streams, d_model)

        Returns:
            output: (batch, seq, d_model)
        """
        batch_size, seq_len, num_streams, d_model = streams.shape

        if self.collapse_method == "first":
            # Just use the first (main) stream
            return streams[:, :, 0, :]

        elif self.collapse_method == "concat":
            # Concatenate and project
            concat = streams.reshape(batch_size, seq_len, num_streams * d_model)
            return nn.Dense(self.d_model, name="collapse_proj")(concat)

        else:  # weighted_sum
            # Learned weighted sum of streams
            weights = self.param("stream_weights", nn.initializers.ones, (num_streams,))
            normalized_weights = jax.nn.softmax(weights)

            # Weighted sum: (batch, seq, num_streams, d_model) * (num_streams,) -> (batch, seq, d_model)
            return jnp.einsum("bsnd,n->bsd", streams, normalized_weights)


def verify_doubly_stochastic(matrix: jnp.ndarray, tolerance: float = 1e-4) -> bool:
    """Verify that a matrix is doubly stochastic."""
    row_sums = matrix.sum(axis=1)
    col_sums = matrix.sum(axis=0)

    rows_ok = jnp.allclose(row_sums, 1.0, atol=tolerance)
    cols_ok = jnp.allclose(col_sums, 1.0, atol=tolerance)

    return bool(rows_ok and cols_ok)


if __name__ == "__main__":
    print("--- Testing Manifold Constrained Hyper-Connections ---")
    key = jax.random.PRNGKey(0)
    batch_size, seq_len, d_model = 2, 16, 256
    num_streams = 4

    # Test Sinkhorn-Knopp
    print("\n1. Testing Sinkhorn-Knopp projection:")
    logits = jax.random.normal(key, (num_streams, num_streams))
    ds_matrix = sinkhorn_knopp_exp(logits, num_iterations=10)
    print(f"   Input logits shape: {logits.shape}")
    print(f"   Output matrix shape: {ds_matrix.shape}")
    print(f"   Row sums: {ds_matrix.sum(axis=1)}")
    print(f"   Col sums: {ds_matrix.sum(axis=0)}")
    print(f"   Is doubly stochastic: {verify_doubly_stochastic(ds_matrix)}")

    # Test stream initialization
    print("\n2. Testing HyperConnectionStream:")
    x = jax.random.normal(key, (batch_size, seq_len, d_model))
    stream_init = HyperConnectionStream(num_streams=num_streams, d_model=d_model)
    stream_vars = stream_init.init(key, x)
    streams = stream_init.apply(stream_vars, x)
    print(f"   Input shape: {x.shape}")
    print(f"   Streams shape: {streams.shape}")

    # Test ManifoldHyperConnection
    print("\n3. Testing ManifoldHyperConnection:")
    sublayer_out = jax.random.normal(key, (batch_size, seq_len, d_model))
    mhc = ManifoldHyperConnection(num_streams=num_streams, sinkhorn_iterations=5)
    mhc_vars = mhc.init(key, streams, sublayer_out)
    new_streams, output = mhc.apply(mhc_vars, streams, sublayer_out)
    print(f"   Input streams: {streams.shape}")
    print(f"   Sublayer output: {sublayer_out.shape}")
    print(f"   New streams: {new_streams.shape}")
    print(f"   Output: {output.shape}")

    # Verify mixing matrix is doubly stochastic
    mixing_logits = mhc_vars["params"]["mixing_logits"]
    mixing_matrix = sinkhorn_knopp_exp(mixing_logits, num_iterations=5)
    print(
        f"   Mixing matrix doubly stochastic: {verify_doubly_stochastic(mixing_matrix)}"
    )

    # Test stream collapsing
    print("\n4. Testing StreamCollapser:")
    collapser = StreamCollapser(d_model=d_model, collapse_method="weighted_sum")
    collapse_vars = collapser.init(key, new_streams)
    collapsed = collapser.apply(collapse_vars, new_streams)
    print(f"   Streams: {new_streams.shape}")
    print(f"   Collapsed output: {collapsed.shape}")

    print("\n--- Manifold Hyper-Connections test passed! ---")
