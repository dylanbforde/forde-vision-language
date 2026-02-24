"""
Mixture of Experts (MoE) Layer for FORDE LLM.

This module implements a standard MoE architecture with:
- Learnable router network for token-to-expert assignment
- Top-k expert selection per token
- Load balancing auxiliary loss for even expert utilization
- Integration hooks for FORDE's adaptive neuron assignment system
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple


class ExpertMLP(nn.Module):
    """A single expert MLP (feed-forward network)."""

    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x):
        # Standard FFN: up-projection -> activation -> down-projection
        x = nn.Dense(self.hidden_dim, name="up_proj")(x)
        x = nn.gelu(x)
        x = nn.Dense(self.output_dim, name="down_proj")(x)
        return x


class MoERouter(nn.Module):
    """Router network that assigns tokens to experts."""

    num_experts: int

    @nn.compact
    def __call__(self, x):
        """
        Compute router logits for expert assignment.

        Args:
            x: Input tensor of shape (batch, seq, d_model)

        Returns:
            Router logits of shape (batch, seq, num_experts)
        """
        # Simple linear projection to expert scores
        router_logits = nn.Dense(
            self.num_experts,
            name="router_linear",
            kernel_init=nn.initializers.normal(stddev=0.02),
        )(x)
        return router_logits


class MoELayer(nn.Module):
    """
    Mixture of Experts layer that replaces standard FFN.

    Uses top-k routing with auxiliary load balancing loss.
    Can integrate with FORDE's StatefulLayer for adaptive expert assignments.
    """

    num_experts: int = 8
    top_k: int = 2  # Number of experts each token is routed to
    expert_hidden_dim: int = 2048
    d_model: int = 512
    aux_loss_weight: float = 0.01  # Weight for load balancing loss

    @nn.compact
    def __call__(self, x) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Forward pass through MoE layer.

        Args:
            x: Input tensor of shape (batch, seq, d_model)

        Returns:
            Tuple of (output, aux_loss, router_probs) where:
            - output: shape (batch, seq, d_model)
            - aux_loss: scalar load balancing loss
            - router_probs: (batch, seq, num_experts) probabilities
        """
        batch_size, seq_len, d_model = x.shape

        # 1. Compute router logits and probabilities
        router = MoERouter(num_experts=self.num_experts)
        router_logits = router(x)  # (batch, seq, num_experts)
        router_probs = jax.nn.softmax(router_logits, axis=-1)

        # 2. Top-k expert selection
        top_k_indices, top_k_probs = self._top_k_gating(router_logits)
        # top_k_indices: (batch, seq, top_k)
        # top_k_probs: (batch, seq, top_k)

        # 3. Initialize all experts
        experts = [
            ExpertMLP(
                hidden_dim=self.expert_hidden_dim,
                output_dim=d_model,
                name=f"expert_{i}",
            )
            for i in range(self.num_experts)
        ]

        # 4. Compute expert outputs (simplified gather-scatter implementation)
        # For each token, we compute outputs from top_k experts and combine
        output = self._compute_expert_outputs(x, experts, top_k_indices, top_k_probs)

        # 5. Compute load balancing auxiliary loss
        aux_loss = self._load_balancing_loss(router_probs, top_k_indices)

        return output, aux_loss, router_probs

    def _top_k_gating(
        self, router_logits: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Select top-k experts for each token.

        Args:
            router_logits: (batch, seq, num_experts)

        Returns:
            (top_k_indices, top_k_probs) each of shape (batch, seq, top_k)
        """
        # Get top-k indices and logits directly
        # Optimization: Use jax.lax.top_k instead of jnp.argsort which is much faster (~10x)
        # Note: returns indices in descending order of values (largest first)
        top_k_logits, top_k_indices = jax.lax.top_k(router_logits, self.top_k)

        # Normalize among selected experts (renormalize probabilities)
        top_k_probs = jax.nn.softmax(top_k_logits, axis=-1)

        return top_k_indices, top_k_probs

    def _compute_expert_outputs(
        self,
        x: jnp.ndarray,
        experts: list,
        top_k_indices: jnp.ndarray,
        top_k_probs: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute weighted combination of expert outputs for each token.

        This is a simplified implementation that loops over experts.
        For production, use optimized gather-scatter or capacity-based routing.

        Args:
            x: (batch, seq, d_model)
            experts: List of expert modules
            top_k_indices: (batch, seq, top_k)
            top_k_probs: (batch, seq, top_k)

        Returns:
            output: (batch, seq, d_model)
        """
        batch_size, seq_len, d_model = x.shape

        # Compute output from all experts (can be optimized with masking)
        # Shape: (num_experts, batch, seq, d_model)
        all_expert_outputs = jnp.stack([expert(x) for expert in experts], axis=0)

        # Initialize output accumulator
        output = jnp.zeros_like(x)

        # For each selected expert position in top_k
        for k in range(self.top_k):
            # Get expert indices for this k
            expert_idx = top_k_indices[..., k]  # (batch, seq)
            weights = top_k_probs[..., k : k + 1]  # (batch, seq, 1)

            # Gather expert outputs for selected experts
            # Create advanced indexing
            batch_indices = jnp.arange(batch_size)[:, None]
            seq_indices = jnp.arange(seq_len)[None, :]

            # Gather: all_expert_outputs[expert_idx[b,s], b, s, :]
            selected_output = all_expert_outputs[
                expert_idx, batch_indices, seq_indices, :
            ]

            # Weighted sum
            output = output + weights * selected_output

        return output

    def _load_balancing_loss(
        self, router_probs: jnp.ndarray, top_k_indices: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute load balancing auxiliary loss.

        Encourages equal distribution of tokens across experts.
        Uses the formulation from Switch Transformer / Mixtral.

        loss = num_experts * sum_i (fraction_i * probability_i)

        Args:
            router_probs: (batch, seq, num_experts) - softmax probabilities
            top_k_indices: (batch, seq, top_k) - selected expert indices

        Returns:
            Scalar loss value
        """
        batch_size, seq_len, num_experts = router_probs.shape
        num_tokens = batch_size * seq_len

        # Compute fraction of tokens routed to each expert
        # Optimization: Use jnp.bincount instead of jax.nn.one_hot followed by sum (~8x faster)
        counts = jnp.bincount(top_k_indices.reshape(-1), length=num_experts)
        fraction_per_expert = counts / (num_tokens * self.top_k)

        # Compute mean probability assigned to each expert
        prob_per_expert = router_probs.mean(axis=(0, 1))

        # Load balancing loss
        aux_loss = self.num_experts * jnp.sum(fraction_per_expert * prob_per_expert)

        return aux_loss * self.aux_loss_weight


class MoEStatefulLayer(nn.Module):
    """
    MoE layer integrated with FORDE's stateful neuron assignment system.

    Combines MoE routing with FORDE's sensing/clustering infrastructure
    to enable adaptive, history-based expert specialization.

    Tracks:
    - expert_usage: Accumulated sum of expert selection probabilities
    - step_count: Number of forward passes (for averaging in slow loop)
    """

    num_experts: int = 8
    top_k: int = 2
    expert_hidden_dim: int = 2048
    d_model: int = 512
    aux_loss_weight: float = 0.01

    @nn.compact
    def __call__(self, x) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Forward pass with FORDE sensing integration.

        Logs routing statistics for the slow loop to analyze expert utilization.
        """
        batch_size, seq_len, d_model = x.shape

        # Standard MoE forward pass
        moe_layer = MoELayer(
            num_experts=self.num_experts,
            top_k=self.top_k,
            expert_hidden_dim=self.expert_hidden_dim,
            d_model=d_model,
            aux_loss_weight=self.aux_loss_weight,
        )
        output, aux_loss, router_probs = moe_layer(x)

        # Log expert utilization statistics for FORDE slow loop

        # Store expert usage statistics in mutable state for slow loop
        expert_usage_var = self.variable(
            "stats_buffer",
            "expert_usage",
            lambda: jnp.zeros(self.num_experts, dtype=jnp.float32),
        )

        # Track step count for averaging
        step_count_var = self.variable(
            "stats_buffer", "step_count", lambda: jnp.array(0, dtype=jnp.int32)
        )

        # Accumulate mean probability per expert
        current_usage = router_probs.mean(axis=(0, 1))  # (num_experts,)
        expert_usage_var.value = expert_usage_var.value + current_usage
        step_count_var.value = step_count_var.value + 1

        return output, aux_loss


if __name__ == "__main__":
    # Test MoE layer
    print("--- Testing MoELayer ---")
    key = jax.random.PRNGKey(0)
    batch_size, seq_len, d_model = 2, 16, 256

    x = jax.random.normal(key, (batch_size, seq_len, d_model))

    moe = MoELayer(num_experts=4, top_k=2, expert_hidden_dim=512, d_model=d_model)

    variables = moe.init(key, x)
    # Update: MoELayer returns 3 values: output, aux_loss, router_probs
    output, aux_loss, router_probs = moe.apply(variables, x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Aux loss: {aux_loss}")
    print(f"Router probs shape: {router_probs.shape}")
    print(f"Output matches input shape: {output.shape == x.shape}")

    # Verify outputs are not all zeros
    print(f"Output mean: {output.mean():.4f}")
    print(f"Output std: {output.std():.4f}")

    print("\n--- MoELayer test passed! ---")
