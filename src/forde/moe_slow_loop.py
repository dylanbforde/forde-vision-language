"""
FORDE Slow Loop for MoE-based LLM.

Adapts the original FORDE slow loop (sense, cluster, smooth, actuate) for
Mixture of Experts architecture. Instead of per-neuron assignments, this
version tracks:

1. **Expert-level statistics**: How often each expert is used, what types
   of tokens it processes, specialization patterns
2. **Neuron-level within experts**: Which neurons within experts are
   specialist vs generalist
3. **Router adaptation**: Optionally adjust router biases based on
   expert utilization and specialization

The goal is to enable emergent expert specialization during training,
where some experts naturally become specialists for certain token
types while others remain generalists.
"""

import jax
import jax.numpy as jnp
from flax.core import unfreeze
from typing import Dict, Tuple, Any

# Handle imports
try:
    from src.forde.sensing import calculate_neuron_stats, hoyer_sparsity
    from src.forde.clustering import cluster_neurons_gmm
except ModuleNotFoundError:
    from sensing import calculate_neuron_stats, hoyer_sparsity
    from clustering import cluster_neurons_gmm


def calculate_expert_stats(router_probs: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate statistics for each expert based on routing patterns.

    Args:
        router_probs: (batch, seq, num_experts) - Router probability distribution

    Returns:
        (num_experts, D) array of expert statistics
    """
    num_experts = router_probs.shape[-1]

    # 1. Usage frequency: How often is each expert selected?
    usage_freq = router_probs.mean(axis=(0, 1))  # (num_experts,)

    # 2. Usage variance: How consistent is the expert's usage?
    usage_var = router_probs.var(axis=(0, 1))  # (num_experts,)

    # 3. Selection entropy: How "peaked" is the distribution when this expert is selected?
    # Higher entropy = more uncertain selection
    eps = 1e-8
    entropy = (
        -(router_probs * jnp.log(router_probs + eps)).sum(axis=-1).mean(axis=(0, 1))
    )
    entropy_per_expert = jnp.full(
        num_experts, entropy
    )  # Same for all experts in basic version

    # 4. Load imbalance: Deviation from uniform distribution
    uniform = 1.0 / num_experts
    load_imbalance = jnp.abs(usage_freq - uniform)

    # 5. Sparsity of routing (per expert - how "confidently" is this expert selected)
    # When an expert is in top-k, what's the average weight?
    max_probs_mask = router_probs == router_probs.max(axis=-1, keepdims=True)
    expert_selection_confidence = jnp.where(max_probs_mask, router_probs, 0.0).sum(
        axis=(0, 1)
    ) / (max_probs_mask.sum(axis=(0, 1)) + eps)

    # Stack into feature vector
    stats = jnp.stack(
        [
            usage_freq,
            usage_var,
            entropy_per_expert,
            load_imbalance,
            expert_selection_confidence,
        ],
        axis=-1,
    )  # (num_experts, 5)

    return stats


class MoESlowLoopState:
    """
    Tracks accumulated statistics for the MoE slow loop.

    This is a simple container for stats that accumulate over the
    "fast" training steps between slow loop executions.
    """

    def __init__(self, num_experts: int, d_model: int, num_layers: int):
        self.num_experts = num_experts
        self.d_model = d_model
        self.num_layers = num_layers

        # Accumulated statistics
        self.step_count = 0

        # Per-expert stats: (num_layers, num_experts, D)
        self.expert_usage_sum = jnp.zeros((num_layers, num_experts))
        self.expert_selection_count = jnp.zeros((num_layers, num_experts))

        # Router entropy tracking
        self.router_entropy_sum = jnp.zeros(num_layers)

    def reset(self):
        """Reset all accumulated stats after slow loop execution."""
        self.step_count = 0
        self.expert_usage_sum = jnp.zeros_like(self.expert_usage_sum)
        self.expert_selection_count = jnp.zeros_like(self.expert_selection_count)
        self.router_entropy_sum = jnp.zeros_like(self.router_entropy_sum)


def collect_moe_stats_from_variables(
    mutable_variables: Dict, num_layers: int, num_experts: int
) -> Tuple[jnp.ndarray, int]:
    """
    Extract MoE statistics from model's mutable variables.

    Args:
        mutable_variables: Model's mutable state containing stats_buffer
        num_layers: Number of model layers
        num_experts: Number of experts per layer

    Returns:
        Tuple of (expert_usage_stats, step_count)
        - expert_usage_stats: (num_layers, num_experts)
        - step_count: Number of accumulated steps
    """
    stats_buffer = mutable_variables.get("stats_buffer", {})

    # Initialize output
    expert_usage = jnp.zeros((num_layers, num_experts))
    step_count = 0

    # Traverse stats_buffer to find expert_usage entries
    def find_expert_usage(pytree, layer_idx=0):
        nonlocal expert_usage, step_count

        if isinstance(pytree, dict):
            if "expert_usage" in pytree:
                # Found expert usage stats
                usage = pytree["expert_usage"]
                if usage.shape[0] == num_experts:
                    expert_usage = expert_usage.at[layer_idx].set(usage)

            if "step_count" in pytree:
                step_count = max(step_count, int(pytree["step_count"]))

            # Recursively search
            for k, v in pytree.items():
                if k.startswith("layer_") or "moe" in k.lower():
                    # Extract layer index if possible
                    try:
                        idx = int(k.split("_")[-1]) if "_" in k else layer_idx
                    except ValueError:
                        idx = layer_idx
                    find_expert_usage(v, idx)
                else:
                    find_expert_usage(v, layer_idx)

    find_expert_usage(stats_buffer)

    return expert_usage, step_count


def cluster_experts(
    expert_stats: jnp.ndarray,
    num_clusters: int = 3,
    random_key: jax.random.PRNGKey = None,
) -> Tuple[jnp.ndarray, Dict]:
    """
    Cluster experts based on their usage statistics.

    Identifies different "roles" among experts:
    - Cluster 0: Generalist (high usage, low specialization)
    - Cluster 1: Specialist (focused usage, high specialization)
    - Cluster 2: Under-utilized (low usage, could be reassigned)

    Args:
        expert_stats: (num_experts, D) or (num_layers * num_experts, D)
        num_clusters: Number of role clusters
        random_key: Random key for GMM

    Returns:
        Tuple of (assignments, gmm_params)
    """
    if random_key is None:
        random_key = jax.random.PRNGKey(0)

    # Flatten if multi-layer
    original_shape = expert_stats.shape
    flat_stats = expert_stats.reshape(-1, expert_stats.shape[-1])

    # Use existing GMM clustering
    assignments, gmm_params = cluster_neurons_gmm(
        flat_stats, num_clusters=num_clusters, random_key=random_key
    )

    return assignments.reshape(original_shape[:-1]), gmm_params


def compute_router_adjustments(
    expert_assignments: jnp.ndarray,
    expert_usage: jnp.ndarray,
    target_balance: float = 0.1,
) -> jnp.ndarray:
    """
    Compute router bias adjustments based on expert clustering.

    The idea is to nudge the router to:
    - Use under-utilized experts more
    - Rely less on over-utilized generalists
    - Preserve specialist routing patterns

    Args:
        expert_assignments: (num_experts,) cluster assignments
        expert_usage: (num_experts,) current usage frequencies
        target_balance: Target maximum deviation from uniform

    Returns:
        (num_experts,) router bias adjustments
    """
    num_experts = expert_usage.shape[0]
    uniform = 1.0 / num_experts

    # Calculate desired adjustment
    # Under-utilized experts get positive bias, over-utilized get negative
    deviation = expert_usage - uniform

    # Scale by how much we want to correct
    # Limit adjustment magnitude
    max_adjustment = 0.1
    adjustments = -deviation * target_balance
    adjustments = jnp.clip(adjustments, -max_adjustment, max_adjustment)

    # Don't adjust specialists as much (preserve their patterns)
    # Assuming cluster 1 = specialist
    specialist_mask = expert_assignments == 1
    adjustments = jnp.where(specialist_mask, adjustments * 0.5, adjustments)

    return adjustments


def moe_slow_loop_step(
    model_params: Dict,
    mutable_variables: Dict,
    config: Any,
    key: jax.random.PRNGKey,
    epoch: int,
    step: int,
) -> Tuple[Dict, Dict, Dict]:
    """
    Perform the FORDE slow loop for MoE.

    Steps:
    1. Sense: Collect expert usage statistics
    2. Cluster: Group experts by role (generalist/specialist/under-utilized)
    3. Analyze: Compute expert specialization metrics
    4. Actuate: Optionally adjust router biases

    Args:
        model_params: Model parameters (may include router params)
        mutable_variables: Mutable state with stats_buffer
        config: Model configuration (LLMConfig)
        key: Random key for clustering
        epoch: Current epoch
        step: Current step

    Returns:
        Tuple of (updated_params, updated_mutable_vars, diagnostics)
    """
    print(f"\n{'=' * 50}")
    print(f"MoE Slow Loop - Epoch {epoch}, Step {step}")
    print(f"{'=' * 50}")

    num_layers = config.num_layers
    num_experts = config.num_experts

    # 1. SENSE: Collect accumulated stats
    expert_usage, step_count = collect_moe_stats_from_variables(
        mutable_variables, num_layers, num_experts
    )

    if step_count == 0:
        print("No stats accumulated yet, skipping slow loop.")
        return model_params, mutable_variables, {"skipped": True}

    # Normalize by step count
    expert_usage = expert_usage / step_count

    print("\n--- Sensing ---")
    print(f"Steps accumulated: {step_count}")
    print(f"Expert usage shape: {expert_usage.shape}")

    # Calculate per-layer expert usage statistics
    for layer_idx in range(min(num_layers, 3)):  # Show first 3 layers
        layer_usage = expert_usage[layer_idx]
        print(f"Layer {layer_idx} expert usage: {layer_usage}")

    # 2. CLUSTER: Group experts by behavior
    print("\n--- Clustering ---")

    # Create feature vector for clustering
    # Use usage statistics across all layers
    usage_mean = expert_usage.mean(axis=0)  # (num_experts,)
    usage_var = expert_usage.var(axis=0)  # (num_experts,)

    # Create simple feature matrix for clustering
    cluster_features = jnp.stack([usage_mean, usage_var], axis=-1)

    key, cluster_key = jax.random.split(key)
    assignments, gmm_params = cluster_experts(
        cluster_features, num_clusters=3, random_key=cluster_key
    )

    # Count experts per cluster
    for c in range(3):
        count = (assignments == c).sum()
        cluster_role = {0: "Generalist", 1: "Specialist", 2: "Under-utilized"}
        print(f"Cluster {c} ({cluster_role.get(c, 'Unknown')}): {count} experts")

    # 3. SMOOTH: Apply 3D smoothing (optional)
    # Reshape assignments to (1, 1, num_experts) for 1D smoothing, or (1, 2, 4) if 8 experts
    # For demonstration, we'll treat it as a 1D line of experts per layer
    # If we had multiple layers, we could smooth across layers too

    # Try to reshape to a grid if num_experts is composite
    grid_h = int(jnp.sqrt(num_experts))
    while num_experts % grid_h != 0:
        grid_h -= 1
    grid_w = num_experts // grid_h

    # Reshape to (1, grid_h, grid_w) for 3D smoothing (treating batch/layer as dim 0)
    # Here we just use 1 layer for simplicity of the demo
    assignment_grid = assignments.reshape(1, grid_h, grid_w)

    try:
        try:
            from src.forde.smoothing import smooth_assignments_3d
        except ImportError:
            from smoothing import smooth_assignments_3d

        print("\n--- Smoothing ---")
        print(f"Reshaped assignments to grid: {assignment_grid.shape}")

        smoothed_grid = smooth_assignments_3d(
            assignment_grid, kernel_size=3, num_clusters=3
        )
        smoothed_assignments = smoothed_grid.flatten()

        # Check changes
        changes = (assignments != smoothed_assignments).sum()
        print(f"Smoothing changed {changes} assignments")
        assignments = smoothed_assignments

    except ImportError:
        print("\n--- Smoothing skipped (function not found) ---")

    # 4. ANALYZE: Compute specialization metrics
    print("\n--- Analysis ---")

    # Expert utilization imbalance
    uniform = 1.0 / num_experts
    imbalance = jnp.abs(usage_mean - uniform).mean()
    print(f"Mean load imbalance: {imbalance:.4f}")

    # Entropy of expert distribution (lower = more specialized routing)
    eps = 1e-8
    routing_entropy = -(usage_mean * jnp.log(usage_mean + eps)).sum()
    max_entropy = jnp.log(num_experts)
    relative_entropy = routing_entropy / max_entropy
    print(f"Routing entropy (relative): {relative_entropy:.4f}")

    # 5. ACTUATE: Update router biases
    print("\n--- Actuation ---")

    # Compute recommended adjustments
    adjustments = compute_router_adjustments(assignments, usage_mean)
    print(f"Recommended router adjustments: {adjustments}")

    # Apply adjustments to model parameters
    # We need to find the router bias parameters in the pytree
    # They are typically named 'router_linear' -> 'bias'

    def update_router_bias(path, param):
        if "router_linear" in path and "bias" in path:
            # Found a router bias!
            # Check shape matches adjustments
            if param.shape == adjustments.shape:
                print(f"Updating router bias at path: {path}")
                return param + adjustments
        return param

    # Traverse and update
    # Helper to reconstruct path for logging
    def map_with_path(fn, tree):
        def _map(path, node):
            if isinstance(node, dict) or hasattr(node, "keys"):
                return {k: _map(path + (k,), v) for k, v in node.items()}
            else:
                return fn(path, node)

        return _map((), tree)

    # Since model_params is FrozenDict, we need to unfreeze/freeze or use tree_map
    # But tree_map doesn't give paths. We can use flax.traverse_util
    from flax import traverse_util

    flat_params = traverse_util.flatten_dict(unfreeze(model_params))
    updated_flat_params = {}

    updates_count = 0
    for path, param in flat_params.items():
        # Check if this is a router bias
        # Path is tuple like ('params', 'layers_0', 'moe', 'router_linear', 'bias')
        if "router_linear" in path and "bias" in path:
            if param.shape == adjustments.shape:
                updated_flat_params[path] = param + adjustments
                updates_count += 1
            else:
                updated_flat_params[path] = param
        else:
            updated_flat_params[path] = param

    if updates_count > 0:
        print(f"Applied updates to {updates_count} router biases")
        updated_params = traverse_util.unflatten_dict(updated_flat_params)
    else:
        print("No matching router biases found to update")
        updated_params = model_params

    # 6. RESET: Clear stats buffer
    def reset_leaf(x):
        return jnp.zeros_like(x)

    mutable_vars_unfrozen = unfreeze(mutable_variables)
    if "stats_buffer" in mutable_vars_unfrozen:
        mutable_vars_unfrozen["stats_buffer"] = jax.tree.map(
            reset_leaf, mutable_vars_unfrozen["stats_buffer"]
        )

    print("\nStats buffer reset.")
    print(f"{'=' * 50}\n")

    # Collect diagnostics
    diagnostics = {
        "expert_usage": usage_mean,
        "assignments": assignments,
        "load_imbalance": imbalance,
        "routing_entropy": relative_entropy,
        "adjustments": adjustments,
        "step_count": step_count,
    }

    return updated_params, mutable_vars_unfrozen, diagnostics


if __name__ == "__main__":
    print("--- Testing MoE Slow Loop Components ---\n")

    key = jax.random.PRNGKey(42)
    num_experts = 8
    batch_size, seq_len = 4, 32

    # Test expert stats calculation
    print("1. Testing calculate_expert_stats:")
    router_probs = jax.nn.softmax(
        jax.random.normal(key, (batch_size, seq_len, num_experts)), axis=-1
    )
    stats = calculate_expert_stats(router_probs)
    print(f"   Router probs shape: {router_probs.shape}")
    print(f"   Expert stats shape: {stats.shape}")
    print(f"   Usage frequencies: {stats[:, 0]}")

    # Test clustering
    print("\n2. Testing cluster_experts:")
    key, cluster_key = jax.random.split(key)
    assignments, gmm = cluster_experts(stats, num_clusters=3, random_key=cluster_key)
    print(f"   Assignments: {assignments}")

    # Test smoothing
    print("\n3. Testing 3D smoothing:")
    # Reshape to (1, 2, 4)
    grid = assignments.reshape(1, 2, 4)
    try:
        try:
            from src.forde.smoothing import smooth_assignments_3d
        except ImportError:
            from smoothing import smooth_assignments_3d

        smoothed = smooth_assignments_3d(grid, kernel_size=3, num_clusters=3)
        print(f"   Smoothed shape: {smoothed.shape}")
        print(f"   Changes: {(grid != smoothed).sum()}")
    except ImportError:
        print("   Smoothing function not found (check path)")

    # Test router adjustments
    print("\n4. Testing compute_router_adjustments:")
    usage = stats[:, 0]  # Usage frequency
    adjustments = compute_router_adjustments(assignments, usage)
    print(f"   Current usage: {usage}")
    print(f"   Adjustments: {adjustments}")

    print("\n--- MoE Slow Loop tests passed! ---")
