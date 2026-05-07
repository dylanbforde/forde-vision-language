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
from collections.abc import Mapping
from typing import Dict, Tuple, Any

# Handle imports
try:
    from src.forde.clustering import cluster_neurons_gmm
    from src.forde.ot_assignment import (
        assign_expert_roles_ot,
        ot_config_from_model_config,
    )
except ModuleNotFoundError:
    from clustering import cluster_neurons_gmm
    from ot_assignment import assign_expert_roles_ot, ot_config_from_model_config


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
    # Optimized: Use bincount instead of where/sum to avoid memory allocation overhead
    max_probs = router_probs.max(axis=-1).reshape(-1)
    max_indices = router_probs.argmax(axis=-1).reshape(-1)

    expert_confidence_sum = jnp.bincount(
        max_indices, weights=max_probs, length=num_experts
    )
    expert_count = jnp.bincount(max_indices, length=num_experts)

    expert_selection_confidence = expert_confidence_sum / (expert_count + eps)

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
) -> Tuple[Dict[str, jnp.ndarray], int]:
    """
    Extract MoE statistics from model's mutable variables.

    Args:
        mutable_variables: Model's mutable state containing stats_buffer
        num_layers: Number of model layers
        num_experts: Number of experts per layer

    Returns:
        Tuple of (expert_stats, step_count)
        - expert_stats: Dict of arrays keyed by statistic name
        - step_count: Number of accumulated steps
    """
    stats_buffer = mutable_variables.get("stats_buffer", {})

    # Initialize output
    expert_usage = jnp.zeros((num_layers, num_experts))
    expert_usage_sq = jnp.zeros((num_layers, num_experts))
    expert_top1_confidence_sum = jnp.zeros((num_layers, num_experts))
    expert_top1_count = jnp.zeros((num_layers, num_experts))
    router_entropy = jnp.zeros((num_layers,))
    token_count = jnp.zeros((num_layers,), dtype=jnp.int32)
    step_count = 0

    # Traverse stats_buffer to find expert_usage entries
    def find_expert_usage(pytree, layer_idx=0):
        nonlocal expert_usage
        nonlocal expert_usage_sq
        nonlocal expert_top1_confidence_sum
        nonlocal expert_top1_count
        nonlocal router_entropy
        nonlocal token_count
        nonlocal step_count

        if isinstance(pytree, Mapping):
            if "expert_usage" in pytree:
                # Found expert usage stats
                usage = pytree["expert_usage"]
                if usage.shape[0] == num_experts:
                    expert_usage = expert_usage.at[layer_idx].set(usage)

            if "expert_usage_sq" in pytree:
                usage_sq = pytree["expert_usage_sq"]
                if usage_sq.shape[0] == num_experts:
                    expert_usage_sq = expert_usage_sq.at[layer_idx].set(usage_sq)

            if "expert_top1_confidence_sum" in pytree:
                confidence_sum = pytree["expert_top1_confidence_sum"]
                if confidence_sum.shape[0] == num_experts:
                    expert_top1_confidence_sum = expert_top1_confidence_sum.at[
                        layer_idx
                    ].set(confidence_sum)

            if "expert_top1_count" in pytree:
                top1_count = pytree["expert_top1_count"]
                if top1_count.shape[0] == num_experts:
                    expert_top1_count = expert_top1_count.at[layer_idx].set(top1_count)

            if "router_entropy" in pytree:
                router_entropy = router_entropy.at[layer_idx].set(
                    jnp.asarray(pytree["router_entropy"])
                )

            if "token_count" in pytree:
                token_count = token_count.at[layer_idx].set(
                    jnp.asarray(pytree["token_count"], dtype=jnp.int32)
                )

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

    safe_step_count = max(step_count, 1)
    usage_mean = expert_usage / safe_step_count
    usage_sq_mean = expert_usage_sq / safe_step_count
    usage_var = jnp.maximum(usage_sq_mean - usage_mean**2, 0.0)
    selection_confidence = expert_top1_confidence_sum / jnp.maximum(
        expert_top1_count, 1.0
    )
    router_entropy_mean = router_entropy / safe_step_count

    expert_stats = {
        "usage_mean": usage_mean,
        "usage_var": usage_var,
        "selection_confidence": selection_confidence,
        "router_entropy": router_entropy_mean,
        "token_count": token_count,
    }

    return expert_stats, step_count


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
    dustbin_fraction: jnp.ndarray | None = None,
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
        dustbin_fraction: Optional per-expert dustbin mass fraction from OT

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

    # Experts assigned to the under-used role get a stronger positive recovery
    # nudge when they are below uniform usage.
    underused_mask = expert_assignments == 2
    underused_boost = jnp.maximum(uniform - expert_usage, 0.0) * target_balance
    adjustments = jnp.where(underused_mask, adjustments + underused_boost, adjustments)

    if dustbin_fraction is not None:
        confidence_scale = 1.0 - 0.5 * jnp.clip(dustbin_fraction, 0.0, 1.0)
        adjustments = adjustments * confidence_scale

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
    expert_stats, step_count = collect_moe_stats_from_variables(
        mutable_variables, num_layers, num_experts
    )

    if step_count == 0:
        print("No stats accumulated yet, skipping slow loop.")
        return model_params, mutable_variables, {"skipped": True}

    expert_usage = expert_stats["usage_mean"]
    expert_usage_var = expert_stats["usage_var"]
    expert_selection_confidence = expert_stats["selection_confidence"]

    print("\n--- Sensing ---")
    print(f"Steps accumulated: {step_count}")
    print(f"Expert usage shape: {expert_usage.shape}")

    # Calculate per-layer expert usage statistics
    for layer_idx in range(min(num_layers, 3)):  # Show first 3 layers
        layer_usage = expert_usage[layer_idx]
        print(f"Layer {layer_idx} expert usage: {layer_usage}")

    # 2. ASSIGN: Group experts by behavior
    print("\n--- Assignment ---")

    # Use usage statistics across all layers
    usage_mean = expert_usage.mean(axis=0)  # (num_experts,)
    usage_var = expert_usage_var.mean(axis=0)  # (num_experts,)
    selection_confidence = expert_selection_confidence.mean(axis=0)

    # Feature order is fixed by OT role prototypes: usage, variance, confidence.
    assignment_features = jnp.stack(
        [usage_mean, usage_var, selection_confidence], axis=-1
    )

    assignment_method = getattr(config, "slow_loop_assignment_method", "ot")
    ot_result = None
    dustbin_fraction = None
    if assignment_method == "ot":
        ot_config = ot_config_from_model_config(config)
        ot_result = assign_expert_roles_ot(assignment_features, usage_mean, ot_config)
        assignments = ot_result.role_ids
        dustbin_fraction = ot_result.dustbin_fraction
        print("Assignment method: unbalanced OT")
        print(f"Role masses: {ot_result.diagnostics['role_masses']}")
        print(f"Dustbin mass: {ot_result.dustbin_mass:.4f}")
    elif assignment_method == "gmm":
        key, cluster_key = jax.random.split(key)
        assignments, _gmm_params = cluster_experts(
            assignment_features, num_clusters=3, random_key=cluster_key
        )
        print("Assignment method: GMM")
    else:
        raise ValueError(
            f"Unsupported slow_loop_assignment_method={assignment_method!r}."
        )

    # Count experts per cluster
    for c in range(3):
        count = (assignments == c).sum()
        cluster_role = {0: "Generalist", 1: "Specialist", 2: "Under-utilized"}
        print(f"Cluster {c} ({cluster_role.get(c, 'Unknown')}): {count} experts")

    # 3. SMOOTH: Apply 3D smoothing (optional ablation)
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

    if getattr(config, "slow_loop_smoothing", False):
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
    else:
        print("\n--- Smoothing skipped (disabled) ---")

    previous_assignments = mutable_variables.get("stats_buffer", {}).get(
        "last_assignments"
    )
    if (
        previous_assignments is not None
        and previous_assignments.shape == assignments.shape
    ):
        assignment_churn = jnp.mean(previous_assignments != assignments)
    else:
        assignment_churn = jnp.array(0.0)

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
    adjustment_scale = getattr(config, "ot_router_adjustment_scale", 0.1)
    adjustments = compute_router_adjustments(
        assignments,
        usage_mean,
        target_balance=adjustment_scale,
        dustbin_fraction=dustbin_fraction,
    )
    print(f"Recommended router adjustments: {adjustments}")

    # Apply adjustments to model parameters
    # We need to find the router bias parameters in the pytree
    # They are typically named 'router_linear' -> 'bias'

    updates_count = 0

    def update_router_bias(path_items, param):
        nonlocal updates_count
        path = [
            str(item.key) if hasattr(item, "key") else str(item) for item in path_items
        ]

        if "router_linear" in path and "bias" in path:
            if param.shape == adjustments.shape:
                updates_count += 1
                return param + adjustments
        return param

    updated_params = jax.tree_util.tree_map_with_path(update_router_bias, model_params)

    if updates_count > 0:
        print(f"Applied updates to {updates_count} router biases")
    else:
        print("No matching router biases found to update")

    # 6. RESET: Clear stats buffer
    def reset_leaf(x):
        return jnp.zeros_like(x)

    mutable_vars_unfrozen = unfreeze(mutable_variables)
    if "stats_buffer" in mutable_vars_unfrozen:
        mutable_vars_unfrozen["stats_buffer"] = jax.tree.map(
            reset_leaf, mutable_vars_unfrozen["stats_buffer"]
        )
        mutable_vars_unfrozen["stats_buffer"]["last_assignments"] = assignments

    print("\nStats buffer reset.")
    print(f"{'=' * 50}\n")

    # Collect diagnostics
    diagnostics = {
        "assignment_method": assignment_method,
        "expert_usage": usage_mean,
        "expert_usage_var": usage_var,
        "selection_confidence": selection_confidence,
        "assignments": assignments,
        "load_imbalance": imbalance,
        "routing_entropy": relative_entropy,
        "router_entropy": expert_stats["router_entropy"].mean(),
        "assignment_churn": assignment_churn,
        "adjustments": adjustments,
        "router_adjustment_norm": jnp.linalg.norm(adjustments),
        "step_count": step_count,
    }
    if ot_result is not None:
        diagnostics.update(
            {
                "role_probs": ot_result.role_probs,
                "role_masses": ot_result.diagnostics["role_masses"],
                "dustbin_mass": ot_result.dustbin_mass,
                "mean_dustbin_fraction": ot_result.diagnostics["mean_dustbin_fraction"],
                "transport_entropy": ot_result.diagnostics["transport_entropy"],
                "mean_role_confidence": ot_result.diagnostics["mean_role_confidence"],
            }
        )
    else:
        diagnostics.update(
            {
                "role_probs": jax.nn.one_hot(assignments, 3),
                "role_masses": jnp.bincount(assignments, length=3) / num_experts,
                "dustbin_mass": jnp.array(0.0),
                "mean_dustbin_fraction": jnp.array(0.0),
                "transport_entropy": jnp.array(0.0),
                "mean_role_confidence": jnp.array(1.0),
            }
        )

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
