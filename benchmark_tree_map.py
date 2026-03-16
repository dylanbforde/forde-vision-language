import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import traverse_util
from flax.core import freeze, unfreeze
import time

# Create a dummy parameter dict
def create_dummy_params(num_layers=12):
    params = {}
    for i in range(num_layers):
        layer_params = {
            'moe': {
                'router_linear': {
                    'kernel': jnp.zeros((512, 8)),
                    'bias': jnp.zeros((8,))
                },
                'expert_0': {'kernel': jnp.zeros((512, 2048)), 'bias': jnp.zeros((2048,))},
                'expert_1': {'kernel': jnp.zeros((512, 2048)), 'bias': jnp.zeros((2048,))},
            },
            'attention': {
                'q_proj': {'kernel': jnp.zeros((512, 512))},
                'k_proj': {'kernel': jnp.zeros((512, 512))},
                'v_proj': {'kernel': jnp.zeros((512, 512))},
                'o_proj': {'kernel': jnp.zeros((512, 512))},
            }
        }
        params[f'layer_{i}'] = layer_params
    return freeze({'params': params})

params = create_dummy_params(24) # 24 layers for more depth
adjustments = jnp.ones((8,))

# Method 1: traverse_util
def update_with_traverse_util(model_params, adjustments):
    flat_params = traverse_util.flatten_dict(unfreeze(model_params))
    updated_flat_params = {}

    updates_count = 0
    for path, param in flat_params.items():
        if "router_linear" in path and "bias" in path:
            if param.shape == adjustments.shape:
                updated_flat_params[path] = param + adjustments
                updates_count += 1
            else:
                updated_flat_params[path] = param
        else:
            updated_flat_params[path] = param

    return traverse_util.unflatten_dict(updated_flat_params), updates_count

# Method 2: tree_map_with_path
def update_with_tree_map(model_params, adjustments):
    updates_count = 0
    def _map_fn(path, param):
        nonlocal updates_count
        # path is a tuple of DictKey, SequenceKey, etc.
        path_strs = [p.key if hasattr(p, 'key') else str(p) for p in path]
        if "router_linear" in path_strs and "bias" in path_strs:
            if param.shape == adjustments.shape:
                updates_count += 1
                return param + adjustments
        return param

    updated_params = jax.tree_util.tree_map_with_path(_map_fn, model_params)
    return updated_params, updates_count

# Warmup
res1, _ = update_with_traverse_util(params, adjustments)
res2, _ = update_with_tree_map(params, adjustments)

import timeit
n_runs = 100
t1 = timeit.timeit(lambda: update_with_traverse_util(params, adjustments), number=n_runs)
t2 = timeit.timeit(lambda: update_with_tree_map(params, adjustments), number=n_runs)

print(f"traverse_util: {t1:.4f}s")
print(f"tree_map_with_path: {t2:.4f}s")
print(f"Speedup: {t1/t2:.2f}x")
