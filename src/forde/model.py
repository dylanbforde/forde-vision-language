import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence
from dataclasses import dataclass

from src.forde.sensing import calculate_neuron_stats # Import the sensing function

# As per the README, the binary_step function needs a custom gradient.
# The forward pass is a step function, but the backward pass (gradient) 
# should be a straight-through estimator (i.e., pretends the function was y=x).
@jax.custom_jvp
def binary_step(x):
    return jnp.where(x > 0, 1.0, 0.0)

@binary_step.defjvp
def binary_step_jvp(primals, tangents):
    x, = primals
    x_dot, = tangents
    primal_out = binary_step(x)
    # The gradient is 1, implementing the straight-through estimator.
    tangent_out = x_dot
    return primal_out, tangent_out

class StatefulLayer(nn.Module):
    """A layer that replaces the standard MLP in a Transformer block."""
    features: int

    @nn.compact
    def __call__(self, z):
        """Performs the forward pass of the stateful layer."""
        assignments_var = self.variable(
            'state',
            'assignments',
            lambda: jnp.zeros(self.features, dtype=jnp.int32)
        )
        assignments = assignments_var.value
        
        x = nn.Dense(self.features, name="dense_layer")(z)

        # --- Gradient Capture & Sensing ---
        # 1. Gradient Sink Pattern:
        #    We create a variable 'sink' initialized to zeros.
        #    We add this sink to the activation 'x'.
        #    During the backward pass, we will differentiate w.r.t. this 'sink' variable.
        #    The gradient of the loss w.r.t. 'sink' is exactly the gradient w.r.t. 'x'.
        grad_sink = self.variable(
            'grad_sinks',
            'sink',
            lambda: jnp.zeros_like(x)
        )
        # Add the sink (zeros) to x. This doesn't change the forward pass value,
        # but connects 'grad_sink' to the computation graph.
        x = x + grad_sink.value

        # 2. Read the gradients that were captured in the *previous* training step
        #    from the 'grad_buffer' collection.
        grad_var = self.variable(
            'grad_buffer',
            'pre_activation_grad',
            lambda: jnp.zeros_like(x) # Initialize with zeros for the first step
        )
        prev_step_grads = grad_var.value

        # 3. Use the (potentially delayed) gradients to calculate neuron stats.
        #    Note: We use the original 'x' (before sink addition) for stats calculation
        #    to avoid any potential circular dependency issues, though x+0 is same.
        current_stats = calculate_neuron_stats(x, prev_step_grads)
        
        # --- Stats Aggregation ---
        # Aggregate stats in a mutable variable collection
        D = current_stats.shape[1]

        def init_neuron_stats_buffer():
            return {f'neuron_{i}': jnp.zeros(D, dtype=jnp.float32) for i in range(self.features)}

        stats_buffer_collection = self.variable(
            'stats_buffer',
            'data',
            lambda: {
                'neuron_stats': init_neuron_stats_buffer(),
                'step_count': jnp.array(0, dtype=jnp.int32)
            }
        )
        
        # Update neuron_stats and increment step_count
        updated_neuron_stats = stats_buffer_collection.value['neuron_stats'].copy() 
        for i in range(self.features):
            updated_neuron_stats[f'neuron_{i}'] += current_stats[i]

        stats_buffer_collection.value = {
            'neuron_stats': updated_neuron_stats,
            'step_count': stats_buffer_collection.value['step_count'] + 1
        }

        # --- Functional Pathways ---
        path0_out = nn.relu(x)
        path1_out = nn.tanh(x)
        path2_out = binary_step(x)

        output_p0_p1 = jnp.where(assignments == 0, path0_out, path1_out)
        multiplexed_output = jnp.where(assignments == 2, path2_out, output_p0_p1)

        z_projected = nn.Dense(self.features, name="projection_layer")(z)
        gate = jnp.where(assignments == 2, 0.1, 1.0)
        final_output = multiplexed_output + (gate * z_projected)

        return final_output

class FORDETransformerBlock(nn.Module):
    """A Transformer block that uses a StatefulLayer instead of a standard MLP."""
    features: int
    num_heads: int

    @nn.compact
    def __call__(self, x):
        # Standard pre-normalization Transformer architecture
        # First sub-layer: Multi-head self-attention
        y = nn.LayerNorm()(x)
        y = nn.SelfAttention(num_heads=self.num_heads)(y)
        x = x + y

        # Second sub-layer: StatefulLayer
        y = nn.LayerNorm()(x)
        y = StatefulLayer(features=self.features)(y)
        x = x + y

        return x

class VisionTransformer(nn.Module):
    """A Vision Transformer model using FORDETransformerBlocks."""
    patch_size: int
    num_layers: int
    features: int
    num_heads: int
    image_size: int = 224

    @nn.compact
    def __call__(self, x):
        # 1. Patch and embed the input image
        x = nn.Conv(
            self.features,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding='VALID',
            name='patch_embed'
        )(x)
        
        # Reshape to (batch, num_patches, features)
        batch_size, h, w, c = x.shape
        x = jnp.reshape(x, (batch_size, h * w, c))
        num_patches = h * w

        # 2. Prepend CLS token and add positional embeddings
        cls_token = self.param('cls_token', nn.initializers.zeros, (1, 1, self.features))
        cls_token = jnp.tile(cls_token, (batch_size, 1, 1))
        x = jnp.concatenate([cls_token, x], axis=1)

        pos_embedding = self.param(
            'pos_embedding',
            nn.initializers.normal(stddev=0.02),
            (1, num_patches + 1, self.features)
        )
        x = x + pos_embedding

        # 3. Process through Transformer blocks
        for _ in range(self.num_layers):
            x = FORDETransformerBlock(features=self.features, num_heads=self.num_heads)(x)
        
        # 4. Final normalization
        x = nn.LayerNorm(name='final_norm')(x)

        return x

class TextTransformer(nn.Module):
    """A Text Transformer model using FORDETransformerBlocks."""
    vocab_size: int
    num_layers: int
    features: int
    num_heads: int
    max_len: int = 128

    @nn.compact
    def __call__(self, x):
        # 1. Embed tokens and add positional embeddings
        x = nn.Embed(num_embeddings=self.vocab_size, features=self.features)(x)
        pos_embedding = self.param(
            'pos_embedding',
            nn.initializers.normal(stddev=0.02),
            (1, self.max_len, self.features)
        )
        x = x + pos_embedding

        # 2. Process through Transformer blocks
        for _ in range(self.num_layers):
            x = FORDETransformerBlock(features=self.features, num_heads=self.num_heads)(x)
        
        # 3. Final normalization
        x = nn.LayerNorm(name='final_norm')(x)

        return x

@dataclass
class VisionConfig:
    patch_size: int
    num_layers: int
    features: int
    num_heads: int

@dataclass
class TextConfig:
    vocab_size: int
    num_layers: int
    features: int
    num_heads: int
    max_len: int

class FORDEModel(nn.Module):
    """The final dual-encoder model with projection heads."""
    vision_config: VisionConfig
    text_config: TextConfig
    projection_dim: int

    @nn.compact
    def __call__(self, image, text):
        # 1. Encode image and text
        vision_output = VisionTransformer(**self.vision_config.__dict__)(image)
        text_output = TextTransformer(**self.text_config.__dict__)(text)

        # 2. Extract CLS token output
        vision_cls = vision_output[:, 0]
        text_cls = text_output[:, 0]

        # 3. Project to shared embedding space
        image_embedding = nn.Dense(features=self.projection_dim, name="vision_projection")(vision_cls)
        text_embedding = nn.Dense(features=self.projection_dim, name="text_projection")(text_cls)

        # 4. Add a learnable temperature parameter
        logit_scale = self.param('logit_scale', nn.initializers.zeros, ())

        return image_embedding, text_embedding, logit_scale

if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
    batch_size = 4
    
    # --- Test FORDEModel ---
    print("\n--- Initializing FORDEModel ---")
    dummy_image = jnp.ones((batch_size, 224, 224, 3))
    dummy_text = jnp.ones((batch_size, 128), dtype=jnp.int32)

    vision_config = VisionConfig(patch_size=16, num_layers=2, features=128, num_heads=4)
    text_config = TextConfig(vocab_size=30522, num_layers=2, features=128, num_heads=4)

    model = FORDEModel(vision_config=vision_config, text_config=text_config, projection_dim=64)

    variables = model.init(key, dummy_image, dummy_text)
    image_embed, text_embed, logit_scale = model.apply(variables, dummy_image, dummy_text)

    print(f"Input image shape: {dummy_image.shape}")
    print(f"Input text shape: {dummy_text.shape}")
    print(f"Output image embedding shape: {image_embed.shape}")
    print(f"Output text embedding shape: {text_embed.shape}")
    print(f"Logit scale: {logit_scale}")
    print("FORDEModel executed successfully.")