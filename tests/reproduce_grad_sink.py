
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

class TestModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        # We want to capture the gradient w.r.t this intermediate value 'y'
        y = nn.Dense(1)(x)
        
        # Gradient Sink Pattern
        # We add a 'sink' variable which is initialized to 0.
        # We will differentiate w.r.t this sink variable.
        sink = self.variable('grad_sinks', 'sink', lambda: jnp.zeros_like(y))
        y_sink = y + sink.value
        
        z = nn.Dense(1)(y_sink)
        return z

def main():
    key = jax.random.PRNGKey(0)
    x = jnp.ones((1, 1))
    
    model = TestModel()
    variables = model.init(key, x)
    params = variables['params']
    grad_sinks = variables['grad_sinks'] # Should be zeros
    
    print("Initial grad_sinks:", grad_sinks)
    
    def loss_fn(params, grad_sinks, x):
        # We pass grad_sinks as an input to be differentiated
        variables = {'params': params, 'grad_sinks': grad_sinks}
        y_out = model.apply(variables, x)
        return jnp.mean(y_out**2)
    
    # We want gradients w.r.t params AND grad_sinks
    grad_fn = jax.grad(loss_fn, argnums=(0, 1))
    
    param_grads, sink_grads = grad_fn(params, grad_sinks, x)
    
    print("\nParam grads:", param_grads)
    print("Sink grads (Gradient w.r.t intermediate y):", sink_grads)
    
    # Verification:
    # y = w1 * x + b1
    # z = w2 * y + b2
    # L = z^2
    # dL/dy = dL/dz * dz/dy = 2z * w2
    
    # Let's manually calculate expected gradient
    w1 = params['Dense_0']['kernel']
    b1 = params['Dense_0']['bias']
    w2 = params['Dense_1']['kernel']
    b2 = params['Dense_1']['bias']
    
    y = jnp.dot(x, w1) + b1
    z = jnp.dot(y, w2) + b2
    dL_dz = 2 * z
    dL_dy = jnp.dot(dL_dz, w2.T)
    
    print("\nExpected dL/dy:", dL_dy)
    print("Calculated Sink Grad:", sink_grads['sink'])
    
    assert jnp.allclose(sink_grads['sink'], dL_dy)
    print("\nSUCCESS: Gradient Sink pattern works!")

if __name__ == "__main__":
    main()
