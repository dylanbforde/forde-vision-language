# FORDE (Functional Organisation via Route-Density Estimation) Model

## Primary Objective

The primary objective is to create a Stateful Adaptive Network (SAN), a single large scale neural network that develops emergent functional specialisation during training. This architecture, named FORDE (Functional Organisation via Route-Density Estimation) Model moves beyond static, pre-defined components like (standard MOE) by introducing a dynamic, history-based feedback loop.

This loop senses the computational role of each neuron, clusters them into “types” and then dynamically rewires their function (activation, gradient, and connectivity). The goal is to create a model thta self-organises into spatially-contiguous “areas” optimised for different sub tasks, mimicking a form of neural plasticity.

## Core Approach

to solve the critical issues of instability and computational cost, the model’s logic is decoupled into two timescales:

1. The fast loop runs every training step, this is the main training engine
    1. It performs the standard forward and backward pass to update the network weights (params)
    2. It reads a cached, static (’brain map’) (the neuron_assignments) from the variables state. For this step, the network acts like a fixed, multiplexed architecture.
    3. Its second job is that it silently logs the gradient and activation statistics (Gini, GDP, etc) for every neuron and aggregates them.
2. The slow loop runs every n steps, this is the brain update
    1. It runs the expensive non-jit able logic to update the functional map of the network
    2. It ingests the aggregated statistics from the last N ‘fast’ steps and runs the 4 stage pipeline of 1. Sense, 2. Cluster, 3. Smooth, 4. Actuate
    3. It saves a new ‘brain map’ (neuron_assignments) back into the variables state, which the ‘Fast Loop’ will then use the next N steps.

## Methodology

Using Jax/Flax, the Conceptual Captions Dataset, and the Huggingface Datasets with streaming=True to avoid downloading.

### Phase 1

This phase is the get good curriculum, the first 10k training steps (maybe), the slow brain feedback loop is off.

The neurons are forced into a single, default “Generalist” state (Path 0). The model trains as a standard vanilla transformer.

The goal is to allow the params to converge to a state where the gradients are stable, non-random, and carry meaningful information about the task.

### Phase 2

This phase begins at 10k training steps + 1, and continues for the rest of the training. It consists of the ‘fast’ loop running N times and the ‘slow’ loop running once.

#### The Fast Loop

- Reads the cached neuron_assignments and performs a standard forward/backward pass
- Sensing: On each step this function calculates and logs a rich feature vector for every neuron. This is the Strengthened Sensing Mechanism. Instead of just Gini,GDP, we compute a D-dimensional vector
    - Grad Gini (sparsity): Hoyer’s Sparsity of the neuron’s gradient column, which measures the role concentration
    - Grad GDP (magnitude): L1 Norm of the gradient column. Measures influence/importance.
    - Activation Gini: Hoyer’s Sparsity of the neurons activation over the batch. Measures Firing pattern (spare vs broad)
    - Activation GDP: L1 Norm of the activation, measures overall firing rate
    - Activation Variance: Variance of the activation, measures stability vs dynamism
- Logging these [num_neurons, d_features] vectors are aggregated over the N steps

#### The Slow loop

- Cluster (GMM)
    - This is the core of ‘emergence’, instead of hard coding if statements, we let the model discover its own functional types
    - We fit a Gaussian Mixture Model with k components to the [num_neurons, d_features] data.
    - M-step (learn): The GMM fitting process finds the k cluster centers (means) and shapes (covariances). These gmm params are themselves saved as a learned ‘meta-state’ in the variables.
    - E-step (assign) The model uses the new GMM to calculate the most likely cluster (assign 0,1,2,etc) for each neuron, resulting in new_assignments vector
- Smooth Neighborhood
    - This encourages the formation of biologically plausible ‘continguous-areas’.
    - The new assignments vector (ex shape 1024) is reshaped into its spatial 2d grid (32x32).
    - A 2d convolution is applied with a simple 3x3 gaussian blur kernel. This acts as a ‘diffusion’ or ‘consensus’ step, forcing neighboring neurons to influence each others state.
    - The result is a smoothed_assignments grid.
- Actuate (Update)
    - The final smoothed_assignments grid and the new gmm_params are saved back into the JAX variables pytree.
    - The fast loop will now read and use this new functional map for the next N steps

## Breakdown

### 1. The FORDE-Transformer block
    1. The ‘macro-architecture’ is a stack of these blocks. We are not replacing the Transformer, instead we are making its ‘brain’ adaptive.
        1. If standard is self-attention → add/norm → MLP → add/norm then instead it would be
        2. self-attention → add/norm → stateful_layer → add/norm
        
        The self attention layer handles communication, the stateful layer handles the adaptive processing of that communication 
        
    2. The stateful layer (actuator)
        1. custom nn.Module that replaces the static MLP
            1. input is a token vector z (from attention layer)
            2. it reads its cached integer assignment from the variables state
            3. it uses jnp.where to multiplex the token z to one of k processing paths
                1. 0 (generalist): F_z = relu(z)
                2. 1 (pooling/generalist): f_z = tanh(z)
                3. 2 (Specialist): f_z = binary_step(z)
            4. the custom gradient (binary step) is just a straight through estimator, the forward pass is 0/1 and the backward pass pretends the function was y=x allowing gradients to flow
            5. to create highways for information, (skip connection), and make tihs dynamically controllable
                1. the final output is a gated residual connection where the gate is also controlled by the neurons assignment, where specialist neurons are forced to process and the generalist neurons can be skipped, an output like f_z + (gate * z) assuming z is the projected residual.
### 2. Baseline Model (”Forde-lite”)
    1. An ablation model that removes the slow loop and the GMM
    2. it uses hard coded, rule-based assignments based on instantaneous, not historical stats from the sensing step (e.g is_spec = grad_gini > 0.8)
    3. if the full forde-gmm cannot outperform the simpler forde-lite model the extra complexity of the GMM and the slow brain is probably not justified
### 3. Critical Logging at the end of every slow loop
    1. The Brain Scan (’heat map’) (32x32) plot of the smoothed_assignments grid. This is the primary proof of ‘area’ formation.
    2. The feature space (scatter plot), a 2d plot of the gini and gdp features, coloured by their new cluster assignment. This is the primary debug tool to see if the clusters are useful
    3. The census (histogram), a bar chart of the neuron distribution, health check to see if model collapsed to outputting the same type for everything

Use the metrics from conceptual captions

## Project Structure
```
/home/dylan/code/python/functional-organisation/
├── .gitignore
├── .python-version
├── main.py
├── pyproject.toml
├── README.md
├── src/
│   ├── forde/
│   │   ├── __init__.py
│   │   ├── model.py        # The main FORDE-Transformer block and stateful layer
│   │   ├── sensing.py      # Logic for the "Sensing" step (calculating stats)
│   │   ├── clustering.py   # GMM logic for the "Cluster" step
│   │   ├── smoothing.py    # Convolutional smoothing logic
│   │   └── actuation.py    # Logic for updating the model state
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py      # Huggingface dataset streaming and preprocessing
│   ├── training/
│   │   ├── __init__.py
│   │   └── train.py        # The main training loop (fast and slow loops)
│   └── utils/
│       ├── __init__.py
│       └── logging.py      # Plotting for "Brain Scan", feature space, etc.
└── scripts/
    └── run_training.sh     # Example script to configure and run training
```