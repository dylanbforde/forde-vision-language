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

## Macro-Architecture (CLIP-Style Dual Encoder)

The overall model is a **dual-encoder architecture** that learns a shared embedding space for images and text, inspired by CLIP. It does not generate text (i.e., it is not an image captioner). The goal is to produce image and text vectors that are mathematically close in the shared space if they are semantically related.

1.  **Two Towers**: The model consists of two separate Transformer-based encoders:
    *   An **Image Encoder** (`VisionTransformer`) that processes images.
    *   A **Text Encoder** (`TextTransformer`) that processes text captions.
2.  **Shared Spine**: Both encoders are built from a stack of `FORDETransformerBlock`s. This shared, stateful architecture is the core of the experiment.
3.  **Projection Heads**: The final `[CLS]` token output from each encoder is projected into a shared, lower-dimensional embedding space by a simple dense layer.
4.  **Contrastive Loss**: The model is trained using a contrastive loss function. In each batch, it calculates the cosine similarity between all image and text embeddings. The loss function then works to maximize the similarity of the correct `(image, text)` pairs while minimizing the similarity of all other pairs.

## Breakdown

### 1. The FORDE-Transformer block
This is the fundamental building block for both the image and text encoders. We are not replacing the Transformer, but making its internal MLP adaptive.

*   **Standard Block**: `self-attention` → `add/norm` → `MLP` → `add/norm`
*   **FORDE Block**: `self-attention` → `add/norm` → `StatefulLayer` → `add/norm`

The self-attention layer handles communication, while the `StatefulLayer` handles the adaptive processing of that communication.

### 2. The StatefulLayer (Actuator)
This is a custom `nn.Module` that replaces the static MLP. Its key features are:

1.  **Stateful Assignments**: It reads a cached integer `assignment` for each neuron from the model's state variables.
2.  **Multiplexed Paths**: It uses `jnp.where` to route its input `z` to one of `k` processing paths based on the neuron's assignment:
    *   `0` (Generalist): `F(z) = relu(z)`
    *   `1` (Pooling/Generalist): `F(z) = tanh(z)`
    *   `2` (Specialist): `F(z) = binary_step(z)`
3.  **Custom Gradient**: The `binary_step` uses a straight-through estimator, allowing gradients to flow while the forward pass remains discontinuous.
4.  **Gated Residual Connection**: The final output is `F(z) + (gate * z_projected)`. The `gate` is controlled by the neuron's assignment, dampening the residual connection for specialist neurons (`gate=0.1`) and forcing the network to rely on their processed output. Generalist neurons use a standard residual connection (`gate=1.0`).

### 3. Baseline Model (”Forde-lite”)
An ablation model that removes the slow loop and the GMM. It uses hard-coded, rule-based assignments based on instantaneous, not historical, stats from the sensing step (e.g., `is_spec = grad_gini > 0.8`). If the full FORDE-GMM model cannot outperform this simpler baseline, the extra complexity is not justified.

### 4. Critical Logging
At the end of every slow loop, we will generate:

1.  **The Brain Scan**: A heatmap of the `smoothed_assignments` grid to prove the formation of specialized areas.
2.  **The Feature Space**: A scatter plot of the Gini/GDP features, colored by cluster assignment, to debug cluster quality.
3.  **The Census**: A histogram of the neuron distribution to monitor model health.

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
