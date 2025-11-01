# Project TODOs

### Data Pipeline
- [x] **`src/data/dataset.py`**: Implement image fetching and processing from URLs.
- [x] **`src/data/dataset.py`**: Choose and implement a text tokenizer for captions.
- [x] **`src/training/train.py`**: Replace manual batching with a parallel `DataLoader` to accelerate data loading.

### Model Architecture
- [x] **`src/forde/model.py`**: Implement state management in `StatefulLayer`.
- [x] **`src/forde/model.py`**: Implement multiplexing logic in `StatefulLayer`.
- [x] **`src/forde/model.py`**: Implement the gated residual connection in `StatefulLayer`.
- [x] **`src/forde/model.py`**: Define the `FORDETransformerBlock`.
- [x] **`src/forde/model.py`**: Assemble the full `VisionTransformer` (image encoder).
- [x] **`src/forde/model.py`**: Assemble the `TextTransformer` (text encoder).
- [x] **`src/forde/model.py`**: Create the final dual-encoder model with projection heads.

### Training
- [x] **`src/training/train.py`**: Implement the contrastive loss function.

### FORDE Slow Loop (Functional Organisation)
- [x] **`src/forde/sensing.py`**: Implement neuron statistics calculation (Grad Gini, Grad GDP, Act Gini, Act GDP, Act Variance).
- [x] **`src/forde/sensing.py`**: Integrate neuron statistics logging into `StatefulLayer` and `train.py`.
- [x] **`src/forde/sensing.py`**: Implement aggregation of neuron statistics over N fast steps.
- [x] **`src/forde/clustering.py`**: Implement GMM-based clustering of neuron statistics.
- [x] **`src/forde/smoothing.py`**: Implement 2D convolutional smoothing of neuron assignments.
- [ ] **`src/forde/actuation.py`**: Implement updating of `neuron_assignments` in the model's state (placeholder implementation, needs final path).
- [x] **`src/training/train.py`**: Integrate the slow loop to run every N steps (placeholder for stat aggregation and actuation).