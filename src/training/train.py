"""
LLM Pretraining Script for FORDE Decoder Model.

This training script supports:
- Next-token prediction language modeling loss
- MoE auxiliary load balancing loss
- Configurable model architecture (MoE, NSA, mHC)
- Streaming dataset loading
"""

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state, checkpoints
from tqdm.auto import tqdm
import argparse
from dataclasses import asdict
from typing import Dict, Any
from datetime import datetime

# Handle imports
try:
    from src.forde.model import FORDEDecoderLM, LLMConfig
    from src.data.dataset import create_lm_dataset, create_dummy_dataset
except ModuleNotFoundError:
    import sys
    import os

    # Add project root to path
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    sys.path.insert(0, project_root)

    from src.forde.model import FORDEDecoderLM, LLMConfig
    from src.data.dataset import create_lm_dataset, create_dummy_dataset


try:
    from torch.utils.tensorboard import SummaryWriter

    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("Warning: tensorboard not found. Logging will be disabled.")


class TrainState(train_state.TrainState):
    """Extended train state to track auxiliary losses and mutable stats."""

    stats_buffer: Dict[str, Any]


def create_train_state(
    config: LLMConfig,
    key: jax.random.PRNGKey,
    learning_rate: float,
    weight_decay: float = 0.01,
):
    """Create training state with model and optimizer."""
    model = FORDEDecoderLM(config=config)

    # Initialize with dummy input
    dummy_seq_len = min(64, config.max_seq_len)
    dummy_input = jnp.ones((1, dummy_seq_len), dtype=jnp.int32)
    variables = model.init(key, dummy_input)
    params = variables["params"]

    # Extract mutable state (stats_buffer)
    mutable_vars = {k: v for k, v in variables.items() if k != "params"}
    if "stats_buffer" not in mutable_vars:
        mutable_vars["stats_buffer"] = {}

    # Create optimizer with weight decay
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping
        optax.adamw(learning_rate, weight_decay=weight_decay),
    )

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        stats_buffer=mutable_vars["stats_buffer"],
    )


def compute_loss(
    params,
    stats_buffer,
    apply_fn,
    input_ids: jnp.ndarray,
    vocab_size: int,
    aux_loss_weight: float = 1.0,
):
    """
    Compute language modeling loss with MoE auxiliary loss.
    """
    # Pass mutable stats_buffer to capture updates
    # We need to pass 'stats_buffer' in mutable list to get it back
    (logits, aux_loss), new_mutable_vars = apply_fn(
        {"params": params, "stats_buffer": stats_buffer},
        input_ids,
        mutable=["stats_buffer"],
    )

    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    # Cross-entropy loss
    lm_loss = optax.softmax_cross_entropy_with_integer_labels(
        shift_logits.reshape(-1, vocab_size), shift_labels.reshape(-1)
    ).mean()

    total_loss = lm_loss + aux_loss_weight * aux_loss

    metrics = {"lm_loss": lm_loss, "aux_loss": aux_loss, "total_loss": total_loss}

    return total_loss, (metrics, new_mutable_vars)


@jax.jit
def train_step(state, batch, vocab_size, aux_loss_weight):
    """JIT-compiled training step.

    Note: vocab_size and aux_loss_weight must be concrete/static values.
    The function is recompiled for each unique combination of these values.
    """

    def loss_fn(params):
        # We need to capture the updated stats_buffer here
        # But jax.value_and_grad only returns the first output of the function
        # So we need to bundle everything into the return value

        (logits, aux_loss), new_mutable_vars = state.apply_fn(
            {"params": params, "stats_buffer": state.stats_buffer},
            batch["input_ids"],
            mutable=["stats_buffer"],
        )

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_labels = batch["input_ids"][:, 1:]

        # Get vocab size from logits shape (static at trace time)
        v_size = shift_logits.shape[-1]

        # Cross-entropy loss
        lm_loss = optax.softmax_cross_entropy_with_integer_labels(
            shift_logits.reshape(-1, v_size), shift_labels.reshape(-1)
        ).mean()

        total_loss = lm_loss + aux_loss_weight * aux_loss

        metrics = {"lm_loss": lm_loss, "aux_loss": aux_loss, "total_loss": total_loss}

        return total_loss, (metrics, new_mutable_vars)

    (loss, (metrics, new_mutable_vars)), grads = jax.value_and_grad(
        loss_fn, has_aux=True
    )(state.params)
    state = state.apply_gradients(grads=grads)

    # Update stats_buffer in state
    state = state.replace(stats_buffer=new_mutable_vars["stats_buffer"])

    # Compute gradient norm for monitoring
    grad_norm = optax.global_norm(grads)
    metrics["grad_norm"] = grad_norm

    return state, metrics


@jax.jit
def eval_step(state, batch, aux_loss_weight):
    """Evaluate one batch without applying gradients or retaining stats updates."""

    (logits, aux_loss), _new_mutable_vars = state.apply_fn(
        {"params": state.params, "stats_buffer": state.stats_buffer},
        batch["input_ids"],
        mutable=["stats_buffer"],
    )
    shift_logits = logits[:, :-1, :]
    shift_labels = batch["input_ids"][:, 1:]
    v_size = shift_logits.shape[-1]
    lm_loss = optax.softmax_cross_entropy_with_integer_labels(
        shift_logits.reshape(-1, v_size), shift_labels.reshape(-1)
    ).mean()
    total_loss = lm_loss + aux_loss_weight * aux_loss
    perplexity = jnp.exp(jnp.minimum(lm_loss, 20.0))
    return {
        "lm_loss": lm_loss,
        "aux_loss": aux_loss,
        "total_loss": total_loss,
        "perplexity": perplexity,
    }


def parse_role_priors(raw: str) -> tuple[float, ...]:
    """Parse comma-separated role priors from the CLI."""
    try:
        return tuple(float(value.strip()) for value in raw.split(",") if value.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid --ot_role_priors value {raw!r}."
        ) from exc


def log_slow_loop_diagnostics(writer, diagnostics, step: int):
    """Write scalar slow-loop diagnostics to TensorBoard."""
    if writer is None:
        return

    scalar_keys = [
        "load_imbalance",
        "routing_entropy",
        "router_entropy",
        "assignment_churn",
        "router_adjustment_norm",
        "dustbin_mass",
        "mean_dustbin_fraction",
        "transport_entropy",
        "mean_role_confidence",
    ]
    for key in scalar_keys:
        if key in diagnostics:
            writer.add_scalar(f"SlowLoop/{key}", float(jax.device_get(diagnostics[key])), step)

    if "role_masses" in diagnostics:
        for role_idx, mass in enumerate(jax.device_get(diagnostics["role_masses"])):
            writer.add_scalar(f"SlowLoop/role_{role_idx}_mass", float(mass), step)


def main():
    parser = argparse.ArgumentParser(description="Train FORDE LLM")

    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--max_steps", type=int, default=1000, help="Max steps per epoch"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument(
        "--aux_loss_weight", type=float, default=0.01, help="MoE aux loss weight"
    )
    parser.add_argument(
        "--log_interval", type=int, default=10, help="Steps between logging"
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=0,
        help="Steps between validation batches (0 to disable)",
    )
    parser.add_argument(
        "--eval_batches",
        type=int,
        default=1,
        help="Validation batches per evaluation interval",
    )

    # Data arguments
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--eval_dataset_name", type=str, default=None)
    parser.add_argument("--eval_dataset_config", type=str, default=None)
    parser.add_argument("--eval_split", type=str, default="validation")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2")
    parser.add_argument(
        "--no_streaming", action="store_true", help="Disable Hugging Face streaming"
    )

    # Model arguments
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
    parser.add_argument(
        "--num_heads", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument(
        "--num_experts", type=int, default=4, help="Number of MoE experts"
    )
    parser.add_argument("--window_size", type=int, default=128, help="NSA window size")
    parser.add_argument(
        "--max_seq_len", type=int, default=512, help="Maximum sequence length"
    )

    # Feature flags
    parser.add_argument("--no_moe", action="store_true", help="Disable MoE")
    parser.add_argument(
        "--no_nsa", action="store_true", help="Disable sparse attention"
    )
    parser.add_argument(
        "--no_mhc", action="store_true", help="Disable hyper-connections"
    )
    parser.add_argument(
        "--use_dummy_data", action="store_true", help="Use dummy data for testing"
    )
    parser.add_argument(
        "--slow_loop_interval",
        type=int,
        default=100,
        help="Steps between slow loop runs (0 to disable)",
    )
    parser.add_argument(
        "--slow_loop_assignment_method",
        choices=("ot", "gmm"),
        default="ot",
        help="Expert role assignment method for the slow loop",
    )
    parser.add_argument(
        "--slow_loop_smoothing",
        action="store_true",
        help="Apply spatial smoothing after hard role assignment",
    )
    parser.add_argument("--ot_epsilon", type=float, default=1.0)
    parser.add_argument("--ot_tau_q", type=float, default=1.0)
    parser.add_argument("--ot_tau_k", type=float, default=1.0)
    parser.add_argument("--ot_n_iters", type=int, default=30)
    parser.add_argument("--ot_refine_steps", type=int, default=2)
    parser.add_argument("--ot_temperature", type=float, default=1.0)
    parser.add_argument(
        "--ot_role_priors",
        type=parse_role_priors,
        default=(0.35, 0.45, 0.15, 0.05),
        help="Comma-separated role priors, including dustbin when enabled",
    )
    parser.add_argument(
        "--no_ot_dustbin",
        action="store_true",
        help="Disable the OT dustbin/overflow role",
    )
    parser.add_argument(
        "--ot_router_adjustment_scale",
        type=float,
        default=0.1,
        help="Router bias correction scale used after OT assignment",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="forde_llm",
        help="Name for the TensorBoard experiment",
    )
    args = parser.parse_args()

    # Build config
    config = LLMConfig(
        vocab_size=50257,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        head_dim=args.d_model // args.num_heads,
        max_seq_len=args.max_seq_len,
        use_moe=not args.no_moe,
        num_experts=args.num_experts,
        top_k_experts=2,
        expert_hidden_dim=args.d_model * 4,
        slow_loop_assignment_method=args.slow_loop_assignment_method,
        slow_loop_smoothing=args.slow_loop_smoothing,
        ot_num_roles=3,
        ot_use_dustbin=not args.no_ot_dustbin,
        ot_epsilon=args.ot_epsilon,
        ot_tau_q=args.ot_tau_q,
        ot_tau_k=args.ot_tau_k,
        ot_n_iters=args.ot_n_iters,
        ot_refine_steps=args.ot_refine_steps,
        ot_role_priors=args.ot_role_priors,
        ot_temperature=args.ot_temperature,
        ot_router_adjustment_scale=args.ot_router_adjustment_scale,
        use_sparse_attention=not args.no_nsa,
        window_size=args.window_size,
        compression_ratio=4,
        top_k_global=32,
        use_hyper_connections=not args.no_mhc,
        num_streams=2,
        sinkhorn_iterations=3,
        dropout_rate=0.0,  # No dropout for now
    )

    print("=" * 60)
    print("FORDE LLM Training")
    print("=" * 60)
    print("\nModel Configuration:")
    for key, value in asdict(config).items():
        print(f"  {key}: {value}")

    # Initialize
    key = jax.random.PRNGKey(42)
    key, init_key, data_key = jax.random.split(key, 3)

    print("\nInitializing model...")
    state = create_train_state(config, init_key, args.learning_rate, args.weight_decay)

    param_count = sum(x.size for x in jax.tree.leaves(state.params))
    print(f"Total parameters: {param_count:,}")

    # TensorBoard logging
    writer = None
    if HAS_TENSORBOARD:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = f"runs/{args.experiment_name}_{timestamp}"
        print(f"Logging to: {log_dir}")
        writer = SummaryWriter(log_dir=log_dir)

    # Create dataset
    print("\nCreating dataset...")
    if args.use_dummy_data:
        dataset = create_dummy_dataset(
            vocab_size=config.vocab_size,
            seq_len=args.max_seq_len,
            num_samples=args.batch_size * args.max_steps,
        )
    else:
        dataset = create_lm_dataset(
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            split=args.train_split,
            vocab_size=config.vocab_size,
            max_seq_len=args.max_seq_len,
            streaming=not args.no_streaming,
            tokenizer_name=args.tokenizer_name,
        )

    eval_dataset = None
    eval_iterator = None
    if args.eval_interval > 0:
        print("\nCreating evaluation dataset...")
        if args.use_dummy_data:
            eval_dataset = create_dummy_dataset(
                vocab_size=config.vocab_size,
                seq_len=args.max_seq_len,
                num_samples=args.batch_size * args.eval_batches,
            )
        else:
            eval_dataset = create_lm_dataset(
                dataset_name=args.eval_dataset_name or args.dataset_name,
                dataset_config=args.eval_dataset_config or args.dataset_config,
                split=args.eval_split,
                vocab_size=config.vocab_size,
                max_seq_len=args.max_seq_len,
                streaming=not args.no_streaming,
                tokenizer_name=args.tokenizer_name,
            )
        eval_iterator = iter(eval_dataset.batch(args.batch_size))

    # Training loop
    print("\nStarting training...")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Slow loop interval: {args.slow_loop_interval} steps")
    print(f"  Assignment method: {args.slow_loop_assignment_method}")

    step = 0
    for epoch in range(args.num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.num_epochs} ---")

        # Create batches
        epoch_iterator = dataset.batch(args.batch_size)

        pbar = tqdm(epoch_iterator, total=args.max_steps, desc=f"Epoch {epoch + 1}")

        for batch in pbar:
            if step >= args.max_steps:
                break

            # Ensure batch is properly formatted
            if isinstance(batch, dict):
                input_ids = jnp.array(batch["input_ids"])
            else:
                input_ids = jnp.array(batch)

            # Truncate if needed
            if input_ids.shape[1] > args.max_seq_len:
                input_ids = input_ids[:, : args.max_seq_len]

            batch_dict = {"input_ids": input_ids}

            # Training step
            state, metrics = train_step(
                state, batch_dict, config.vocab_size, args.aux_loss_weight
            )

            # Logging
            if step % args.log_interval == 0:
                pbar.set_postfix(
                    {
                        "loss": f"{metrics['total_loss']:.4f}",
                        "lm": f"{metrics['lm_loss']:.4f}",
                        "aux": f"{metrics['aux_loss']:.4f}",
                        "gnorm": f"{metrics['grad_norm']:.2f}",
                    }
                )

                if writer:
                    writer.add_scalar("Loss/total", metrics["total_loss"].item(), step)
                    writer.add_scalar("Loss/lm", metrics["lm_loss"].item(), step)
                    writer.add_scalar("Loss/aux", metrics["aux_loss"].item(), step)
                    writer.add_scalar(
                        "Training/grad_norm", metrics["grad_norm"].item(), step
                    )

            if eval_dataset is not None and step > 0 and step % args.eval_interval == 0:
                eval_totals = {
                    "total_loss": 0.0,
                    "lm_loss": 0.0,
                    "aux_loss": 0.0,
                    "perplexity": 0.0,
                }
                for _ in range(args.eval_batches):
                    try:
                        eval_batch = next(eval_iterator)
                    except StopIteration:
                        eval_iterator = iter(eval_dataset.batch(args.batch_size))
                        eval_batch = next(eval_iterator)

                    if isinstance(eval_batch, dict):
                        eval_input_ids = jnp.array(eval_batch["input_ids"])
                    else:
                        eval_input_ids = jnp.array(eval_batch)
                    if eval_input_ids.shape[1] > args.max_seq_len:
                        eval_input_ids = eval_input_ids[:, : args.max_seq_len]

                    eval_metrics = eval_step(
                        state,
                        {"input_ids": eval_input_ids},
                        args.aux_loss_weight,
                    )
                    for key_name in eval_totals:
                        eval_totals[key_name] += float(
                            jax.device_get(eval_metrics[key_name])
                        )

                eval_metrics_mean = {
                    key_name: value / args.eval_batches
                    for key_name, value in eval_totals.items()
                }
                print(
                    "  Eval: "
                    f"loss={eval_metrics_mean['total_loss']:.4f}, "
                    f"ppl={eval_metrics_mean['perplexity']:.2f}"
                )
                if writer:
                    for key_name, value in eval_metrics_mean.items():
                        writer.add_scalar(f"Eval/{key_name}", value, step)

            # Run slow loop periodically
            if (
                args.slow_loop_interval > 0
                and step > 0
                and step % args.slow_loop_interval == 0
            ):
                key, slow_key = jax.random.split(key)

                # Import slow loop
                try:
                    from src.forde.moe_slow_loop import moe_slow_loop_step
                except ModuleNotFoundError:
                    import sys

                    sys.path.insert(
                        0, "/home/dylan/code/python/functional-organisation"
                    )
                    from src.forde.moe_slow_loop import moe_slow_loop_step

                print(f"\n[Step {step}] Running slow loop...")

                # Execute slow loop
                # This updates params (router biases) and resets stats_buffer
                new_params, new_mutable_vars, diagnostics = moe_slow_loop_step(
                    model_params=state.params,
                    mutable_variables={"stats_buffer": state.stats_buffer},
                    config=config,
                    key=slow_key,
                    epoch=epoch,
                    step=step,
                )

                # Update state
                state = state.replace(
                    params=new_params, stats_buffer=new_mutable_vars["stats_buffer"]
                )

                if "skipped" not in diagnostics:
                    log_slow_loop_diagnostics(writer, diagnostics, step)
                    print(
                        "  Slow loop complete. "
                        f"method={diagnostics['assignment_method']}, "
                        f"dustbin={float(jax.device_get(diagnostics['dustbin_mass'])):.4f}, "
                        f"churn={float(jax.device_get(diagnostics['assignment_churn'])):.4f}"
                    )

            step += 1

        print(f"Epoch {epoch + 1} complete. Final loss: {metrics['total_loss']:.4f}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    # Save final checkpoint
    if args.checkpoint_dir:
        print(f"Saving checkpoint to {args.checkpoint_dir}...")
        checkpoints.save_checkpoint(
            ckpt_dir=args.checkpoint_dir,
            target=state,
            step=args.max_steps,
            overwrite=True,
            keep=1,
        )
        print("Checkpoint saved.")

    if writer:
        writer.close()


if __name__ == "__main__":
    main()
