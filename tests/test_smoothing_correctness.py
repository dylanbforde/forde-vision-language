import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve, convolve2d

from src.forde.smoothing import smooth_assignments, smooth_assignments_3d


def _smooth_assignments_reference(assignment_grid, kernel_size=3, num_clusters=3):
    kernel = jnp.ones((kernel_size, kernel_size)) / (kernel_size**2)
    one_hot_grid = jax.nn.one_hot(assignment_grid, num_clusters)
    original_h, original_w, _ = one_hot_grid.shape
    pad_h = max(0, kernel_size + 1 - original_h)
    pad_w = max(0, kernel_size + 1 - original_w)
    padding_config = (
        (pad_h // 2, pad_h - pad_h // 2),
        (pad_w // 2, pad_w - pad_w // 2),
        (0, 0),
    )
    padded_one_hot_grid = jnp.pad(one_hot_grid, padding_config, "constant")
    smoothed = jnp.stack(
        [
            convolve2d(padded_one_hot_grid[:, :, i], kernel, mode="same")
            for i in range(num_clusters)
        ],
        axis=-1,
    )
    smoothed = smoothed[
        padding_config[0][0] : padding_config[0][0] + original_h,
        padding_config[1][0] : padding_config[1][0] + original_w,
        :,
    ]
    return jnp.argmax(smoothed, axis=-1)


def _smooth_assignments_3d_reference(assignment_grid, kernel_size=3, num_clusters=3):
    kernel = jnp.ones((kernel_size, kernel_size, kernel_size)) / (kernel_size**3)
    one_hot_grid = jax.nn.one_hot(assignment_grid, num_clusters)
    d, h, w, _ = one_hot_grid.shape
    smoothed_channels = []

    for c in range(num_clusters):
        channel = one_hot_grid[..., c]
        pad_d = max(0, kernel_size - channel.shape[0])
        pad_h = max(0, kernel_size - channel.shape[1])
        pad_w = max(0, kernel_size - channel.shape[2])

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            padding = (
                (pad_d // 2, pad_d - pad_d // 2),
                (pad_h // 2, pad_h - pad_h // 2),
                (pad_w // 2, pad_w - pad_w // 2),
            )
            padded_channel = jnp.pad(channel, padding, "edge")
        else:
            padded_channel = channel
            padding = ((0, 0), (0, 0), (0, 0))

        smoothed = convolve(padded_channel, kernel, mode="same")

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            start_d = padding[0][0]
            start_h = padding[1][0]
            start_w = padding[2][0]
            smoothed = smoothed[
                start_d : start_d + d,
                start_h : start_h + h,
                start_w : start_w + w,
            ]

        smoothed_channels.append(smoothed)

    return jnp.argmax(jnp.stack(smoothed_channels, axis=-1), axis=-1)


def test_smooth_assignments_matches_reference():
    grid = jnp.array(
        [
            [0, 0, 1, 1],
            [0, 2, 1, 1],
            [2, 2, 2, 1],
            [2, 0, 0, 1],
        ]
    )

    expected = _smooth_assignments_reference(grid, kernel_size=3, num_clusters=3)
    actual = smooth_assignments(grid, kernel_size=3, num_clusters=3)

    assert jnp.array_equal(actual, expected)


def test_smooth_assignments_3d_matches_reference_with_padding():
    grid = jnp.zeros((1, 2, 2), dtype=jnp.int32)

    expected = _smooth_assignments_3d_reference(grid, kernel_size=3, num_clusters=2)
    actual = smooth_assignments_3d(grid, kernel_size=3, num_clusters=2)

    assert actual.shape == grid.shape
    assert jnp.array_equal(actual, expected)
