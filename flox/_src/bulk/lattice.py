""" everything related to particle densities """

from enum import Enum
from typing import cast

import jax.numpy as jnp
from jaxtyping import Array, Float, Integer  # type: ignore

__all__ = [
    "neighbors",
    "lattice_indices",
    "lattice_weights",
    "scatter",
    "gather",
]


def neighbors(num_dims: int, num_neighbors: int = 1) -> jnp.ndarray:
    xi = (
        jnp.linspace(
            -num_neighbors,
            num_neighbors,
            2 * num_neighbors + 1,
            dtype=jnp.int32,
        )
        for _ in range(num_dims)
    )
    grid = jnp.meshgrid(*xi, indexing="xy")
    return jnp.reshape(jnp.stack(grid, -1), (-1, num_dims))


def lattice_indices(
    positions: Float[Array, "D"],
    neighbors: Float[Array, "N D"],
    resolution: tuple[int, ...],
) -> Integer[Array, "N D"]:
    """N: neighbors, D: spatial dimensions"""
    res = cast(Array, jnp.array(resolution))
    return (jnp.floor(positions[None, :] * res[None, :] + neighbors)).astype(
        jnp.int32
    )


def lattice_weights(
    positions: Float[Array, "D"],
    idxs: Integer[Array, "N D"],
    resolution: tuple[int, ...],
    gain=0.5,
) -> jnp.ndarray:
    """N: neighbors, D: spatial dimensions"""
    res = cast(Array, jnp.array(resolution))
    diff = jnp.abs(positions[None] * res[None] - idxs - 0.5)
    diff = jnp.minimum(diff, res[None] - diff)
    dist2 = jnp.sum(jnp.square(diff), axis=1)
    weights = jnp.exp(-dist2 * gain)
    weights = weights / (jnp.sum(weights, keepdims=True) + 1e-6)
    return weights


def scatter(inp, weights, idxs, res):
    """
    scatters a signal per particle onto a lattice of given resolution

    inp: [*B, P, *F]
    weights: [*B, P, N]
    idxs: [*B, P, N, D]
    res: tuple of length D

    out: [*B, *D, *F]
    """
    batch = idxs.shape[:-3]
    nbatch = len(batch)
    feats = inp.shape[nbatch + 1 :]
    nfeats = len(feats)

    # weigh signal
    # [*B, P, *F], [*B, P, N] -> [*B, P, N, *F]
    bslices = (slice(None),) * (len(batch) + 1)
    weighed_signal = (
        inp[bslices + (None,)]
        * weights[bslices + (slice(None),) + (None,) * len(feats)]
    )

    # allocate empty lattice
    # [*B, *D, *F]
    lattice = jnp.zeros(batch + res + feats)

    # scatter signal into lattice
    # [*B, *D, *F]
    slices = jnp.ogrid[tuple(map(slice, idxs.shape[:-1]))]
    bslices = tuple(slices[:nbatch])
    islices = tuple(idxs[..., i] for i in range(len(res)))
    return lattice.at[bslices + islices].add(weighed_signal)


class AggregationMethod(Enum):
    Sum = "sum"
    Nothing = None


def gather(
    inp, idxs, aggregation_method: AggregationMethod = AggregationMethod.Sum
):
    """
    gathers a signal on a lattice onto individual particles

    inp: [*B, *D, *F]
    idxs: [*B, P, N, D]

    out: [*B, P, *F] / [*B, P, N, *F]
    """
    batch = idxs.shape[:-3]
    nbatch = len(batch)
    ndims = idxs.shape[-1]

    slices = jnp.ogrid[tuple(map(slice, idxs.shape[:-1]))]
    bslices = tuple(slices[:nbatch])
    islices = tuple(idxs[..., i] for i in range(ndims))

    out = inp[bslices + islices]
    match aggregation_method:
        case AggregationMethod.Sum:
            out = out.sum(axis=nbatch + ndims - 1)
        case AggregationMethod.Nothing:
            out = out
        case _:
            raise ValueError(f"Unknown aggregation method {aggregation_method}")
    return out
