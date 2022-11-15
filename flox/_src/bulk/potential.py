from collections.abc import Callable
from typing import TypeVar

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float  # type: ignore

from flox import geom
from flox._src.geom.manifold import Euclidean, Manifold, TangentN

P = TypeVar("P")

Scalar = Float[Array, ""]
VectorN = Float[Array, "N"]
MatrixNN = Float[Array, "N N"]

PairPotential = Callable[[P, P], Scalar]

__all__ = ["pairwise_penalty", "soft_edge", "lennard_jones_edge"]


def soft_edge(d, dmin):
    return jnp.square(jax.nn.relu((dmin - d) / (dmin)))


def lennard_jones_edge(d, sigma, soften=0.0, min_threshold=1e-5):
    d = jnp.where(d < min_threshold, 1, d + soften)
    rec = sigma / d
    return 4 * (rec**12 - rec**6)


def displacement(
    a: Float[Array, ""], b: Float[Array, ""], m: Manifold = Euclidean()
) -> TangentN:
    return m.tangent(m.projx(a), b - a)


def distance(
    a: Float[Array, ""], b: Float[Array, ""], m: Manifold = Euclidean()
) -> Scalar:
    return geom.norm(displacement(a, b, m))


def pairwise(
    x: VectorN,
    pair_fn: PairPotential,
) -> MatrixNN:
    return jax.vmap(jax.vmap(pair_fn, in_axes=(None, 0)), in_axes=(0, None))(
        x, x
    )


def pairwise_penalty(
    edge: Callable[[Scalar], Scalar],
    manifold: Manifold = Euclidean(),
    mask_diagonal: bool = True,
    reduction: Callable[[MatrixNN], Scalar] = jnp.mean,
):
    def pair_fn(x, y):
        return edge(distance(x, y, manifold))

    def eval(x: VectorN) -> Scalar:
        N = x.shape[0]
        m = pairwise(x, pair_fn)
        if mask_diagonal:
            m = m.at[jnp.arange(N), jnp.arange(N)].set(0.0)
        return reduction(m)

    return eval
