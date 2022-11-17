from enum import Enum
from typing import cast

import equinox as eqx
import jax.numpy as jnp
from jax_dataclasses import pytree_dataclass
from jaxtyping import Array, Float  # type: ignore

from .euclidean import gram_schmidt, norm, unit

__all__ = [
    "Torus",
    "Sphere",
    "Euclidean",
    "TangentSpaceMethod",
    "tangent_space",
]

PointN = Float[Array, "N"]
VectorN = Float[Array, "N"]
TangentN = Float[Array, "N"]
Scalar = Float[Array, ""]


class Manifold:
    def expm(self, x: PointN, dx: VectorN, t: Scalar) -> PointN:
        ...

    def tangent(self, x: PointN, dx: VectorN) -> TangentN:
        ...

    def shift(self, x: PointN, dx: VectorN) -> PointN:
        ...

    def projx(self, x: PointN) -> PointN:
        ...


@pytree_dataclass(frozen=True)
class Euclidean(Manifold):
    def expm(self, x: PointN, dx: VectorN, t: Scalar) -> PointN:
        return self.shift(x, t * dx)

    def tangent(self, x: PointN, dx: VectorN) -> TangentN:
        return dx

    def shift(self, x: PointN, dx: VectorN) -> PointN:
        return self.projx(x + dx)

    def projx(self, x: VectorN) -> PointN:
        return x


@pytree_dataclass(frozen=True)
class Torus(Manifold):
    box: Float[Array, "N"]

    def expm(self, x: PointN, dx: VectorN, t: Scalar) -> PointN:
        return self.shift(x, t * dx)

    def tangent(self, x: PointN, dx: VectorN) -> TangentN:
        return jnp.mod(dx + self.box * 0.5, self.box) - self.box * 0.5

    def shift(self, x: PointN, dx: VectorN) -> PointN:
        return jnp.mod(x + dx, self.box)

    def projx(self, x: VectorN) -> PointN:
        return jnp.mod(x, self.box)


@pytree_dataclass(frozen=True)
class Sphere(Manifold):
    def expm(self, x: PointN, dx: VectorN, t: Scalar) -> PointN:
        n = norm(dx)
        return jnp.cos(t * n) * x + jnp.sin(t * n) * (dx / n)

    def tangent(self, x: PointN, dx: VectorN) -> TangentN:
        E = tangent_space(x)
        return E.T @ E @ dx  # type: ignore

    def shift(self, x: PointN, dx: VectorN) -> PointN:
        return self.expm(x, dx, jnp.ones(()))

    def projx(self, x: PointN) -> PointN:
        return cast(PointN, unit(x))


class TangentSpaceMethod(Enum):
    GramSchmidt = 0
    SVD = 1


def tangent_space(
    p: VectorN, method: TangentSpaceMethod = TangentSpaceMethod.GramSchmidt
) -> Float[Array, "N-1 N"]:
    """computes tangent space of a point"""
    tangent_space = None
    match (method):
        case TangentSpaceMethod.SVD:
            *_, V = jnp.linalg.svd(jnp.outer(p, p))
            tangent_space = V[1:]
        case TangentSpaceMethod.GramSchmidt:
            dim = len(p)
            vs = jnp.eye(dim)
            idx = jnp.argsort(vs @ p)
            vs = jnp.concatenate([p[None], vs[idx[:-1]]])
            tangent_space = gram_schmidt(cast(Array, vs))[1:]
        case _:
            raise ValueError(f"Unknown tangent space method {method}.")
    return tangent_space
