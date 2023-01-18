""" Everything related to basic Euclidean geometry. """

from typing import cast

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float  # type: ignore

__all__ = [
    "proj",
    "gram_schmidt",
    "outer",
    "inner",
    "norm",
    "squared_norm",
    "unit",
    "det",
    "det3x3",
]

Scalar = Float[Array, ""]
VectorN = Float[Array, "N"]
Matrix3x3 = Float[Array, "3"]
MatrixNxN = Float[Array, "N"]


def proj(v: VectorN, u: VectorN) -> VectorN:
    """projects `v` onto `u`"""
    return inner(v, u) / inner(u, u) * u


def gram_schmidt(vs: Float[Array, "M N"]) -> Float[Array, "M N"]:
    """transforms `num` vectors with `dim` dimensions into an orthonomal bundle
    using the Gram-Schmidt method"""
    us: list[VectorN] = []
    for v in vs:
        u = v - sum(proj(v, u) for u in us)
        u = u / norm(u)
        us.append(u)
    q = jnp.stack(us)
    return cast(Array, q)


def outer(a: VectorN, b: VectorN) -> Float[Array, "N N"]:
    return a[:, None] * b[None, :]


def inner(a: VectorN, b: VectorN) -> Scalar:
    """inner product between two vectors

    NOTE: this is necessary as there is a
          bug when using jnp.inner with jax.jit
    """
    return jnp.tensordot(a, b, (-1, -1))  # type: ignore


def norm(x: VectorN, eps: float = 1e-12) -> Scalar:
    """norm between two vectors

    NOTE: this is necessary to guarantee
          numerical stability close to
            ||x|| = 0
    """
    return jnp.sqrt(squared_norm(x) + eps)


def squared_norm(x: VectorN, eps: float = 1e-12) -> Scalar:
    """norm between two vectors

    NOTE: this is necessary to guarantee
          numerical stability close to
            ||x|| = 0
    """
    return inner(x, x)


def unit(x: VectorN) -> VectorN:
    """normalizes a vector and turns it into a unit vector"""
    return x * jax.lax.rsqrt(squared_norm(x))


def det1x1(M: Array) -> Scalar:
    return M[0, 0]


def det2x2(M: Array) -> Scalar:
    return M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]


def det3x3(M: Matrix3x3) -> Scalar:
    return inner(M[0], jnp.cross(M[1], M[2]))


def det(M: MatrixNxN) -> Scalar:
    match M.shape:
        case (1, 1):
            return det1x1(M)
        case (2, 2):
            return det2x2(M)
        case (3, 3):
            return det3x3(M)
        case _:
            return jnp.linalg.det(M)
