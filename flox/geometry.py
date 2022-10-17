from collections.abc import Callable
from enum import Enum
from multiprocessing.sharedctypes import Value
from typing import cast

import jax
import jax.numpy as jnp

from jaxtyping import Float, Array  # type: ignore


Scalar = Float[Array, ""]
VectorN = Float[Array, "N"]


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
    # return x / norm(x)


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


def volume_change(fun: Callable[[VectorN], VectorN]) -> Callable[[VectorN], Scalar]:
    """computes volument change of the function N -> M"""
    jac = jax.jacobian(fun)

    def volume(p: VectorN) -> Scalar:
        j = jac(p)
        if j.shape == (1, 1):
            return j.reshape()
        e = tangent_space(p)
        ej = j @ e.T  # type: ignore
        g = ej.T @ ej
        return jnp.sqrt(jnp.abs(jnp.linalg.det(g)))

    return volume
