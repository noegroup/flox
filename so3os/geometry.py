from collections.abc import Callable

import jax
import jax.numpy as jnp

from .array_types import VectorN, VectorM, UnitVectorN, MatrixNxM, TangentBasisN, Scalar


def proj(v: VectorN, u: VectorN) -> VectorN:
    """projects `v` onto `u`"""
    return inner(v, u) / inner(u, u) * u


def gram_schmidt(vs: MatrixNxM) -> MatrixNxM:
    """transforms `num` vectors with `dim` dimensions into an orthonomal bundle
    using the Gram-Schmidt method"""
    us: list[VectorN] = []
    for v in vs:
        u = v
        for u in us:
            u -= proj(v, u)
        u = u / norm(u)
        us.append(u)
    q = jnp.stack(us)
    return q


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


def unit(x: VectorN) -> UnitVectorN:
    """normalizes a vector and turns it into a unit vector"""
    return x / norm(x)


def tangent_space(p: VectorN, method: str = "svd") -> TangentBasisN:
    """computes tangent space of a point"""
    if method in "svd":
        *_, V = jnp.linalg.svd(jnp.outer(p, p))
        return V[1:]
    elif method == "gram-schmidt":
        dim = len(p)
        vs = jnp.eye(dim)
        idx = jnp.argsort(vs @ p)
        vs = jnp.concatenate([p[None], vs[idx[:-1]]])
        return gram_schmidt(vs)[1:]
    else:
        raise ValueError(f"Unknown method {method}.")


def volume_change(
    fun: Callable[[VectorN], VectorM],
    tangent_space: Callable[[VectorN], TangentBasisN],
) -> Callable[[VectorN], Scalar]:
    """computes volument change of the function N -> M"""
    jac = jax.jacobian(fun)

    def volume(p: VectorN) -> Scalar:
        j = jac(p)
        if j.shape == (1, 1):
            return j.reshape()
        e = tangent_space(p)
        ej = j @ e.T  # type: ignore
        g = ej.T @ ej
        return jnp.sqrt(jnp.linalg.det(g))

    return volume
