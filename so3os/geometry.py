from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Float, Array

Scalar = Float[Array, ""]


def Vector(N: int | str):
    return Float[Array, f"{N}"]


def UnitVector(N: int | str):
    return Vector(N)


def Matrix(N: int | str, M: int | str):
    return Float[Array, f"{N} {M}"]


def TangentBasis(N: int | str):
    if isinstance(N, int) or N.isdigit():
        return Matrix(f"{int(N) - 1}", N)
    else:
        return Matrix(f"{N}-1", N)


def proj(v: Vector("N"), u: Vector("N")) -> Vector("N"):
    """ projects `v` onto `u` """
    return inner(v, u) / inner(u, u) * u


def gram_schmidt(vs: Matrix("M", "N")) -> Matrix("M", "N"):
    """ transforms `num` vectors with `dim` dimensions into an orthonomal bundle
        using the Gram-Schmidt method """
    us = []
    for v in vs:
        u = v - sum(proj(v, u) for u in us)
        u = u / norm(u)
        us.append(u)
    q = jnp.stack(us)
    return q


def inner(a: Vector("N"), b: Vector("N")) -> Scalar:
    """ inner product between two vectors

        NOTE: this is necessary as there is a
              bug when using jnp.inner with jax.jit
    """
    return jnp.tensordot(a, b, (-1, -1))


def norm(x: Vector("N"), eps: float = 1e-12) -> Scalar:
    """ squared norm between two vectors

        NOTE: this is necessary to guarantee
              numerical stability close to
                ||x|| = 0
    """
    return jnp.sqrt(inner(x, x) + eps)


def unit(x: Vector("N")) -> UnitVector("N"):
    """ normalizes a vector and turns it into a unit vector """
    return x / norm(x)


def tangent_space(p: Vector("N"), method: str = "svd") -> TangentBasis("N"):
    """ computes tangent space of a point """
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
    fun: Callable[[Vector("N")], Vector("M")],
    tangent_space: Callable[[Vector("N")], TangentBasis("N")],
) -> Callable[[Vector("N")], Scalar]:
    """ computes volument change of the function N -> M """
    jac = jax.jacobian(fun)

    def volume(p: Vector("N")) -> Scalar:
        j = jac(p)
        if j.shape == (1, 1):
            return j.reshape()
        e = tangent_space(p)
        ej = j @ e.T
        g = ej.T @ ej
        return jnp.sqrt(jnp.linalg.det(g))

    return volume
