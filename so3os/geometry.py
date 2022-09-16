import jax.numpy as jnp
from jaxtyping import Float, Array

Scalar = Float[Array, ""]

Vector3 = Float[Array, "3"]
VectorN = Float[Array, "N"]

UnitVector2 = Float[Array, "2"]
UnitVectorN = Float[Array, "N"]

Quaternion = Float[Array, "4"]

EulerAngles = Float[Array, "3"]

Mat4x4 = Float[Array, "4 4"]
Mat3x3 = Float[Array, "3 3"]
Mat2x2 = Float[Array, "2 2"]


def proj(v: VectorN, u: VectorN) -> VectorN:
    """ projects `v` onto `u` """
    return inner(v, u) / inner(u, u) * u


def gram_schmidt(vs: Float[Array, "num dim"]) -> Float[Array, "num dim"]:
    """ transforms `num` vectors with `dim` dimensions into an orthonomal bundle
        using the Gram-Schmidt method """
    us = []
    for v in vs:
        u = v - sum(proj(v, u) for u in us)
        u = u / norm(u)
        us.append(u)
    q = jnp.stack(us)
    return q * jnp.sign(jnp.linalg.det(vs))


def inner(a: VectorN, b: VectorN) -> Scalar:
    """ inner product between two vectors

        NOTE: this is necessary as there is a
              bug when using jnp.inner with jax.jit
    """
    return jnp.tensordot(a, b, (-1, -1))


def norm(x: VectorN, eps: float = 1e-12) -> Scalar:
    """ squared norm between two vectors

        NOTE: this is necessary to guarantee
              numerical stability close to
                ||x|| = 0
    """
    return jnp.sqrt(inner(x, x) + eps)


def unit(x: VectorN) -> UnitVectorN:
    """ normalizes a vector and turns it into a unit vector """
    return x / norm(x)
