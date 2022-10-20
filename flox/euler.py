""" Everything related to Euler-angle representations of SO(3). """

from functools import partial
from typing import cast

import jax.numpy as jnp
from jaxtyping import Array, Float  # type: ignore

Matrix2x2 = Float[Array, "2 2"]
Matrix3x3 = Float[Array, "3 3"]
EulerAngles = Float[Array, "3"]
Scalar = Float[Array, ""]


def rotmat2d(theta: Scalar) -> Matrix2x2:
    """returns a 2D rotation matrix
    rotating around the angle `theta`"""
    ct = jnp.cos(theta)
    st = jnp.sin(theta)
    return cast(Array, jnp.array([[ct, -st], [st, ct]]))


def canonical_rotation(theta: Scalar, axis: int) -> Matrix3x3:
    """returns matrix of a 2D rotation in 3D
    along the specified `axis`"""
    axes = jnp.array(list(set(range(3)) - {axis}))
    x, y = jnp.meshgrid(axes, axes)
    return jnp.eye(3).at[x, y].set(rotmat2d(theta))


def to_euler(R: Matrix3x3) -> EulerAngles:
    """converts a 3x3 rotation matrix to canonical rotations
    given in ZXZ euler angles"""
    alpha = jnp.arctan2(R[0, 2], R[1, 2])
    beta = jnp.arccos(R[2, 2])
    gamma = jnp.arctan2(R[2, 0], -R[2, 1])
    return cast(Array, jnp.stack([alpha, beta, gamma]))


def from_euler(angles: EulerAngles) -> Matrix3x3:
    """converts canonical rotations given in ZXZ euler angles
    into 3x3 rotation matrices"""
    alpha, beta, gamma = angles
    rotz = partial(canonical_rotation, axis=2)
    rotx = partial(canonical_rotation, axis=0)
    return rotz(alpha) @ (rotx(beta) @ rotz(gamma))
