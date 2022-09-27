from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp

from .geometry import Matrix, Vector, Scalar, unit, norm, inner, squared_norm
from .quaternion import Quaternion, qrot3d, mat_to_quat


def tripod(frame: Matrix(3, 3)) -> Matrix(3, 3):
    """ computes a unique orthogonal basis for input points """
    p1, p2, p3 = frame
    e1 = unit(p2 - p1)
    e2 = unit(jnp.cross(p3 - p1, e1))
    e3 = jnp.cross(e2, e1)
    return jnp.array([-e3, -e2, e1])


def to_euclidean(
    q: Quaternion, p: Vector(3), r0: Scalar, r1: Scalar, a: Scalar,
):
    """ converts from pose + internal dofs into euclidean coordinates

        the internal dofs are the two edge lengths and the inner angle
        of the triangle spanned by the three points defining the frame

                + p0
          r0 -> |
                |
            p1  + <- a
                 \
            r1 -> \
                   + p1

        q: global rotation
        p: global translation
        r0, r1: edge lengths of the triangle
        a: inner angle of the triangle
    """
    frame = jnp.array([[0, 0, 0], [0, 0, r0], [r1 * jnp.cos(a), 0, r1 * jnp.sin(a)]])
    return jax.vmap(partial(qrot3d, q))(frame) + p[None]


def from_euclidean(
    frame: Matrix(3, 3),
) -> Tuple[
    Quaternion, Vector(3), Scalar, Scalar, Scalar,
]:
    """ converts an euclidean frame into pose + internal dofs

        the internal dofs are the two edge lengths and the inner angle
        of the triangle spanned by the three points defining the frame
        (see `to_euclidean`)
    """
    q = mat_to_quat(tripod(frame))
    p = frame[(0,)]
    frame = frame - p
    r0 = norm(frame[0] - frame[1])
    r1 = norm(frame[0] - frame[2])
    a = jnp.arcsin(inner(unit(frame[1]), unit(frame[2])))
    return q, p, r0, r1, a


def to_euclidean_log_jacobian(r0: Scalar, r1: Scalar, a: Scalar) -> Scalar:
    r0sq = r0 ** 2
    r1sq = r1 ** 2
    return jnp.log(8 * r0sq * r1sq * jnp.cos(a)) + 0.5 * jnp.log(4 * (r0sq + r1sq) + 1)


def from_euclidean_log_jacobian(frame: Matrix(3, 3)) -> Scalar:
    p0 = frame[(0,)]
    frame = frame - p0
    r0sq = squared_norm(frame[0] - frame[1])
    r1sq = squared_norm(frame[0] - frame[2])
    cos_a_sq = 1 - inner(unit(frame[1]), unit(frame[2])) ** 2
    return -jnp.log(8 * r0sq * r1sq) - 0.5 * (
        jnp.log(cos_a_sq * (4 * (r0sq + r1sq) + 1))
    )
