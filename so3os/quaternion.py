import jax.numpy as jnp

from .jax_utils import Switch
from .geometry import Vector3, Quaternion, Mat4x4, Mat3x3


def to_quat(r: Vector3) -> Quaternion:
    """ embeds a 3D vector as a quaternion """
    return jnp.concatenate((jnp.zeros((1,)), r))


def from_quat(q: Quaternion) -> Vector3:
    """ projects a quaternion on its 3D vector component """
    return q[1:]


def qprod(qa: Quaternion, qb: Quaternion) -> Quaternion:
    """ quaternion product """
    w0, x0, y0, z0 = qa
    w1, x1, y1, z1 = qb
    return jnp.stack(
        [
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
        ]
    )


def qmat(q: Quaternion) -> Mat4x4:
    """ maps a quaternion on its 4x4 matrix representing the
        quaternion product """
    w1, x1, y1, z1 = q
    return jnp.stack(
        [
            jnp.stack([w1, -x1, -y1, -z1]),
            jnp.stack([x1, w1, -z1, y1]),
            jnp.stack([y1, z1, w1, -x1]),
            jnp.stack([z1, -y1, x1, w1]),
        ]
    )


def qconj(q: Quaternion) -> Quaternion:
    """ conjugates the quaternion """
    w0, x0, y0, z0 = q
    return jnp.stack([w0, -x0, -y0, -z0])


def qrot3d(q: Quaternion, r: Vector3) -> Vector3:
    """ rotates a 3D vector by a quaternion """
    p = to_quat(r)
    p_ = qprod(q, qprod(p, qconj(q)))
    return from_quat(p_)


def quat_to_mat(q: Quaternion) -> Mat3x3:
    """ projects quaternion onto the corresponding
        3x3 rotation matrix

        this is a surjective but not injective map!
    """
    qw, qi, qj, qk = q
    return jnp.stack(
        [
            jnp.stack(
                [
                    1 - 2 * (qj ** 2 + qk ** 2),
                    2 * (qi * qj - qk * qw),
                    2 * (qi * qk + qj * qw),
                ]
            ),
            jnp.stack(
                [
                    2 * (qi * qj + qk * qw),
                    1 - 2 * (qi ** 2 + qk ** 2),
                    2 * (qj * qk - qi * qw),
                ]
            ),
            jnp.stack(
                [
                    2 * (qi * qk - qj * qw),
                    2 * (qj * qk + qi * qw),
                    1 - 2 * (qi ** 2 + qj ** 2),
                ]
            ),
        ]
    )


def mat_to_quat(R: Mat3x3) -> Quaternion:
    """ embeds 3x3 rotation matrix as *one* of two possible
        quaternion preimages

        this is an injective but not surjective map
    """
    tr = jnp.trace(R)

    switch = Switch()

    @switch.add_branch(tr > 0)
    def _(R):
        s = jnp.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
        return jnp.stack([qw, qx, qy, qz])

    @switch.add_branch((R[0, 0] > R[1, 1]) & (R[0, 0] > R[2, 2]))
    def _(R):
        s = jnp.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
        return jnp.stack([qw, qx, qy, qz])

    @switch.add_branch((R[1, 1] > R[2, 2]))
    def _(R):
        s = jnp.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
        return jnp.stack([qw, qx, qy, qz])

    @switch.default_branch()
    def _(R):
        s = jnp.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
        return jnp.stack([qw, qx, qy, qz])

    return switch(R)
