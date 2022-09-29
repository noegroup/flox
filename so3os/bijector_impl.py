from functools import partial

import jax.numpy as jnp

from .array_types import UnitVectorN, VectorN, Scalar, Vector3, Matrix3x3
from .bijector import bijector
from .geometry import tangent_space, volume_change
from .moebius import (
    moebius_project,
    moebius_volume_change,
    double_moebius_project,
    double_moebius_inverse,
    double_moebius_volume_change,
)
from .rigid import (
    from_euclidean,
    to_euclidean,
    to_euclidean_log_jacobian,
    from_euclidean_log_jacobian,
)
from .quaternion import Quaternion


def _affine_forward(
    p: VectorN, shift: VectorN, log_scale: VectorN
) -> tuple[VectorN, Scalar]:
    """transforms p -> p * exp(log_scale) + shift"""
    return jnp.exp(log_scale) * p + shift, jnp.sum(log_scale)


def _affine_inverse(
    p: VectorN, shift: VectorN, log_scale: VectorN
) -> tuple[VectorN, Scalar]:
    """transforms p -> (p - shift) / exp(log_scale)"""
    return _affine_forward(p, -shift * jnp.exp(-log_scale), -log_scale)


def _moebius_forward(p: UnitVectorN, q: VectorN) -> tuple[UnitVectorN, Scalar]:
    p_ = moebius_project(p, q)
    vol = moebius_volume_change(p, q)
    return p_, jnp.log(vol)


def _double_moebius_forward(p: UnitVectorN, q: VectorN) -> tuple[UnitVectorN, Scalar]:
    p_ = double_moebius_project(p, q)
    vol = double_moebius_volume_change(p, q)
    return p_, jnp.log(vol)


def _double_moebius_inverse(p: UnitVectorN, q: VectorN) -> tuple[UnitVectorN, Scalar]:
    p_ = double_moebius_inverse(p, q)
    vol = volume_change(
        partial(double_moebius_inverse, q=q),
        partial(tangent_space, method="gram-schmidt"),
    )(p)
    return p_, jnp.log(vol)


def _rigid_forward(
    q: Quaternion, p: Vector3, r0: Scalar, r1: Scalar, a: Scalar
) -> tuple[Matrix3x3, Scalar]:
    frame = to_euclidean(q, p, r0, r1, a)
    log_vol = to_euclidean_log_jacobian(r0, r1, a)
    return frame, log_vol


def _rigid_inverse(
    frame: Matrix3x3,
) -> tuple[tuple[Quaternion, Vector3, Scalar, Scalar, Scalar], Scalar]:
    q, p, r0, r1, a = from_euclidean(frame)
    log_vol = from_euclidean_log_jacobian(frame)
    return (q, p, r0, r1, a), log_vol


affine_transform = bijector(_affine_forward, _affine_inverse)
moebius_transform = bijector(_moebius_forward, _moebius_forward)
double_moebius_transform = bijector(_double_moebius_forward, _double_moebius_inverse)
rigid_transform = bijector(_rigid_forward, _rigid_inverse)
