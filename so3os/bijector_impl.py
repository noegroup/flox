from functools import partial
from typing import Tuple

import jax.numpy as jnp

from .bijector import bijector
from .geometry import Vector, Scalar, UnitVector, tangent_space, volume_change
from .moebius import (
    moebius_project,
    moebius_volume_change,
    double_moebius_project,
    double_moebius_inverse,
    double_moebius_volume_change,
    double_moebius_inverse_volume_change,
)


def _affine_forward(
    p: Vector("N"), shift: Vector("N"), log_scale: Vector("N")
) -> Tuple[Vector("N"), Scalar]:
    """ transforms p -> p * exp(log_scale) + shift """
    return jnp.exp(log_scale) * p + shift, jnp.sum(log_scale)


def _affine_inverse(
    p: Vector("N"), shift: Vector("N"), log_scale: Vector("N")
) -> Tuple[Vector("N"), Scalar]:
    """ transforms p -> (p - shift) / exp(log_scale) """
    return _affine_forward(p, -shift * jnp.exp(-log_scale), -log_scale)


def _moebius_forward(
    p: UnitVector("N"), q: Vector("N")
) -> Tuple[UnitVector("N"), Scalar]:
    p_ = moebius_project(p, q)
    vol = moebius_volume_change(p, q)
    return p_, jnp.log(vol)


def _double_moebius_forward(
    p: UnitVector("N"), q: Vector("N")
) -> Tuple[UnitVector("N"), Scalar]:
    p_ = double_moebius_project(p, q)
    vol = double_moebius_volume_change(p, q)
    return p_, jnp.log(vol)


def _double_moebius_inverse(
    p: UnitVector("N"), q: Vector("N")
) -> Tuple[UnitVector("N"), Scalar]:
    p_ = double_moebius_inverse(p, q)
    vol = double_moebius_inverse_volume_change(p, q)
    return p_, jnp.log(vol)


affine_transform = bijector(_affine_forward, _affine_inverse)
moebius_transform = bijector(_moebius_forward, _moebius_forward)
double_moebius_transform = bijector(_double_moebius_forward, _double_moebius_inverse)
