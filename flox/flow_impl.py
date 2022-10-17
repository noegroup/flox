from dataclasses import astuple, dataclass
import jax.numpy as jnp
from jaxtyping import Float, Array  # type: ignore
from .convex import (
    forward_log_volume,
    inverse_log_volume,
    numeric_inverse,
    potential_gradient,
)

from .moebius import (
    double_moebius_inverse,
    double_moebius_inverse_volume_change,
    double_moebius_project,
    double_moebius_volume_change,
    moebius_project,
    moebius_volume_change,
)
import rigid
from .flow_api import Transformed, Volume


VectorN = Float[Array, "N"]
VectorM = Float[Array, "M"]
MatrixMxN = Float[Array, "M N"]
Matrix3x3 = Float[Array, "3 3"]


def affine_forward(
    p: VectorN, shift: VectorN, log_scale: VectorN
) -> tuple[VectorN, Volume]:
    """transforms p -> p * exp(log_scale) + shift"""
    return jnp.exp(log_scale) * p + shift, jnp.sum(log_scale)


def affine_inverse(
    p: VectorN, shift: VectorN, log_scale: VectorN
) -> tuple[VectorN, Volume]:
    """transforms p -> (p - shift) / exp(log_scale)"""
    return affine_forward(p, -shift * jnp.exp(-log_scale), -log_scale)


@dataclass
class Affine:
    shift: VectorN
    scale: VectorN

    def forward(self, input: VectorN) -> Transformed[VectorN]:
        return Transformed(*affine_forward(input, self.shift, self.scale))

    def inverse(self, input: VectorN) -> Transformed[VectorN]:
        return Transformed(*affine_inverse(input, self.shift, self.scale))


def moebius_forward(p: VectorN, q: VectorN) -> tuple[VectorN, Volume]:
    p_ = moebius_project(p, q)
    vol = moebius_volume_change(p, q)
    return p_, jnp.log(vol)


@dataclass
class Moebius:
    reflection: VectorN

    def forward(self, input: VectorN) -> Transformed[VectorN]:
        return Transformed(*moebius_forward(input, self.reflection))

    def inverse(self, input: VectorN) -> Transformed[VectorN]:
        return Transformed(*moebius_forward(input, self.reflection))


def double_moebius_forward_transform(p: VectorN, q: VectorN) -> tuple[VectorN, Volume]:
    p_ = double_moebius_project(p, q)
    vol = double_moebius_volume_change(p, q)
    return p_, jnp.log(vol)


def double_moebius_inverse_transform(p: VectorN, q: VectorN) -> tuple[VectorN, Volume]:
    p_ = double_moebius_inverse(p, q)
    vol = double_moebius_inverse_volume_change(p, q)
    return p_, jnp.log(vol)


@dataclass
class DoubleMoebius:
    reflection: VectorN

    def forward(self, input: VectorN) -> Transformed[VectorN]:
        return Transformed(*double_moebius_forward_transform(input, self.reflection))

    def inverse(self, input: VectorN) -> Transformed[VectorN]:
        return Transformed(*double_moebius_inverse_transform(input, self.reflection))


@dataclass
class ConvexPotential:
    ctrlpts: MatrixMxN
    weights: VectorM
    bias: VectorM

    def forward(self, input: VectorN) -> Transformed[VectorN]:
        output = potential_gradient(input, self.ctrlpts, self.weights, self.bias)
        logprob = forward_log_volume(potential_gradient)(
            input, self.ctrlpts, self.weights, self.bias
        )
        return Transformed(output, logprob)

    def inverse(self, input: VectorN) -> Transformed[VectorN]:
        output = numeric_inverse(potential_gradient)(
            input, self.ctrlpts, self.weights, self.bias
        )
        logprob = inverse_log_volume(potential_gradient)(
            output, self.ctrlpts, self.weights, self.bias
        )
        return Transformed(output, logprob)


@dataclass
class Rigid:
    def forward(self, input: Matrix3x3) -> Transformed[rigid.Rigid]:
        output = rigid.Rigid(*rigid.from_euclidean(input))
        logprob = rigid.from_euclidean_log_jacobian(input)
        return Transformed(output, logprob)

    def inverse(self, input: rigid.Rigid) -> Transformed[Matrix3x3]:
        output = rigid.to_euclidean(*astuple(input))
        logprob = rigid.to_euclidean_log_jacobian(*astuple(input))
        return Transformed(output, logprob)
