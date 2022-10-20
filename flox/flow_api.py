""" This module contains the general api for constructing flows. """

from collections.abc import Reversible
from functools import reduce
from typing import Callable, Generic, Protocol, TypeVar, cast, runtime_checkable

from jax_dataclasses import pytree_dataclass
from jaxtyping import Array, Float  # type: ignore

from flox.func_utils import Lens

Volume = Float[Array, ""]

Input = TypeVar("Input")
Output = TypeVar("Output")

T = TypeVar("T")

A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")
D = TypeVar("D")

X = TypeVar("X")
Y = TypeVar("Y")


@pytree_dataclass(frozen=True)
class Transformed(Generic[T]):
    payload: T
    logprob: Volume


@runtime_checkable
class Transform(Protocol[Input, Output]):
    def forward(self, input: Input) -> Transformed[Output]:
        ...

    def inverse(self, input: Output) -> Transformed[Input]:
        ...


@runtime_checkable
class VolumeAccumulator(Protocol[Input, Output]):
    def forward(self, input: Transformed[Input]) -> Transformed[Output]:
        ...

    def inverse(self, input: Transformed[Output]) -> Transformed[Input]:
        ...


def unpack_volume(lens: Lens[A, B, C, D]) -> Lens[A, Transformed[B], C, Transformed[D]]:
    def inject(a: A, new: Transformed[D]) -> Transformed[B]:
        return Transformed(lens.inject(a, new.payload), new.logprob)

    return Lens(lens.project, inject)


@pytree_dataclass(frozen=True)
class LensedTransform(Transform[A, B], Generic[A, B, C, D]):
    transform: Transform[C, D]
    forward_lens: Lens[A, B, C, D]
    inverse_lens: Lens[B, A, D, C]

    def forward(self, input: A) -> Transformed[B]:
        return unpack_volume(self.forward_lens)(self.transform.forward)(input)

    def inverse(self, input: B) -> Transformed[A]:
        return unpack_volume(self.inverse_lens)(self.transform.inverse)(input)


@pytree_dataclass(frozen=True)
class Coupling(Transform[A, B], Generic[A, B, C, D, X, Y]):
    get_params: Callable[[X], Y]
    get_forward_transform: Lens[A, Transform[C, D], X, Y]
    get_inverse_transform: Lens[B, Transform[C, D], X, Y]
    get_forward_target: Lens[A, B, C, D]
    get_inverse_target: Lens[B, A, D, C]

    def forward(self, input: A) -> Transformed[B]:
        transform = self.get_forward_transform(self.get_params)(input)
        return LensedTransform(
            transform, self.get_forward_target, self.get_inverse_target
        ).forward(input)

    def inverse(self, input: B) -> Transformed[A]:
        transform = self.get_inverse_transform(self.get_params)(input)
        return LensedTransform(
            transform, self.get_forward_target, self.get_inverse_target
        ).inverse(input)


@pytree_dataclass(frozen=True)
class SimpleCoupling(Transform[A, A], Generic[A, C]):

    transform_cstr: Callable[[A], Transform[C, C]]
    get_target: Lens[A, A, C, C]

    def forward(self, input: A) -> Transformed[A]:
        transform = self.transform_cstr(input)
        return LensedTransform(transform, self.get_target, self.get_target).forward(
            input
        )

    def inverse(self, input: A) -> Transformed[A]:
        transform = self.transform_cstr(input)
        return LensedTransform(transform, self.get_target, self.get_target).inverse(
            input
        )


@pytree_dataclass(frozen=True)
class AdditiveVolumeAccumulator(VolumeAccumulator[Input, Output]):
    transform: Transform[Input, Output]

    def forward(self, input: Transformed[Input]) -> Transformed[Output]:
        transformed = self.transform.forward(input.payload)
        return Transformed(transformed.payload, input.logprob + transformed.logprob)

    def inverse(self, input: Transformed[Output]) -> Transformed[Input]:
        transformed = self.transform.inverse(input.payload)
        return Transformed(transformed.payload, input.logprob + transformed.logprob)


@pytree_dataclass(frozen=True)
class Sequential(Generic[Input, Output]):
    transforms: Reversible[VolumeAccumulator]

    def forward(self, input: Transformed[Input]) -> Transformed[Output]:
        return cast(
            Transformed[Output],
            reduce(
                lambda input, transform: transform.forward(input),
                self.transforms,
                input,
            ),
        )

    def inverse(self, input: Transformed[Output]) -> Transformed[Input]:
        return cast(
            Transformed[Input],
            reduce(
                lambda input, transform: transform.inverse(input),
                reversed(self.transforms),
                input,
            ),
        )
