""" This module contains the general api for constructing flows. """

from collections.abc import Callable, Reversible
from functools import partial
from typing import Any, Generic, Protocol, TypeVar, cast, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
import lenses
from jax_dataclasses import pytree_dataclass
from jaxtyping import Array, Float  # type: ignore

from flox._src.util.func import Lens, LensLike, lift

Volume = Float[Array, ""]

Input = TypeVar("Input")
Output = TypeVar("Output")

T = TypeVar("T")
S = TypeVar("S")

A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")
D = TypeVar("D")

X = TypeVar("X")
Y = TypeVar("Y")

__all__ = [
    "Transformed",
    "Transform",
    "pure",
    "bind",
    "LensedTransform",
    "Coupling",
    "SimpleCoupling",
    "Pipe",
    "Lambda",
    "Inverted",
    "VectorizedTransform",
    "LayerStack",
]


@pytree_dataclass(frozen=True)
class Transformed(Generic[T]):
    obj: T
    ldj: Volume


@runtime_checkable
class Transform(Protocol[Input, Output]):
    def forward(self, input: Input) -> Transformed[Output]:
        ...

    def inverse(self, input: Output) -> Transformed[Input]:
        ...


def pure(obj: T) -> Transformed[T]:
    """monadic pure operator"""
    return Transformed(obj, jnp.zeros(()))


def bind(a: Transformed[T], t: Transform[T, S]) -> Transformed[S]:
    """monadic bind operator"""
    tb = t.forward(a.obj)
    return Transformed(tb.obj, tb.ldj + a.ldj)


def unpack_volume(
    lens: LensLike[A, B, C, D]
) -> Lens[A, Transformed[B], C, Transformed[D]]:
    def inject(a: A, new: Transformed[D]) -> Transformed[B]:
        return Transformed(lens.inject(a, new.obj), new.ldj)

    return Lens(lens.project, inject)


@pytree_dataclass(frozen=True)
class LensedTransform(Transform[A, B], Generic[A, B, C, D]):
    transform: Transform[C, D]
    forward_lens: LensLike[A, B, C, D]
    inverse_lens: LensLike[B, A, D, C]

    def forward(self, input: A) -> Transformed[B]:
        return unpack_volume(self.forward_lens)(self.transform.forward)(input)

    def inverse(self, input: B) -> Transformed[A]:
        return unpack_volume(self.inverse_lens)(self.transform.inverse)(input)


@pytree_dataclass(frozen=True)
class Coupling(Transform[A, B], Generic[A, B, C, D, X, Y]):
    get_params: Callable[[X], Y]
    get_forward_transform: LensLike[A, Transform[C, D], X, Y]
    get_inverse_transform: LensLike[B, Transform[C, D], X, Y]
    get_forward_target: LensLike[A, B, C, D]
    get_inverse_target: LensLike[B, A, D, C]

    def forward(self, input: A) -> Transformed[B]:
        transform = lift(self.get_forward_transform, self.get_params)(input)
        return LensedTransform(
            transform, self.get_forward_target, self.get_inverse_target
        ).forward(input)

    def inverse(self, input: B) -> Transformed[A]:
        transform = lift(self.get_inverse_transform, self.get_params)(input)
        return LensedTransform(
            transform, self.get_forward_target, self.get_inverse_target
        ).inverse(input)


@pytree_dataclass(frozen=True)
class SimpleCoupling(Transform[A, A], Generic[A, C]):

    transform_cstr: Callable[[A], Transform[C, C]]
    get_target: Lens[A, A, C, C]

    def forward(self, input: A) -> Transformed[A]:
        transform = self.transform_cstr(input)
        return LensedTransform(
            transform, self.get_target, self.get_target
        ).forward(input)

    def inverse(self, input: A) -> Transformed[A]:
        transform = self.transform_cstr(input)
        return LensedTransform(
            transform, self.get_target, self.get_target
        ).inverse(input)


@pytree_dataclass(frozen=True)
class Pipe(Generic[Input, Output]):
    transforms: Reversible[Transform[Any, Any]]

    def forward(self, input: Input) -> Transformed[Output]:
        accum = pure(input)
        for t in self.transforms:
            accum = bind(accum, t)
        return cast(Transformed[Output], accum)

    def inverse(self, input: Output) -> Transformed[Input]:
        accum = pure(input)
        for t in map(Inverted[Any, Any], reversed(self.transforms)):
            accum = bind(accum, t)
        return cast(Transformed[Input], accum)


@pytree_dataclass(frozen=True)
class Lambda(Transform[Input, Output]):
    forward: Callable[[Input], Transformed[Output]]
    inverse: Callable[[Output], Transformed[Input]]


@pytree_dataclass(frozen=True)
class Inverted(Transform[Output, Input]):
    inverted: Transform[Input, Output]

    def forward(self, input: Output) -> Transformed[Input]:
        return self.inverted.inverse(input)

    def inverse(self, input: Input) -> Transformed[Output]:
        return self.inverted.forward(input)


@pytree_dataclass(frozen=True)
class VectorizedTransform(Transform[Input, Output]):
    """vmap wrapper for transforms"""

    vmapped: Transform[Input, Output]
    ldj_reduction: Callable[[Any], Volume] | None = jnp.sum

    def forward(self, input: Input) -> Transformed[Output]:
        out = eqx.filter_vmap(type(self.vmapped).forward)(self.vmapped, input)
        if self.ldj_reduction is not None:
            out = lenses.bind(out).ldj.modify(self.ldj_reduction)
        return Transformed(out.obj, out.ldj)

    def inverse(self, input: Output) -> Transformed[Input]:
        out = eqx.filter_vmap(type(self.vmapped).inverse)(self.vmapped, input)
        if self.ldj_reduction is not None:
            out = lenses.bind(out).ldj.modify(self.ldj_reduction)
        return Transformed(out.obj, out.ldj)


T = TypeVar("T")
Op = Transform[T, T]


def layer_stack(x: T, f: Op[T], reverse: bool, unroll: int) -> Transformed[T]:

    params, static = eqx.partition(f, eqx.is_array)  # type: ignore

    def body(x: Transformed[T], params: Op[T]) -> tuple[Transformed[T], None]:
        f: Op[T] = cast(Op[T], eqx.combine(params, static))
        return bind(x, f), None  # type: ignore

    out, _ = jax.lax.scan(body, pure(x), params, reverse=reverse, unroll=unroll)
    return out


class LayerStack(eqx.Module, Generic[T]):

    stacked: Op[T]
    unroll: int = 1

    def forward(self, input: T) -> Transformed[T]:
        return layer_stack(
            input, self.stacked, reverse=False, unroll=self.unroll
        )

    def inverse(self, input: T) -> Transformed[T]:
        return layer_stack(
            input, Inverted(self.stacked), reverse=True, unroll=self.unroll
        )
