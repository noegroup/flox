""" This module contains the general api for constructing flows. """

from collections.abc import Callable, Reversible
from dataclasses import astuple
from functools import partial
from typing import Any, Generic, Protocol, TypeVar, cast, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
import lenses
from jax_dataclasses import pytree_dataclass
from jaxtyping import Array, Float  # type: ignore

from flox._src.util.func import Lens
from flox._src.util.misc import unpack

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
    "Stack",
]


@pytree_dataclass(frozen=True)
class Transformed(Generic[T]):
    obj: T
    ldj: Volume

    def __iter__(self):
        return unpack(self)


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
    lens: Lens[A, B, C, D]
) -> Lens[A, Transformed[B], C, Transformed[D]]:
    def inject(a: A, new: Transformed[D]) -> Transformed[B]:
        return Transformed(lens.inject(a, new.obj), new.ldj)

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
G = TypeVar("G")
Op = Transform[T, T]


def stack(f: Op[T]):

    params, static = cast(tuple[Op[T], Op[T]], eqx.partition(f, eqx.is_array))

    def _apply_fwd_stack(x: T) -> tuple[Transformed[T], Transformed[T]]:
        def body(
            x: Transformed[T], params: Op[T]
        ) -> tuple[Transformed[T], None]:
            f = cast(Op[T], eqx.combine(params, static))
            return bind(x, f), None

        y, _ = jax.lax.scan(
            body,
            init=pure(x),
            xs=params,
        )

        return y, y

    def _apply_bwd_stack(res: tuple[Transformed[T], Op[T]], g: G) -> tuple[G]:
        def body(
            carry: tuple[G, Transformed[T]], params: Op[T]
        ) -> tuple[tuple[G, Transformed[T]], None]:
            f = cast(Op[T], eqx.combine(params, static))
            g, y = carry
            x = bind(y, f)
            f_ = partial(bind, t=f)
            g: G = jax.vjp(f_, x)[1](g)
            return (g, x), None

        y, f = res

        (g, _), _ = jax.lax.scan(
            body, init=(g, y), xs=Inverted(params), reverse=True
        )

        return (g,)

    @jax.custom_vjp
    def _apply_stack(x: T) -> Transformed[T]:
        return _apply_fwd_stack(x)[0]

    _apply_stack.defvjp(_apply_fwd_stack, _apply_bwd_stack)

    return _apply_stack


class Stack(eqx.Module, Generic[T]):

    stacked: Op[T]

    def forward(self, input: T) -> Transformed[T]:
        return stack(self.stacked)(input)

    def inverse(self, input: T) -> Transformed[T]:
        return stack(Inverted(self.stacked))(input)
