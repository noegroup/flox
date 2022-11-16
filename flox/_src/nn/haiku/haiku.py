""" This module contains haiku modules / implementations. """

from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

import haiku as hk
import jax
from jax_dataclasses import pytree_dataclass

from flox._src.flow.api import (
    Input,
    Inverted,
    Lambda,
    Output,
    Pipe,
    T,
    Transform,
    Transformed,
    bind,
    pure,
)

__all__ = ["LayerStack", "to_haiku", "dense"]

T = TypeVar("T")

@runtime_checkable
class TransformFactory(Protocol[Input, Output]):
    def __call__(self) -> Transform[Input, Output]:
        ...


class LayerStack(hk.Module, Transform[T, T]):
    def __init__(
        self,
        factory: TransformFactory[T, T],
        num_layers=1,
        name="stacked",
    ):
        super().__init__(name=name)

        def body(
            inp: Transformed[T] | None = None,
            out: Transformed[T] | None = None,
        ) -> tuple[Transformed[T] | None, Transformed[T] | None]:
            trafo = factory()

            new_inp = None
            new_out = None

            if inp is not None:
                new_inp = bind(inp, trafo)
            if out is not None:
                new_out = bind(out, Inverted(trafo))

            return new_inp, new_out

        self.stack = hk.without_apply_rng(
            hk.transform(
                hk.experimental.layer_stack(  # type: ignore
                    name="layer_stack",
                    num_layers=num_layers,
                )(body)
            )
        )

    def _apply(
        self,
        inp: Transformed[T] | None = None,
        out: Transformed[T] | None = None,
        reverse: bool = False,
    ) -> tuple[Transformed[T] | None, Transformed[T] | None]:
        init_rng = hk.next_rng_key() if hk.running_init() else None
        params = hk.experimental.transparent_lift(self.stack.init, allow_reuse=True)(  # type: ignore
            init_rng, inp, out, reverse=reverse
        )
        return self.stack.apply(params, inp, out, reverse=reverse)

    def forward(self, input: T) -> Transformed[T]:
        out, _ = self._apply(pure(input), None, reverse=False)
        assert out is not None
        return out

    def inverse(self, input: T) -> Transformed[T]:
        _, out = self._apply(None, pure(input), reverse=True)
        assert out is not None
        return out


def dense(units, activation, name="dense"):
    """utility function that returns a simple densenet made from

    example:
        dense(units=[128, 3], activation=jax.nn.silu)
    """
    layers = []
    with hk.experimental.name_scope(name):  # type: ignore
        for idx, out in enumerate(units):
            layers.append(hk.Linear(out))
            if idx < len(units) - 1:
                layers.append(activation)
    return hk.Sequential(layers, name=name)


@pytree_dataclass(frozen=True)
class HaikuTransform(Generic[Input, Output]):
    pure: hk.MultiTransformed

    def with_params(self, params: dict | Any) -> Transform[Input, Output]:
        forward, inverse = self.pure.apply
        return Lambda[Input, Output](
            jax.tree_util.Partial(forward, params),
            jax.tree_util.Partial(inverse, params),
        )


def to_haiku(
    factory: TransformFactory[Input, Output]
) -> HaikuTransform[Input, Output]:
    def transformed():
        flow = factory()

        def init(input: Input) -> Transformed[Input]:
            return Pipe([flow, Inverted(flow)]).forward(input)

        def forward(input: Input) -> Transformed[Output]:
            return flow.forward(input)

        def inverse(input: Output) -> Transformed[Input]:
            return flow.inverse(input)

        return init, (forward, inverse)

    pure = hk.without_apply_rng(hk.multi_transform(transformed))

    return HaikuTransform[Input, Output](pure)
