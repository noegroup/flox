""" This module contains haiku modules / implementations. """

from functools import partial
from typing import Callable, Generic, Protocol, TypeVar

import haiku as hk
from jax_dataclasses import pytree_dataclass

from flox.func_utils import pipe2

from .flow_api import Input, Output, T, Transformed, VolumeAccumulator

T = TypeVar("T")


class VolumeAccumulatorFactory(Protocol[Input, Output]):
    def __call__(self) -> VolumeAccumulator[Input, Output]:
        ...


class LayerStack(hk.Module, VolumeAccumulator[T, T]):
    def __init__(
        self,
        factory: VolumeAccumulatorFactory[T, T],
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
                new_inp = trafo.forward(inp)
            if out is not None:
                new_out = trafo.inverse(out)

            return new_inp, new_out

        self.stack = hk.without_apply_rng(
            hk.transform(
                hk.experimental.layer_stack(
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
        params = hk.experimental.transparent_lift(self.stack.init, allow_reuse=True)(
            init_rng, inp, out, reverse=reverse
        )
        return self.stack.apply(params, inp, out, reverse=reverse)

    def forward(self, input: Transformed[T]) -> Transformed[T]:
        out, _ = self._apply(input, None, reverse=False)
        assert out is not None
        return out

    def inverse(self, input: Transformed[T]) -> Transformed[T]:
        _, out = self._apply(None, input, reverse=True)
        assert out is not None
        return out


def dense(units, activation, name="dense"):
    """ utility function that returns a simple densenet made from

        example:
            dense(units=[128, 3], activation=jax.nn.silu)
    """
    layers = []
    with hk.experimental.name_scope(name):
        for idx, out in enumerate(units):
            layers.append(hk.Linear(out))
            if idx < len(units) -1:
                layers.append(activation)
    return hk.Sequential(layers, name=name)


class FlowFactory(Protocol[Input, Output]):
    def __call__(self) -> VolumeAccumulator[Input, Output]:
        ...


@pytree_dataclass(frozen=True)
class WrappedVolumeAccumulator(VolumeAccumulator[Input, Output]):
    forward: Callable[[Transformed[Input]], Transformed[Output]]
    inverse: Callable[[Transformed[Input]], Transformed[Output]]


@pytree_dataclass(frozen=True)
class MultiTransformedFlow(Generic[Input, Output]):
    pure: hk.MultiTransformed

    def with_params(self, params: dict) -> VolumeAccumulator[Input, Output]:
        forward, inverse = self.pure.apply
        return WrappedVolumeAccumulator[Input, Output](
            partial(forward, params),
            partial(inverse, params),
        )


def transform_flow(factory: FlowFactory[Input, Output]) -> MultiTransformedFlow[Input, Output]:

    def body():
        flow = factory()

        def init(input: Transformed[Input]) -> Transformed[Input]:
            return pipe2(flow.forward, flow.inverse)(input)
        
        def forward(input: Transformed[Input]) -> Transformed[Output]:
            return flow.forward(input)

        def inverse(input: Transformed[Output]) -> Transformed[Input]:
            return flow.inverse(input)

        return init, (forward, inverse)

    pure = hk.without_apply_rng(hk.multi_transform(body))

    return MultiTransformedFlow[Input, Output](pure)