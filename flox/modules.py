""" This module contains haiku modules / implementations. """

from typing import Callable, TypeVar

import haiku as hk

from .flow_api import T, Transformed, VolumeAccumulator

T = TypeVar("T")


class LayerStack(hk.Module, VolumeAccumulator[T, T]):
    def __init__(
        self,
        factory: Callable[[], VolumeAccumulator[T, T]],
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
