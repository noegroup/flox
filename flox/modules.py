from typing import Callable, Generic, TypeVar, cast

import haiku as hk

from so3os.flow_api import Transform, Transformed

Input = TypeVar("Input")
Output = TypeVar("Output")


class LayerStack(hk.Module, Generic[Input, Output]):
    def __init__(
        self,
        factory: Callable[[], Transform[Input, Output]],
        num_layers=1,
        name="stacked",
    ):
        super().__init__(name=name)

        def body(x: Input | Output, reverse=False):
            trafo = factory()
            if reverse:
                x = cast(Output, x)
                return trafo.inverse(x)
            else:
                x = cast(Input, x)
                return trafo.forward(x)

        self.stack = hk.without_apply_rng(
            hk.transform(
                hk.experimental.layer_stack(
                    name="layer_stack",
                    num_layers=num_layers,
                    pass_reverse_to_layer_fn=True,
                )(body)
            )
        )

    def _apply(self, input: Input, reverse: bool) -> Transformed[Output]:
        init_rng = hk.next_rng_key() if hk.running_init() else None
        params = hk.experimental.transparent_lift(self.stack.init, allow_reuse=True)(
            init_rng, input, reverse=reverse
        )
        return self.stack.apply(params, input, reverse=reverse)

    def forward(self, input: Input) -> Transformed[Output]:
        return self._apply(input, reverse=False)

    def inverse(self, input: Input) -> Transformed[Output]:
        return self._apply(input, reverse=True)
