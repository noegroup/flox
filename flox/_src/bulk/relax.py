from collections.abc import Callable
from functools import partial
from typing import Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from jaxopt._src.base import OptStep
from jaxopt._src.lbfgs import LBFGS
from jaxtyping import Array, Bool, Float  # type: ignore

__all__ = ["relax"]

Scalar = Float[Array, ""]
State = Float[Array, "N"]
Displacement = Float[Array, "N"]
Criterion = Callable[[State], Scalar]


@runtime_checkable
class Solver(Protocol):
    def init_state(self, init_params, *args, **kwargs) -> Any:
        ...

    def update(self, params, state, *args, **kwargs) -> OptStep:
        ...


def relax(
    shift: Callable[[State, Displacement], State],
    penalty: Callable[[State], Scalar],
    solver_factory: Callable[[Criterion], Solver] = partial(LBFGS),
    max_iters: int = 10,
    threshold: float = 1e-6,
) -> Callable[[State], tuple[State, Displacement]]:
    def call(x: State) -> tuple[State, Array]:
        def criterion(dx: Displacement) -> Scalar:
            return penalty(shift(x, dx))

        solver = solver_factory(criterion)
        dx = jnp.zeros_like(x)
        step = solver.update(dx, solver.init_state(dx))

        def cond(state: tuple[int, OptStep]) -> Bool[Array, ""]:
            it, step = state
            return (criterion(step.params) > threshold) & (it < max_iters)

        def body(state: tuple[int, OptStep]) -> tuple[int, OptStep]:
            it, step = state
            step = solver.update(step.params, step.state)
            return it + 1, step

        _, (dx, _) = jax.lax.while_loop(cond, body, (0, step))

        return shift(x, dx), dx

    return call
