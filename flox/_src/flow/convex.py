""" Implementation of convex potential flows on spheres. """

from collections.abc import Callable, Mapping
from functools import partial, wraps
from typing import (
    Any,
    Concatenate,
    ParamSpec,
    Protocol,
    TypeVar,
    cast,
    runtime_checkable,
)

import jax
import jax.numpy as jnp
from jaxopt._src.base import OptStep
from jaxopt._src.lbfgs import LBFGS
from jaxtyping import Array, Bool, Float  # type: ignore

from flox._src.geom.euclidean import det, inner, squared_norm, unit
from flox._src.geom.manifold import TangentSpaceMethod, tangent_space
from flox._src.util.func import compose, pipe

__all__ = []

Scalar = Float[Array, ""]
VectorN = Float[Array, "N"]
Matrix3x3 = Float[Array, "3 3"]
MatrixNxN = Float[Array, "N N"]
Jacobian = MatrixNxN | Float[Array, "N-1 N-1"]


Criterion = Callable[..., Scalar]


@runtime_checkable
class Solver(Protocol):
    def init_state(self, init_params, *args, **kwargs) -> Any:
        ...

    def update(self, params, state, *args, **kwargs) -> OptStep:
        ...

    def run(self, init_params, *args, **kwargs) -> OptStep:
        ...


P = ParamSpec("P")
R = TypeVar("R")


def potential(
    input: VectorN,
    ctrlpts: Float[Array, "M N"],
    weights: Float[Array, "M+1"],
    bias: Float[Array, "M+1"],
    eps: Scalar,
) -> Scalar:
    """Simple convex potential"""
    ctrlpts_ = jnp.concatenate([ctrlpts, input[None]], axis=0)
    y = jnp.log(jax.nn.softplus(bias) + jnp.cosh(ctrlpts_ @ input))
    w = jax.nn.softmax(weights) + eps * jnp.square(input).sum()
    return w @ y


def potential_gradient(
    input: VectorN,
    ctrlpts: Float[Array, "M N"],
    weights: Float[Array, "M+1"],
    bias: Float[Array, "M+1"],
    eps: Scalar,
) -> VectorN:
    return unit(jax.grad(potential)(input, ctrlpts, weights, bias, eps))


def differential(
    fn: Callable[Concatenate[VectorN, P], Scalar],
    embedded: bool = False,
    **kwargs
) -> Callable[Concatenate[VectorN, P], Jacobian]:
    """Computes the differential of fn.
    If tangent space is provided computes it on the manifold.
    If tangent space is not provided computes it in the embedding space
    """

    def eval(
        x: VectorN, *inner_args: P.args, **inner_kwargs: P.kwargs
    ) -> Jacobian:
        dF = jax.jacobian(fn, **kwargs)(x, *inner_args, **inner_kwargs)
        if embedded:
            y = fn(x, *inner_args, **inner_kwargs)
            Ty = tangent_space(y, method=TangentSpaceMethod.GramSchmidt)
            Tx = tangent_space(x, method=TangentSpaceMethod.GramSchmidt)
            dF = Ty @ dF @ jnp.transpose(Tx)
        return dF

    return eval


def numeric_inverse(
    forward: Callable[Concatenate[VectorN, P], VectorN],
    solver_factory: Callable[[Criterion], Solver] = partial(LBFGS),
    threshold: float = 1e-6,
    max_iters: int = 50,
) -> Callable[Concatenate[VectorN, P], VectorN]:
    @wraps(forward)
    def solve(init: VectorN, *args: P.args, **kwargs: P.kwargs) -> VectorN:
        def criterion(candidate: VectorN) -> Scalar:
            return squared_norm(
                forward(unit(candidate), *args, **kwargs) - init
            )

        solver = solver_factory(criterion)

        step = solver.update(init, solver.init_state(init))

        def cond(state: tuple[int, OptStep]) -> Bool[Array, ""]:
            it, step = state
            return (criterion(step.params) > threshold) & (it < max_iters)

        def body(state: tuple[int, OptStep]) -> tuple[int, OptStep]:
            it, step = state
            step = solver.update(step.params, step.state)
            return it + 1, step

        return unit(jax.lax.while_loop(cond, body, (0, step))[1].params)

    return solve


def numeric_inverse_v2(
    forward: Callable[Concatenate[VectorN, P], VectorN],
    solver_factory: Callable[[Criterion], Solver] = partial(LBFGS),
    threshold: float = 1e-5,
    max_iters: int = 20,
) -> Callable[Concatenate[VectorN, P], VectorN]:
    def criterion(
        candidate: VectorN, target: VectorN, *args: P.args, **kwargs: P.kwargs
    ) -> Scalar:
        return squared_norm(forward(unit(candidate), *args, **kwargs) - target)

    solver = solver_factory(criterion)

    @wraps(forward)
    def solve(init: VectorN, *args: P.args, **kwargs: P.kwargs) -> VectorN:

        return unit(solver.run(init, init, *args, **kwargs).params)

    return solve


def pin(
    transform: Callable[Concatenate[VectorN, P], R]
) -> Callable[Concatenate[VectorN, P], R]:
    @wraps(transform)
    def eval(x: VectorN, *args: P.args, **kwargs: P.kwargs) -> R:
        return transform(unit(x), *args, **kwargs)

    return eval


def forward_log_volume(
    transform: Callable[Concatenate[VectorN, P], VectorN]
) -> Callable[Concatenate[VectorN, P], Scalar]:
    def call(x: VectorN, *args: P.args, **kwargs: P.kwargs) -> Scalar:
        return pipe(
            differential(pin(transform), embedded=True), det, jnp.abs, jnp.log
        )(x, *args, **kwargs)

    return call


def inverse_log_volume(
    transform: Callable[Concatenate[VectorN, P], VectorN]
) -> Callable[Concatenate[VectorN, P], Scalar]:

    forward = forward_log_volume(transform)

    def forward_pass(
        x: VectorN, *args: P.args, **kwargs: P.kwargs
    ) -> tuple[Array, tuple[VectorN, tuple, Mapping]]:
        return -forward(x, *args, **kwargs), (x, args, kwargs)

    def backward_pass(res, grad):
        # F: forward transform
        # G: inverse transform

        x, args, kwargs = res

        # TODO: mypy has no option to alter ParamSpecs
        #       might be possible to revise for future versions
        def _transform(x: VectorN, *args, **kwargs) -> Scalar:
            return cast(Callable, transform)(x, *args, **kwargs)

        Jx = differential(pin(_transform))(x, *args, **kwargs)
        Jx_pinv: Array = jnp.linalg.pinv(Jx)
        dF = differential(pin(_transform), embedded=True)
        dFx = dF(x, *args)
        dFx_inv: Array = jnp.linalg.inv(dFx)
        ddFx: Array = jax.jacobian(pin(dF))(x, *args, **kwargs)

        # differential wrt input
        #                J = original jacobian
        #               PJ = projected jacobian
        #       dx inv(PJ) = inv(PJ) (dx PJ) inv(PJ) pinv(J)
        ddGx = -jnp.einsum(
            "ij, jkl, km, ln -> imn", dFx_inv, ddFx, dFx_inv, Jx_pinv
        )

        # differential wrt other args
        ddFargs = jax.jacobian(pin(dF), argnums=range(1, len(args) + 1))(
            x, *args, **kwargs
        )
        dFargs = differential(pin(_transform), argnums=range(1, len(args) + 1))(
            x, *args, **kwargs
        )

        ddGargs = []
        for ddFarg, dFarg in zip(ddFargs, dFargs):
            ddGarg = jnp.einsum("ijk, k..., jm -> im...", ddGx, dFarg, dFx)
            ddGarg += jnp.einsum("ij, jk... -> ik...", dFx_inv, ddFarg)
            ddGarg = -jnp.einsum(
                "ij..., jl -> il...",
                ddGarg,
                dFx_inv,
            )
            ddGargs.append(ddGarg)

        # reduce over gradient
        # d log det M = tr[inv(M) d M]
        return jax.tree_map(
            compose(
                partial(jnp.multiply, grad),
                partial(jnp.trace),
                partial(jnp.einsum, "ij, jk... -> ik...", dFx),
            ),
            (ddGx, *ddGargs),
        )

    @jax.custom_vjp
    @wraps(transform)
    def eval(x: VectorN, *args, **kwargs) -> Scalar:
        return forward_pass(x, *args, **kwargs)[0]

    eval.defvjp(forward_pass, backward_pass)
    return cast(Callable[Concatenate[VectorN, P], Scalar], eval)
