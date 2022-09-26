import inspect
from functools import singledispatch, wraps, partial

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import PyTree
import lenses
from typing import Callable, Tuple

from .geometry import Scalar
from .func_utils import compose, composed


class bijector(eqx.Module):
    forward: eqx.Module | Callable
    inverse: eqx.Module | Callable | None = None

    def __post_init__(self):
        object.__setattr__(self, "__signature__", inspect.signature(self.forward))

    def __str__(self):
        return str(self.forward)

    def __repr__(self):
        return repr(self.forward)

    @property
    def __doc__(self):
        return self.forward.__doc__

    def __call__(self, *args):
        return self.forward(*args)


@singledispatch
def invert(bij):
    raise ValueError(f"No inverse method defined for {bij}.")


@invert.register
def _(bij: bijector):
    if bij.inverse is None:
        raise ValueError(f"No inverse method defined for {bij}.")
    return bijector(bij.inverse, bij.forward)


def default_bind(fn, *args):
    @wraps(fn)
    def bound(x):
        return fn(x, *args)

    return bound


class conditional_bijector(eqx.Module):
    transform: bijector
    context: lenses.ui.base.BaseUiLens
    target: lenses.ui.base.BaseUiLens
    params: Callable[[PyTree], PyTree]
    bind: Callable = default_bind

    def __call__(self, node: PyTree) -> PyTree:
        node = lenses.bind(node)
        context = (node & self.context).get()
        params = self.params(context)
        bound = self.bind(self.transform, *params)
        target, dlogp = (node & self.target).F(bound).get()
        try:
            dlogp = sum(dlogp)
        except TypeError:
            dlogp = dlogp
        out = (node & self.target).set(target)
        return out, dlogp


class accumulate_volume(eqx.Module):
    bijector: eqx.Module

    def __call__(self, node_and_vol: Tuple[PyTree, Scalar]) -> Tuple[PyTree, Scalar]:
        node, vol = node_and_vol
        out = self.bijector(node)
        return lenses.bind(out)[1].modify(partial(jnp.add, vol))


@invert.register
def _(acc: accumulate_volume):
    return accumulate_volume(invert(acc.bijector))


def composed_bijector(*bijectors: Tuple[Callable]) -> Callable:
    return compose(*map(accumulate_volume, bijectors))


@invert.register
def _(comp: composed):
    return compose(*(invert(f) for f in reversed(list(comp.flatten()))))
