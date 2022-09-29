from collections.abc import Callable
from functools import singledispatch, wraps, partial
import inspect
from typing import Concatenate, ParamSpec

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import PyTree
import lenses

from .geometry import Scalar
from .func_utils import compose, composed, unpack_args, unpacked_args

P = ParamSpec("P")

Transform = Callable[..., tuple[PyTree, Scalar]] | eqx.Module
Bijector = Transform
AccumTransform = Callable[Concatenate[tuple, Scalar, P], tuple[PyTree, Scalar]]


class bijector(eqx.Module):
    forward: Transform
    inverse: Transform | None = None

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
    transform: Transform
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


def with_volume(
    fn: Callable[..., PyTree],
    volume: Callable[..., Scalar] | None = None,
    log_volume: bool = False,
) -> Transform:
    def zero(*args):
        return jnp.array(0.0)

    if volume is None:
        volume = zero

    @wraps(fn)
    def transform_with_volume(*args):
        out = fn(*args)
        vol = volume(*args)
        if not log_volume:
            log_vol = jnp.log(vol)
        else:
            log_vol = vol
        return out, log_vol

    return transform_with_volume


@invert.register
def _(bij: conditional_bijector):
    return lenses.bind(bij).transform.modify(invert)


class accumulate_volume(eqx.Module):
    bijector: Transform

    def __call__(self, node_and_vol: tuple[PyTree, Scalar]) -> tuple[PyTree, Scalar]:
        node, vol = node_and_vol
        out = self.bijector(node)
        return lenses.bind(out)[1].modify(partial(jnp.add, vol))


@invert.register
def _(acc: accumulate_volume):
    return accumulate_volume(invert(acc.bijector))


@invert.register
def _(comp: composed):
    return compose(*(invert(f) for f in reversed(list(comp.flatten()))))


@invert.register
def _(fn: unpacked_args):
    return unpack_args(invert(fn.fn))


def unpack_volume(lens: lenses.ui.base.BaseUiLens) -> lenses.ui.base.BaseUiLens:
    def setter(old, update):
        new, vol = update
        return (lens.set(new)(old), vol)

    return lenses.lens.Lens(lens.get(), setter)


def zoomed_transform(
    lens: lenses.ui.base.BaseUiLens, transform: Transform
) -> Transform:
    return unpack_volume(lens).modify(transform)


def zoomed_bijector(lens: lenses.ui.base.BaseUiLens, transform: Bijector) -> Bijector:
    return bijector(
        zoomed_transform(lens, transform), zoomed_transform(lens, invert(transform))
    )
