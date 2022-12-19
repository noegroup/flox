""" equivariant convolution for particle densities """

import math
from functools import lru_cache, partial
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float  # type: ignore

from flox._src.bulk.lattice import (
    lattice_indices,
    lattice_weights,
    neighbors,
    scatter,
)

__all__ = [
    "grid",
    "indices_and_weights",
    "density",
    "gradient",
    "density_features",
    "conv_nd",
    "fft_conv",
    "gauss_conv",
]


def grid(resolution: tuple[int, ...], bounds: tuple[tuple[float, float], ...]):
    """returns grid for filter"""
    xi = (
        jnp.linspace(-min, max, r)
        for ((min, max), r) in zip(bounds, resolution)
    )
    grid = jnp.meshgrid(*xi, indexing="ij")
    return cast(Array, jnp.stack(grid, axis=-1))


def indices_and_weights(pos, resolution):
    idx = jax.vmap(lattice_indices, in_axes=(0, None, None))(
        pos, neighbors(pos.shape[-1]), resolution
    )
    weights = jax.vmap(lattice_weights, in_axes=(0, 0, None))(
        pos, idx, resolution
    )
    return idx, weights


def density(idxs, weights, resolution):
    num_particles = idxs.shape[0]
    return scatter(jnp.ones((num_particles, 1)), weights, idxs, resolution)


def gradient(idx, weights, resolution):
    return discrete_gradient(density(idx, weights, resolution))


def density_features(pos, res, idxs=None, weights=None, smoothen=None):
    ndims = pos.shape[-1]
    if idxs is None:
        idxs = jax.vmap(
            partial(
                lattice_indices,
                neighbors=neighbors(ndims),
                resolution=res,
            )
        )(pos)
    if weights is None:
        weights = jax.vmap(partial(lattice_weights, resolution=res))(pos, idxs)
    d = density(idxs, weights, res)
    g = discrete_gradient(d)
    out = jnp.concatenate([d, g], axis=-1)
    if smoothen is not None:
        out = gauss_conv(out, res, smoothen)
    return out


def gradient_kernel(num_dims):
    buffer = np.zeros((3,) * num_dims + (num_dims,) + (1,))
    il = (slice(0, 1),) + (slice(1, 2),) * (num_dims - 1)
    ir = (slice(2, 3),) + (slice(1, 2),) * (num_dims - 1)
    for i in range(num_dims):
        buffer[il + (i,)] += -1
        buffer[ir + (i,)] += 1
        il = (il[-1],) + il[:-1]
        ir = (ir[-1],) + ir[:-1]
    return buffer


def conv_nd(inp, kernel, batch_dim=False, periodic=True):
    ndims = len(inp.shape) - 1 - batch_dim
    dimstr = "".join(map(str, range(ndims)))

    # add batch dimension
    if not batch_dim:
        inp = inp[None]

    dimension_numbers = jax.lax.conv_dimension_numbers(
        inp.shape,
        kernel.shape,
        (f"N{dimstr}C", f"{dimstr}OI", f"N{dimstr}C"),
    )

    if periodic:
        batch_pad = ((0, 0),)
        spatial_pad = tuple(((s - 1) // 2,) * 2 for s in kernel.shape[:-2])
        channel_pad = ((0, 0),)
        inp = jnp.pad(
            inp, pad_width=batch_pad + spatial_pad + channel_pad, mode="wrap"
        )
    out = jax.lax.conv_general_dilated(
        inp,
        kernel,
        window_strides=(1,) * ndims,
        padding="VALID",
        lhs_dilation=None,
        rhs_dilation=None,
        dimension_numbers=dimension_numbers,
    )

    if not batch_dim:
        out = out[0]
    return out


def discrete_gradient(
    signal: jnp.ndarray, batch_dim: bool = False, periodic: bool = True
):
    """discrete lattice gradient.

    Parameters:
    -----------
    signal: [B, *D]
        signal for which we compute the gradient

    Returns:
    --------
    jnp.ndarray [B, *D, D]
        lattice gradient of signal
    """
    ndims = len(signal.shape) - 1 - batch_dim
    kernel = -1.0 * gradient_kernel(ndims)
    return conv_nd(signal, kernel, batch_dim=batch_dim, periodic=periodic)


def gauss_kernel(resolution: tuple[int, ...], gain: float) -> jnp.ndarray:
    """gaussian smoothing filter.

    Parameters:
    -----------
    resolution: tuple[int, ...]
        resolution of the signal / lattice
    gain: float
        filter gain

    Returns:
    --------
    jnp.ndarray [*D]
        Gaussian kernel
    """
    g = grid(resolution, ((-1, 1),) * len(resolution))
    gauss = jnp.exp(-0.5 * gain * jnp.sum(jnp.square(g), axis=-1))
    gauss = gauss.reshape(*resolution)
    gauss = jnp.roll(
        gauss,
        shift=tuple(r // 2 for r in resolution),
        axis=range(len(resolution)),
    )
    gauss = gauss * jax.lax.rsqrt(jnp.square(gauss).sum() + 1e-12)
    return gauss


def fft_conv(
    inp: jnp.ndarray,
    kernel: jnp.ndarray,
    ndims: int,
    conserve_energy: bool = True,
    pointwise: bool = False,
):
    """spectral convolution using the FFT.

    Parameters:
    -----------
    inp: jnp.ndarray [*D, *F]
        signal on a *D - lattice
    kernel: jnp.ndarray [*S, *F] / [*S, 1]
        filter with *S spectral coeffs

    Return:
    -------
    jnp.ndarray [*D, *F]
        signal after spectral convolution with filter
    """
    nin = len(inp.shape) - ndims
    nout = len(kernel.shape) - ndims - nin
    ntot = nin + nout

    in_channels = inp.shape[-nin:]
    out_channels = kernel.shape[-(nin + nout) : -nin]

    assert kernel.shape[:-ntot] == inp.shape[:-nin]

    axes = tuple(range(ndims))
    spectrum = jnp.fft.fftn(inp, axes=axes)

    reduction_axes = tuple(range(-nin, 0))
    islices = (slice(None),) * ndims + (None,) * nout + (slice(None),) * nin
    if pointwise:
        convolved = jnp.squeeze((spectrum[islices] * kernel), axis=-2)
    else:
        convolved = (spectrum[islices] * kernel).sum(axis=reduction_axes)
    out = jnp.fft.ifftn(
        convolved,
        axes=axes,
    )
    out = cast(Array, out.real)

    if conserve_energy:
        inp_norm = jnp.sqrt(1e-12 + jnp.square(inp).sum(keepdims=True))
        out_norm = jax.lax.rsqrt(1e-12 + jnp.square(out).sum(keepdims=True))
        out = out * out_norm * inp_norm
    return out


def gauss_conv(inp, res, gain, conserve_energy: bool = True):
    return fft_conv(
        inp,
        kernel=gauss_kernel(res, gain)[..., None, None],
        ndims=len(res),
        conserve_energy=conserve_energy,
        pointwise=True,
    )


def convolution_edges(dim):
    edges = neighbors(dim)
    edges = edges / jnp.sqrt(1e-12 + jnp.square(edges).sum(-1, keepdims=True))
    return edges


def gauge_conv(
    focus: Float[Array, "F D"],
    other: Float[Array, "N F D"],
    edges: Float[Array, "N D"],
    kernel: Float[Array, "N F G"],
    order=True,
    handle_tie_break=True,
) -> Float[Array, "G D"]:
    """Gauge equivariant lattice convolution.

    F/G: features, N: neighbors, D: spatial dimensions

    Parameters:
    -----------
    focus: jnp.ndarray [F, D]
        focused cell
    other: jnp.ndarray [N, F, D]
        neighboring cells
    edges: jnp.ndarray [N, D]
        edges from focus to neighbors
    kernel: jnp.ndarray [N, F, G]
        kernel of convolution

    Return:
    -------
    jnp.ndarray [G, D]
        convolved focus
    """

    inner = jnp.einsum("fi, ni -> nf", focus, edges)
    if handle_tie_break:
        _, F, _ = other.shape
        inner_order = inner[None]
        coord_order = jnp.tile(jnp.transpose(edges)[:, :, None], (1, 1, F))
        keys = jnp.concatenate([coord_order, inner_order], axis=0)
        idxs = jnp.lexsort(keys, axis=0)
    else:
        idxs = jnp.argsort(inner, axis=0)
    if order:
        n, f, d = jnp.ogrid[tuple(map(slice, other.shape))]
        other = other[idxs[n, f], f, d]
        n, f = jnp.ogrid[tuple(map(slice, inner.shape))]
        inner = inner[idxs, f]
    inner = (inner + 1) / 9
    return jnp.einsum("nf, nfd, nfg -> gd", inner, other, kernel)
