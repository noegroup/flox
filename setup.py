from setuptools import find_packages, setup

setup(
    name="flox",
    version="0.1.0",
    url="https://github.com/noegroup/flox.git",
    author="Jonas Köhler",
    author_email="jonas.koehler.ks@gmail.com",
    description="flox - flows in jax - sampling focused normalizing flow",
    packages=find_packages(),
    install_requires=[
        "jax >= 0.3.24",
        "jaxlib >= 0.3.24",
        "jaxopt >= 0.5.5",
        "jaxtyping >= 0.2.7",
        "optax >= 0.1.3",
        "lenses >= 1.1.0",
        "numpyro >= 0.10.1",
        "equinox",
        "distrax >= 0.1.2",
    ],
)
