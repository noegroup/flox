from setuptools import find_packages, setup

setup(
    name="flox",
    version="0.1.0",
    url="https://github.com/noegroup/flox.git",
    author="Jonas Köhler",
    author_email="jonas.koehler.ks@gmail.com",
    description="flox - flows in jax - sampling focused normalizing flow",
    python_requires=">3.10",
    packages=find_packages(),
    install_requires=[
        "jax",
        "jaxlib",
        "jaxopt",
        "jax_dataclasses",
        "jaxtyping",
        "optax",
        "lenses",
        "numpyro",
        "equinox",
        "distrax",
    ],
)
