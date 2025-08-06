from .base_distribution import UniformBase, LogNormalBase, GaussianBase

import jax.numpy as jnp
from math import log10

def get_two_moons_prior():
    return UniformBase(
        low=jnp.array([-1.0, -1.0]),
        high=jnp.array([1.0, 1.0]),
        shape=(2,)
    )

def get_slcp_prior():
    return UniformBase(
        low=jnp.array([-3.0, -3.0, -3.0, -3.0, -3.0]),
        high=jnp.array([3.0, 3.0, 3.0, 3.0, 3.0]),
        shape=(5,)
    )

def get_sir_prior():
    return LogNormalBase(
        mean=jnp.array([jnp.log(0.4), jnp.log(1/8)]),
        std=jnp.array([0.5, 0.2]),
        shape=(2,)
    )

def get_normal_gaussian_prior(shape):
    return GaussianBase(
        shape=shape,
        mean=jnp.zeros(shape),
        std=jnp.ones(shape),
    )

def get_lotka_volterra_prior():
    return LogNormalBase(
        mean=jnp.array([-0.125, -3, -0.125, -3]),
        std=jnp.array([0.5, 0.5, 0.5, 0.5]),
        shape=(4,)
    )

def get_odisseo_prior():

    # return UniformBase(
    #     low=jnp.array([ 0.5,
    #                     3, 
    #                     log10(1/4 * 4.3683325e11), 
    #                     log10(1/4 * 68_193_902_782.346756), ]),
    #     high=jnp.array([5, 
    #                     4.5, 
    #                     log10(2 * 4.3683325e11),
    #                     log10(2 * 68_193_902_782.346756),]),
    #     shape=(4,)
    # )
    return UniformBase(
        low = -1 * jnp.ones(4),
        high = jnp.ones(4),
        shape=(4,)
    )

def get_odisseo_prior_AllParameters():

    # return UniformBase(
    #     low=jnp.array([ 0.5,
    #                     3, 
    #                     log10(1/4 * 4.3683325e11), 
    #                     log10(1/4 * 68_193_902_782.346756), ]),
    #     high=jnp.array([5, 
    #                     4.5, 
    #                     log10(2 * 4.3683325e11),
    #                     log10(2 * 68_193_902_782.346756),]),
    #     shape=(4,)
    # )
    return UniformBase(
        low = -1 * jnp.ones(7),
        high = jnp.ones(7),
        shape=(7,)
    )