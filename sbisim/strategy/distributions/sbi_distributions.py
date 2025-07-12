from .base_distribution import UniformBase, LogNormalBase, GaussianBase
from odisseo.option_classes import SimulationParams, MNParams, NFWParams, PlummerParams, PSPParams
from odisseo.units import CodeUnits
from astropy import units as u

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

    code_length = 10.0 * u.kpc
    code_mass = 1e4 * u.Msun
    code_time = 3 * u.Gyr
    code_units = CodeUnits(code_length, code_mass, G=1, unit_time = code_time )  

    params = SimulationParams(t_end = (3 * u.Gyr).to(code_units.code_time).value,  
                            Plummer_params= PlummerParams(Mtot=(10**4.05 * u.Msun).to(code_units.code_mass).value,
                                                            a=(8 * u.pc).to(code_units.code_length).value),
                            MN_params= MNParams(M = (68_193_902_782.346756 * u.Msun).to(code_units.code_mass).value,
                                                a = (3.0 * u.kpc).to(code_units.code_length).value,
                                                b = (0.280 * u.kpc).to(code_units.code_length).value),
                            NFW_params= NFWParams(Mvir=(4.3683325e11 * u.Msun).to(code_units.code_mass).value,
                                                r_s= (16.0 * u.kpc).to(code_units.code_length).value,),      
                            PSP_params= PSPParams(M = 4501365375.06545 * u.Msun.to(code_units.code_mass),
                                                    alpha = 1.8, 
                                                    r_c = (1.9*u.kpc).to(code_units.code_length).value),                    
                            G=code_units.G, ) 
    return UniformBase(
        low=jnp.array([ 0.5,
                        3, 
                        1/4 * log10(4.3683325e11), 
                        1/4 * log10(68_193_902_782.346756), ]),
        high=jnp.array([5, 
                        4.5, 
                        2 * log10(4.3683325e11),
                        2 * log10(68_193_902_782.346756),]),
        shape=(4,)
    )