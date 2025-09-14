from autocvd import autocvd
autocvd(num_gpus = 6, interval=1)
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'

import numpy as np
import matplotlib.pyplot as plt
import time 
from functools import partial  # Change this line - lowercase 'partial'

import jax.numpy as jnp
import jax
from jax import jit

# jax.config.update("jax_enable_x64", True)
from jax.sharding import Mesh, PartitionSpec, NamedSharding         #IMPORT NEEDED FOR AUTOMATIC PARALLELISM
from jax.experimental import shard_map



import pandas as pd
import seaborn as sns
from chainconsumer import Chain, ChainConsumer, Truth



from odisseo import construct_initial_state
from odisseo.dynamics import  DIRECT_ACC_MATRIX, DIRECT_ACC_LAXMAP
from odisseo.option_classes import SimulationConfig, SimulationParams, MNParams, NFWParams, PlummerParams, PSPParams, MN_POTENTIAL, NFW_POTENTIAL, PSP_POTENTIAL
from odisseo.initial_condition import Plummer_sphere
from odisseo.time_integration import time_integration
from odisseo.units import CodeUnits
from odisseo.utils import projection_on_GD1
from astropy import units as u
import blackjax
from tqdm import tqdm


code_length = 10.0 * u.kpc
code_mass = 1e4 * u.Msun
code_time = 3 * u.Gyr
code_units = CodeUnits(code_length, code_mass, G=1, unit_time = code_time )  
N_particles = 1000


config_sim = SimulationConfig(N_particles = N_particles,
                            return_snapshots = False, 
                            num_timesteps = 1000, 
                            external_accelerations=(NFW_POTENTIAL, MN_POTENTIAL, PSP_POTENTIAL), 
                            acceleration_scheme = DIRECT_ACC_MATRIX,
                            softening = (0.1 * u.pc).to(code_units.code_length).value,) #default values
        #the center of mass needs to be integrated backwards in time first 
config_com = config_sim._replace(N_particles=1,)

@jit
def run_simulation(rng_key, params):
    params_samples = SimulationParams(t_end = params['t_end'] * u.Gyr.to(code_units.code_time),
                        Plummer_params = PlummerParams(Mtot=params['Mtot'] * u.Msun.to(code_units.code_mass),
                                                       a = params['a_Plummer'] * u.kpc.to(code_units.code_length),),
                        NFW_params = NFWParams(Mvir=params['M_NFW'] * u.Msun.to(code_units.code_mass),
                                               r_s= params['r_s'] * u.kpc.to(code_units.code_length)),
                        MN_params = MNParams(M = params['M_MN'] * u.Msun.to(code_units.code_mass),
                                             a = params['a_MN'] * u.kpc.to(code_units.code_length),
                                            b = (0.280 * u.kpc).to(code_units.code_length).value),
                        PSP_params= PSPParams(M = 4501365375.06545 * u.Msun.to(code_units.code_mass),
                                                alpha = 1.8, 
                                                r_c = (1.9*u.kpc).to(code_units.code_length).value),  
                        G = code_units.G, )

    params_com = params_samples._replace(t_end=-params_samples.t_end,)
    mass_com = jnp.array([params_samples.Plummer_params.Mtot]) 

    pos_com_final = jnp.array([[params['x'], params['y'], params['z']]]) * u.kpc.to(code_units.code_length)
    vel_com_final = jnp.array([[params['vx'], params['vy'], params['vz']]]) * (u.km/u.s).to(code_units.code_velocity)

    
    #we construmt the initial state of the com 
    initial_state_com = construct_initial_state(pos_com_final, vel_com_final,)
    #we run the simulation backwards in time for the center of mass
    final_state_com = time_integration(initial_state_com, mass_com, config=config_com, params=params_com)
    #we calculate the final position and velocity of the center of mass
    pos_com = final_state_com[:, 0]
    vel_com = final_state_com[:, 1]

    #we construct the initial state of the Plummer sphere
    positions, velocities, mass = Plummer_sphere(key=rng_key, params=params_samples, config=config_sim)
    #we add the center of mass position and velocity to the Plummer sphere particles
    positions = positions + pos_com
    velocities = velocities + vel_com
    #initialize the initial state
    initial_state_stream = construct_initial_state(positions, velocities, )
    #run the simulation
    final_state = time_integration(initial_state_stream, mass, config=config_sim, params=params_samples)

    #projection on the GD1 stream
    stream = projection_on_GD1(final_state, code_units=code_units,)

    return stream

def log_diag_multivariate_normal(x, mean, sigma):
        """
        Log PDF of a multivariate Gaussian with diagonal covariance.
        
        Parameters
        ----------
        x : (D,)
        mean : (D,)
        sigma : (D,)  # standard deviations for each dimension
        """
        diff = (x - mean) / sigma
        D = x.shape[0]
        log_det = 2.0 * jnp.sum(jnp.log(sigma))
        norm_const = -0.5 * (D * jnp.log(2 * jnp.pi) + log_det)
        exponent = -0.5 * jnp.sum(diff**2)
        return norm_const + exponent

@jit
def stream_likelihood(model_stream, obs_stream, obs_errors, ):
    """
    Log-likelihood of observed stars given simulated stream (diagonal covariance).
    
    Parameters
    ----------
    model_stream : (N_model, D)
    obs_stream : (N_obs, D)
    obs_errors : (N_obs, D)   # per-dimension standard deviations
    tau : float
        Stream membership fraction
    p_field : float
        Background probability density
    """
    def obs_log_prob(obs, sigma):
        def model_log_prob(model_point):
            return log_diag_multivariate_normal(obs, model_point, sigma)

        # Compute log_probs for all model points
        log_probs = jax.vmap(model_log_prob)(model_stream)
        
        # Numerically stable average: log(mean(exp(log_probs)))
        log_p_stream = jax.scipy.special.logsumexp(log_probs) - jnp.log(model_stream.shape[0])
        
        return log_p_stream

    # Vectorize over observations
    logL_values = jax.vmap(obs_log_prob)(obs_stream, jnp.repeat(obs_errors, obs_stream.shape[0]).reshape(-1, 6))
    return jnp.sum(logL_values)


true_GD1_observation_path = '/export/data/vgiusepp/odisseo_data/data_fix_position/true.npz'
observation = jnp.array(np.load(true_GD1_observation_path)['x'][:1000]).reshape(1, 1000, 6)
true_theta = jnp.array(np.load(true_GD1_observation_path)['theta'][:1000])

@jit
def evaluate_loglikelihood(observation, theta_1, ):

        key = jax.random.PRNGKey(0)
        # print(f"In the corrector: theta_1 shape: {theta_1.shape}, target shape: {target.shape}")
        output = run_simulation(key, theta_1)

        return stream_likelihood(model_stream=output, obs_stream=observation, obs_errors=jnp.array([0.25, 0.001, 0.15, 5., 0.1, 0.0001]))


# True parameter values (without code_units transformation)
true_params = jnp.array([
    3.0,                    # t_end (Gyr)
    10**4.05,              # Plummer Mtot (Msun) 
    8.0 * 1e-3,            # Plummer a (kpc) - converted from pc to kpc
    4.3683325e11,          # NFW Mvir (Msun)
    16.0,                  # NFW r_s (kpc)
    68_193_902_782.346756, # MN M (Msun)
    3.0,                   # MN a (kpc)
])

# Prior bounds for each parameter
prior_bounds = jnp.array([
    [0.5, 5.0],                                                    # t_end: 0.5 to 5 Gyr
    [10**3, 10**4.5],                                             # Plummer Mtot: 10^3 to 10^4.5 Msun
    [1/4 * 0.008, 2 * 0.008],                                    # Plummer a: 1/4 to 2x true (kpc)
    [1/4 * 4.3683325e11, 2 * 4.3683325e11],  # NFW Mvir: 1/4 to 2x true
    [1/4 * 16.0, 2 * 16.0],                                      # NFW r_s: 1/4 to 2x true (kpc)
    [1/4 * 68_193_902_782.346756, 2 * 68_193_902_782.346756],  # MN M: 1/4 to 2x true
    [1/4 * 3.0, 2 * 3.0],                                        # MN a: 1/4 to 2x true (kpc)
    [10.0, 14.0],                                                 # x position (kpc)
    [0.1, 2.5],                                                   # y position (kpc)
    [6.0, 8.0],                                                   # z position (kpc)
    [90.0, 115.0],                                                # vx velocity (km/s)
    [-280.0, -230.0],                                             # vy velocity (km/s)
    [-120.0, -80.0]                                               # vz velocity (km/s)
])  # Shape: (13, 2) for [min, max] bounds

def draw_from_prior(key):
    """Draw samples from uniform prior."""
    key, subkey = jax.random.split(key)
    u = jax.random.uniform(subkey, shape=(13,))  # Changed from 7 to 13
    
    # Transform uniform [0,1] to [min, max] for each parameter
    samples = prior_bounds[:, 0] + u * (prior_bounds[:, 1] - prior_bounds[:, 0])
    samples = {
        't_end': samples[0], 'Mtot': samples[1], 'a_Plummer': samples[2], 
        'M_NFW': samples[3], 'r_s': samples[4], 'M_MN': samples[5], 'a_MN': samples[6],
        'x': samples[7], 'y': samples[8], 'z': samples[9],
        'vx': samples[10], 'vy': samples[11], 'vz': samples[12]
    }
    return samples

@jit
def evaluate_log_prior(theta):
    """Evaluate log prior density (uniform)."""
    # Check if all parameters are within bounds
    # in_bounds = jnp.all((theta >= prior_bounds[:, 0]) & (theta <= prior_bounds[:, 1]))
    
    # For uniform distribution: log(density) = -log(volume)
    # Volume = product of interval widths
    log_volume = jnp.sum(jnp.log(prior_bounds[:, 1] - prior_bounds[:, 0]))
    
    # Return -inf if out of bounds, -log_volume if in bounds
    # return jnp.where(in_bounds, -log_volume, -jnp.inf)
    return log_volume


def draw_from_proposal(key, theta, sigma2_prop):
    key, sk = jax.random.split(key)
    theta_prime = jax.random.normal(sk, shape=theta.shape) * jnp.sqrt(sigma2_prop) + theta
    return theta_prime, key

@jit
def evaluate_log_posterior_grad(observation, theta):
    """Compute gradient of log posterior."""
    
    def log_posterior(theta):
        log_prior = evaluate_log_prior(theta)
        log_like = evaluate_loglikelihood(observation, theta)
        return log_prior + log_like
    
    # Get both value and gradient
    log_post_val, grad = jax.value_and_grad(log_posterior)(theta)
    return log_post_val, grad

@jit
def evaluate_log_posterior(theta):
    log_prior = evaluate_log_prior(theta)
    log_like = evaluate_loglikelihood(theta_1 = theta, observation=observation,)
    return log_prior + log_like


import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
import pandas as pd
from numpyro.infer import MCMC, NUTS
import numpyro
import time

numpyro.set_host_device_count(1)  # set to number of local devices if you want to run chains vectorized

# ---------------- PARAMETERS / TRANSFORMS ----------------
param_names = ("t_end", "Mtot", "a_Plummer", "M_NFW", "r_s", "M_MN", "a_MN", 
               "x", "y", "z", "vx", "vy", "vz")
D = len(param_names)  # Now D = 13


# Prior bounds for each parameter
prior_bounds_linear = jnp.array([
    [0.5, 5.0],                                                    # t_end: 0.5 to 5 Gyr
    [10**3, 10**4.5],                                             # Plummer Mtot: 10^3 to 10^4.5 Msun
    [1/4 * 0.008, 2 * 0.008],                                    # Plummer a: 1/4 to 2x true (kpc)
    [1/4 * 4.3683325e11, 2 * 4.3683325e11],  # NFW Mvir: 1/4 to 2x true
    [1/4 * 16.0, 2 * 16.0],                                      # NFW r_s: 1/4 to 2x true (kpc)
    [1/4 * 68_193_902_782.346756, 2 * 68_193_902_782.346756],  # MN M: 1/4 to 2x true
    [1/4 * 3.0, 2 * 3.0],                                        # MN a: 1/4 to 2x true (kpc)
    [10.0, 14.0],                                                 # x position (kpc)
    [0.1, 2.5],                                                   # y position (kpc)
    [6.0, 8.0],                                                   # z position (kpc)
    [90.0, 115.0],                                                # vx velocity (km/s)
    [-280.0, -230.0],                                             # vy velocity (km/s)
    [-120.0, -80.0]                                               # vz velocity (km/s)
])  # Shape: (13, 2) for [min, max] bounds

# Choose which parameters we sample in log-space.
# Here: everything except t_end (index 0) is log-transformed
log_mask = jnp.array([False, True, True, True, False, True, False,  # original 7 params
                      False, False, False, False, False, False])    # 6 position/velocity params (linear space)

# Bounds in transformed space:
prior_bounds_trans = jnp.where(log_mask[:, None],
                               jnp.log(prior_bounds_linear),
                               prior_bounds_linear)  # shape (D,2)

# helper: convert theta_log (sampled) -> theta_lin for simulator (dict expected by run_simulation)
@jit
def theta_log_to_dict(theta_log):
    # theta_lin: exp for log_mask entries, identity otherwise
    theta_lin = jnp.where(log_mask, jnp.exp(theta_log), theta_log)
    # build dict in the order of param_names
    return {k: theta_lin[i] for i, k in enumerate(param_names)}

# ---------------- PRIORS & POSTERIOR ----------------
@jit
def log_prior_trans(theta_log):
    """Flat prior in transformed space: uniform in theta_log bounds.
       This corresponds to log-uniform (Jeffreys-like) prior on original parameters for log_mask=True entries.
    """
    in_bounds = jnp.all((theta_log >= prior_bounds_trans[:, 0]) & (theta_log <= prior_bounds_trans[:, 1]))
    # uniform density in transformed space: -log(volume)
    log_vol = jnp.sum(prior_bounds_trans[:, 1] - prior_bounds_trans[:, 0])
    log_density = -log_vol
    return jnp.where(in_bounds, log_density, -jnp.inf)

# Wrap your existing likelihood call:
# evaluate_loglikelihood(observation, theta_dict)  # <- already present in your code
# We will call it with theta_dict = theta_log_to_dict(theta_log)

@jit
def log_likelihood_trans(theta_log):
    theta_dict = theta_log_to_dict(theta_log)
    # You already defined evaluate_loglikelihood(observation, theta_dict)
    # Ensure evaluate_loglikelihood returns a scalar log-likelihood (float)
    return evaluate_loglikelihood(observation, theta_dict)

@jit
def log_posterior_trans(theta_log):
    lp = log_prior_trans(theta_log)
    # If prior is -inf, short-circuit
    return jax.lax.cond(lp > -jnp.inf,
                        lambda th: lp + log_likelihood_trans(th),
                        lambda th: -jnp.inf,
                        operand=theta_log)

# potential function for NUTS (NumPyro expects potential = -log_prob)
@jit
def potential_fn(theta_log):
    return -log_posterior_trans(theta_log)

# ---------------- INITIALIZATION (draw from transformed prior) ----------------
@jit
def draw_from_prior_trans(key):
    u = jax.random.uniform(key, shape=(D,))
    return prior_bounds_trans[:, 0] + u * (prior_bounds_trans[:, 1] - prior_bounds_trans[:, 0])

# ---------------- NUTS / MCMC RUN ----------------
# Tuning parameters
num_chains = 6
num_warmup = 8000
num_samples = 50000
target_accept = 0.8  # typical NUTS tuning: 0.8..0.9

master_key_int = 0
# Prepare initial positions (one per chain)
master_key = jax.random.PRNGKey(master_key_int)
init_keys = jax.random.split(master_key, num_chains)
init_positions = jax.vmap(draw_from_prior_trans)(init_keys)  # shape (num_chains, D)

# Build NUTS kernel with potential_fn
kernel = NUTS(potential_fn=potential_fn, target_accept_prob=target_accept)

mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, progress_bar=True )
start_time = time.time()

# Run MCMC. For potential_fn, pass init_params as initial positions (num_chains, D).
mcmc.run(master_key, init_params=init_positions)
elapsed = time.time() - start_time
print(f"MCMC finished in {elapsed:.1f} s")

# ---------------- EXTRACT & POSTPROCESS ----------------
samples = mcmc.get_samples(group_by_chain=True)
# When using potential_fn, samples key often is 'z' or similar; handle generically:
if isinstance(samples, dict):
    # pick the first array-like entry
    if 'z' in samples:
        raw = samples['z']  # shape (num_chains, num_samples, D)
    else:
        # fallback: first key
        raw = next(iter(samples.values()))
else:
    raw = samples  # if it's already an array

import numpy as np

samples = np.savez(f"mcmc_numpyro_positions_samples_{master_key_int}.npz", samples_log=samples)

