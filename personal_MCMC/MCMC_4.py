from autocvd import autocvd
autocvd(num_gpus = 1, interval=1)

import numpy as np
import matplotlib.pyplot as plt
import time 
from functools import partial  # Change this line - lowercase 'partial'

import jax.numpy as jnp
import jax
from jax import jit

# jax.config.update("jax_enable_x64", True)
from jax.sharding import Mesh, PartitionSpec, NamedSharding         #IMPORT NEEDED FOR AUTOMATIC PARALLELISM



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

code_length = 10.0 * u.kpc
code_mass = 1e4 * u.Msun
code_time = 3 * u.Gyr
code_units = CodeUnits(code_length, code_mass, G=1, unit_time = code_time )  
N_particles = 1000

pos_com_final = jnp.array([[11.8, 0.79, 6.4]]) * u.kpc.to(code_units.code_length)
vel_com_final = jnp.array([[109.5,-254.5,-90.3]]) * (u.km/u.s).to(code_units.code_velocity)

config_sim = SimulationConfig(N_particles = N_particles,
                            return_snapshots = False, 
                            num_timesteps = 1000, 
                            external_accelerations=(NFW_POTENTIAL, MN_POTENTIAL, PSP_POTENTIAL), 
                            acceleration_scheme = DIRECT_ACC_MATRIX,
                            softening = (0.1 * u.pc).to(code_units.code_length).value,) #default values
        #the center of mass needs to be integrated backwards in time first 
config_com = config_sim._replace(N_particles=1,)

def run_simulation(rng_key, params):
    params_samples = SimulationParams(t_end = params[0] * u.Gyr.to(code_units.code_time),
                        Plummer_params = PlummerParams(Mtot=params[1] * u.Msun.to(code_units.code_mass),
                                                       a = params[2] * u.kpc.to(code_units.code_length),),
                        NFW_params = NFWParams(Mvir=params[3] * u.Msun.to(code_units.code_mass),
                                               r_s= params[4] * u.kpc.to(code_units.code_length)),
                        MN_params = MNParams(M = params[5] * u.Msun.to(code_units.code_mass),
                                             a = params[6] * u.kpc.to(code_units.code_length),
                                            b = (0.280 * u.kpc).to(code_units.code_length).value),
                        PSP_params= PSPParams(M = 4501365375.06545 * u.Msun.to(code_units.code_mass),
                                                alpha = 1.8, 
                                                r_c = (1.9*u.kpc).to(code_units.code_length).value),  
                        G = code_units.G, )

    params_com = params_samples._replace(t_end=-params_samples.t_end,)
    mass_com = jnp.array([params_samples.Plummer_params.Mtot]) 
    
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
        
        # Mixture model
        # p_total = tau * jnp.exp(log_p_stream) + (1 - tau) * p_field
        p_total = jnp.exp(log_p_stream)
        return jnp.log(p_total + 1e-30)

    # Vectorize over observations
    logL_values = jax.vmap(obs_log_prob)(obs_stream, jnp.repeat(obs_errors, obs_stream.shape[0]).reshape(-1, 6))
    return jnp.sum(logL_values)


true_GD1_observation_path = '/export/data/vgiusepp/odisseo_data/data_fix_position/true.npz'
observation = jnp.array(np.load(true_GD1_observation_path)['x'][:1000]).reshape(1, 1000, 6)
true_theta = jnp.array(np.load(true_GD1_observation_path)['theta'][:1000])

@jit
def evaluate_loglikelihood(rng_key, observation, theta_1, ):

        key, sk = jax.random.split(rng_key)
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
    [0.5, 5.0],                    # t_end: 0.5 to 5 Gyr
    [10**3, 10**4.5],              # Plummer Mtot: 10^3 to 10^4.5 Msun
    [8.0 * 1e-3 * 0.25, 8.0 * 1e-3 * 2.0],  # Plummer a: 1/4 to 2x true (kpc)
    [4.3683325e11 * 0.25, 4.3683325e11 * 2.0],  # NFW Mvir: 1/4 to 2x true
    [16.0 * 0.25, 16.0 * 2.0],    # NFW r_s: 1/4 to 2x true (kpc)
    [68_193_902_782.346756 * 0.25, 68_193_902_782.346756 * 2.0],  # MN M: 1/4 to 2x true
    [3.0 * 0.25, 3.0 * 2.0],      # MN a: 1/4 to 2x true (kpc)
])  # Shape: (7, 2) for [min, max] bounds

def draw_from_prior(key):
    """Draw samples from uniform prior."""
    key, subkey = jax.random.split(key)
    u = jax.random.uniform(subkey, shape=(7,))
    
    # Transform uniform [0,1] to [min, max] for each parameter
    samples = prior_bounds[:, 0] + u * (prior_bounds[:, 1] - prior_bounds[:, 0])
    return samples

@jit
def evaluate_log_prior(theta):
    """Evaluate log prior density (uniform)."""
    # Check if all parameters are within bounds
    in_bounds = jnp.all((theta >= prior_bounds[:, 0]) & (theta <= prior_bounds[:, 1]))
    
    # For uniform distribution: log(density) = -log(volume)
    # Volume = product of interval widths
    log_volume = jnp.sum(jnp.log(prior_bounds[:, 1] - prior_bounds[:, 0]))
    
    # Return -inf if out of bounds, -log_volume if in bounds
    return jnp.where(in_bounds, -log_volume, -jnp.inf)


def draw_from_proposal(key, theta, sigma2_prop):
    key, sk = jax.random.split(key)
    theta_prime = jax.random.normal(sk, shape=theta.shape) * jnp.sqrt(sigma2_prop) + theta
    return theta_prime, key

@partial(jax.jit, static_argnames=('num_steps',))
def posterior_mcmc(observation, key, num_steps):

    # initialize theta from prior with a split key (keep `key` for the chain)
    key, subkey = jax.random.split(key)
    theta_0 = draw_from_prior(subkey)

    # Calculate rough standard deviations from your prior bounds
    prior_stds = (prior_bounds[:, 1] - prior_bounds[:, 0]) / 8  # Rough estimate

    # Use as initial proposal sigmas (different for each parameter)
    sigma2_prop_vector = (prior_stds * 0.1) ** 2  # Start with 10% of range

    def scan_step(carry, _):
        theta, key = carry

        # propose (and update key)
        theta_prime, key = draw_from_proposal(key, theta, sigma2_prop_vector)

        # split for the accept/reject uniform
        key, subkey = jax.random.split(key)

        # compute log posterior difference
        log_prior_prime = evaluate_log_prior(theta_prime)
        log_like_prime  = evaluate_loglikelihood(key, observation, theta_prime)
        log_prior       = evaluate_log_prior(theta)
        log_like        = evaluate_loglikelihood(key, observation, theta)

        # log hastings ratio
        log_r = (log_prior_prime + log_like_prime) - (log_prior + log_like)

        # accept/reject in log-space:
        # accept if log_r >= 0 OR log(u) < log_r
        logu = jnp.log(jax.random.uniform(subkey))
        accept = (log_r >= 0.0) | (logu < log_r)

        theta_new = jnp.where(accept, theta_prime, theta)

        return (theta_new, key), theta_new

    (theta_final, _), chain = jax.lax.scan(scan_step, (theta_0, key), None, length=num_steps)

    return chain

if __name__ == "__main__":
    # up until num_chains on the order of
    # the < number of cuda cores*, num_chains
    # does not negatively impact the runtime
    # of the sampling because we vmap over all chains! -
    # and on an A100 GPU thats ~7000 cores
    num_chains = 25

    # the runtime scales linearly with
    # num_steps
    # num_steps = 10000
    # burn_in = 5000

    num_steps = 20_000
    burn_in = 5_000

    # total num samples
    num_samples = num_chains * (num_steps - burn_in)
    print(f"Total number of samples: {num_samples}")

    key = jax.random.PRNGKey(4)
    # create independent keys for each chain
    keys = jax.random.split(key, num_chains)

    #SHARD THE INPUT OF THE SIMULATION
    mesh = Mesh(np.array(jax.devices()), ("n_sim",))                                                                  
    keys = jax.device_put(keys, NamedSharding(mesh, PartitionSpec("n_sim")))

    # vmap over the keys, each one runs a full chain
    # we wrap posterior_mcmc to only take key as input
    def run_chain(single_key):
        return posterior_mcmc(observation, single_key, num_steps)

    # vectorized map over chains
    all_chains = jax.vmap(run_chain)(keys)  # shape (num_chains, num_steps, theta_dim)

    all_chain = np.array(all_chains)

    # discard burn-in
    # all_chains_post = all_chains[:, burn_in:, :]  # shape (num_chains, num_steps - burn_in, theta_dim)

    # # concatenate along the chain axis
    # example_chain = all_chains_post.reshape(-1, all_chains_post.shape[-1])

    np.savez("example_chain_4.npz", all_chain=all_chain)