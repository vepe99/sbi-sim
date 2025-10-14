from autocvd import autocvd
autocvd(num_gpus = 1, interval=1)
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '5'

import numpy as np
import matplotlib.pyplot as plt
import time 
from functools import partial  # Change this line - lowercase 'partial'

import jax.numpy as jnp
import jax
from jax import jit
from jax.scipy.special import logsumexp

# jax.config.update("jax_enable_x64", True)
from jax.sharding import Mesh, PartitionSpec, NamedSharding         #IMPORT NEEDED FOR AUTOMATIC PARALLELISM
from jax.experimental import shard_map



import pandas as pd
import seaborn as sns
from chainconsumer import Chain, ChainConsumer, Truth



from odisseo import construct_initial_state
from odisseo.dynamics import  DIRECT_ACC_MATRIX, DIRECT_ACC_LAXMAP
from odisseo.option_classes import SimulationConfig, SimulationParams, MNParams, NFWParams, PlummerParams, PSPParams, MN_POTENTIAL, NFW_POTENTIAL, PSP_POTENTIAL, DIFFRAX_BACKEND, TSIT5
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
N_particles = 10_000

pos_com_final = jnp.array([[11.8, 0.79, 6.4]]) * u.kpc.to(code_units.code_length)
vel_com_final = jnp.array([[109.5,-254.5,-90.3]]) * (u.km/u.s).to(code_units.code_velocity)

config_sim = SimulationConfig(N_particles = N_particles,
                            return_snapshots = False, 
                            num_timesteps = 1000, 
                            external_accelerations=(NFW_POTENTIAL, MN_POTENTIAL, PSP_POTENTIAL), 
                            acceleration_scheme = DIRECT_ACC_MATRIX,
                            softening = (0.1 * u.pc).to(code_units.code_length).value,
                            integrator = DIFFRAX_BACKEND,
                             fixed_timestep = False,
                             diffrax_solver = TSIT5,) #default values
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
def stream_likelihood(model_stream, obs_stream):
    # Define bin edges and create meshgrids
    phi1_bins = jnp.linspace(-120, 70, 65)    # 64 bins
    phi2_bins = jnp.linspace(-8, 2, 33)       # 32 bins
    v1_bins = jnp.linspace(-2., 1.0, 65)      # 64 bins  
    v2_bins = jnp.linspace(-0.10, 0.10, 33)   # 32 bins
    R_bins = jnp.linspace(6, 20, 65)          # 64 bins
    vR_bins = jnp.linspace(-250, 250, 33)     # 32 bins


    # Create meshgrids for bin edges (not centers)
    PHI1, PHI2 = jnp.meshgrid(phi1_bins, phi2_bins, indexing='ij')
    V1, V2 = jnp.meshgrid(v1_bins, v2_bins, indexing='ij')
    R_GRID, VR_GRID = jnp.meshgrid(R_bins, vR_bins, indexing='ij')


    @jit
    def densities(stream):
        # take relevant projections from simulated stream
        x_phi = stream[:, [1,2]]   # phi1, phi2
        x_v   = stream[:, [4,5]]   # vphi1, vphi2
        x_R   = stream[:, [0,3]]   # R, v_radial

        # choose bandwidths (tune or use Silverman's rule)
        bw_phi = jnp.array([2.0, 0.5])     # example: phi1=2deg, phi2=0.5deg
        bw_v   = jnp.array([0.1, 0.01])    # example velocities
        bw_R   = jnp.array([0.5, 20.0])    # example R and vR

        # KDE densities on each meshgrid
        dens_phi = kde2d_on_grid(x_phi, PHI1, PHI2, bw_phi)
        dens_v   = kde2d_on_grid(x_v, V1, V2, bw_v)
        dens_R   = kde2d_on_grid(x_R, R_GRID, VR_GRID, bw_R)
        return dens_phi, dens_v, dens_R

    @jit
    def kde2d_on_grid(x, grid_x, grid_y, bandwidth):
        """
        Evaluate 2D Gaussian KDE on a meshgrid.

        Parameters
        ----------
        x : (N, 2) 
            Simulation data points in 2D (e.g., (phi1, phi2)).
        grid_x, grid_y : (Nx, Ny)
            Meshgrid arrays defining grid coordinates where density is evaluated.
        bandwidth : float or (2,)
            Bandwidth per dimension (std dev of Gaussian kernel).

        Returns
        -------
        dens : (Nx, Ny) 
            KDE density evaluated at grid points.
        """
        N, d = x.shape
        assert d == 2

        # Flatten grid to (G, 2)
        grid_points = jnp.stack([grid_x.ravel(), grid_y.ravel()], axis=1)  # (G,2)

        # Differences (G, N, 2)
        diff = grid_points[:, None, :] - x[None, :, :]

        # Handle bandwidth
        bw = jnp.atleast_1d(bandwidth)
        if bw.shape == (1,):
            bw = jnp.repeat(bw, 2)
        var = bw**2

        # Mahalanobis distance per dimension
        sq = (diff**2) / var  # (G,N,2)

        # log kernel for each (gridpoint, datapoint)
        logk = -0.5 * jnp.sum(sq, axis=-1) - 0.5*jnp.sum(jnp.log(2*jnp.pi*var))

        # logsumexp over datapoints
        log_dens = logsumexp(logk, axis=1) - jnp.log(N)

        dens = jnp.exp(log_dens).reshape(grid_x.shape)
        return dens

    dens_phi_target, dens_v_target, dens_R_target = densities(obs_stream)
    dens_phi, dens_v, dens_R = densities(model_stream)


    return jnp.exp(-0.1 * (
            loss_js(dens_phi_target, dens_phi) 
            +
            loss_js(dens_v_target, dens_v) 
            + 
            loss_js(dens_R_target, dens_R)
            )
            )


    
# Utility: ensure non-neg and avoid zeros
@jit
def _safe(d, eps=1e-12):
    return jnp.clip(d, a_min=eps)


@jit
def loss_js(d_target, d_sim):
    p = _safe(d_target) / jnp.sum(_safe(d_target))
    q = _safe(d_sim)   / jnp.sum(_safe(d_sim))
    m = 0.5 * (p + q)
    return 0.5 * (jnp.sum(p * (jnp.log(p) - jnp.log(m))) + jnp.sum(q * (jnp.log(q) - jnp.log(m))))


@jit
def kde2d_on_grid(x, grid_x, grid_y, bandwidth):
    """
    Evaluate 2D Gaussian KDE on a meshgrid.

    Parameters
    ----------
    x : (N, 2) 
        Simulation data points in 2D (e.g., (phi1, phi2)).
    grid_x, grid_y : (Nx, Ny)
        Meshgrid arrays defining grid coordinates where density is evaluated.
    bandwidth : float or (2,)
        Bandwidth per dimension (std dev of Gaussian kernel).

    Returns
    -------
    dens : (Nx, Ny) 
        KDE density evaluated at grid points.
    """
    N, d = x.shape
    assert d == 2

    # Flatten grid to (G, 2)
    grid_points = jnp.stack([grid_x.ravel(), grid_y.ravel()], axis=1)  # (G,2)

    # Differences (G, N, 2)
    diff = grid_points[:, None, :] - x[None, :, :]

    # Handle bandwidth
    bw = jnp.atleast_1d(bandwidth)
    if bw.shape == (1,):
        bw = jnp.repeat(bw, 2)
    var = bw**2

    # Mahalanobis distance per dimension
    sq = (diff**2) / var  # (G,N,2)

    # log kernel for each (gridpoint, datapoint)
    logk = -0.5 * jnp.sum(sq, axis=-1) - 0.5*jnp.sum(jnp.log(2*jnp.pi*var))

    # logsumexp over datapoints
    log_dens = logsumexp(logk, axis=1) - jnp.log(N)

    dens = jnp.exp(log_dens).reshape(grid_x.shape)
    return dens


true_GD1_observation_path = '/export/data/vgiusepp/odisseo_data/data_fix_position/true.npz'
observation = jnp.array(np.load(true_GD1_observation_path)['x'])
true_theta = jnp.array(np.load(true_GD1_observation_path)['theta'])

@jit
def evaluate_loglikelihood(observation, theta_1, ):

        key = jax.random.PRNGKey(0)
        # print(f"In the corrector: theta_1 shape: {theta_1.shape}, target shape: {target.shape}")
        output = run_simulation(key, theta_1)

        return stream_likelihood(model_stream=output, obs_stream=observation,)


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
    samples = {'t_end': samples[0], 'Mtot': samples[1], 'a_Plummer': samples[2], 'M_NFW': samples[3], 'r_s': samples[4], 'M_MN': samples[5], 'a_MN': samples[6]}
    return samples

@jit
def evaluate_log_prior(theta):
    """Evaluate log prior density (uniform)."""
    # Check if all parameters are within bounds 
    # theta = jnp.array([theta['t_end'], theta['Mtot'], theta['a_Plummer'], theta['M_NFW'], theta['r_s'], theta['M_MN'], theta['a_MN']])
    # in_bounds = jnp.all((theta >= prior_bounds[:, 0]) & (theta <= prior_bounds[:, 1])) 
    in_bounds = jnp.all(jnp.array([
        (theta['t_end'] >= prior_bounds[0, 0]) & (theta['t_end'] <= prior_bounds[0, 1]),
        (theta['Mtot'] >= prior_bounds[1, 0]) & (theta['Mtot'] <= prior_bounds[1, 1]),
        (theta['a_Plummer'] >= prior_bounds[2, 0]) & (theta['a_Plummer'] <= prior_bounds[2, 1]),    
        (theta['M_NFW'] >= prior_bounds[3, 0]) & (theta['M_NFW'] <= prior_bounds[3, 1]),
        (theta['r_s'] >= prior_bounds[4, 0]) & (theta['r_s'] <= prior_bounds[4, 1]),
        (theta['M_MN'] >= prior_bounds[5, 0]) & (theta['M_MN'] <= prior_bounds[5, 1]),
        (theta['a_MN'] >= prior_bounds[6, 0]) & (theta['a_MN'] <= prior_bounds[6, 1]),
    ]))
    
    # For uniform distribution: log(density) = -log(volume)
    # Volume = product of interval widths
    log_volume = jnp.sum(jnp.log(prior_bounds[:, 1] - prior_bounds[:, 0]))
    
    # Return -inf if out of bounds, -log_volume if in bounds
    return jnp.where(in_bounds, -log_volume, -jnp.inf)
    # return -log_volume


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

from blackjax.util import run_inference_algorithm

# Fix: Create a proper random step function instead of passing an array
def random_step_fn(rng_key, position):
    """Generate random step proposals for each parameter."""
    # Use different step sizes for different parameters based on their scale
    step_sizes = jnp.array([
        0.5*1,    # t_end (Gyr) - smaller steps
        0.5*3000.0,  # Mtot (Msun) - larger steps for large values
        0.5*0.0001,  # a_Plummer (kpc) - very small steps
        0.5*1e11,    # M_NFW (Msun) - large steps for large values  
        0.5*10.0,    # r_s (kpc)
        0.5*1e11,    # M_MN (Msun) - large steps for large values
        0.5*1,    # a_MN (kpc)
    ])
    
    # Generate random steps
    random_steps = jax.random.normal(rng_key, shape=(7,)) * step_sizes
    
    # Convert to dictionary format to match your position format
    step_dict = {
        't_end': random_steps[0],
        'Mtot': random_steps[1], 
        'a_Plummer': random_steps[2],
        'M_NFW': random_steps[3],
        'r_s': random_steps[4],
        'M_MN': random_steps[5],
        'a_MN': random_steps[6]
    }
    return step_dict

@partial(jit, static_argnames=['num_chains'])
def run_inference_vmap(rng_key, num_chains=2):
    rng_keys = jax.random.split(rng_key, num_chains)
    intial_positions = jax.vmap(draw_from_prior)(rng_keys)
    # print(intial_positions)
    

    def extract_single_position(idx):
        """Extract a single position dict from the vmapped result."""
        return {key: values[idx] for key, values in intial_positions.items()}
    
    run_inference_vmap = lambda i: run_inference_algorithm(
        rng_key=rng_keys[i],
        inference_algorithm=blackjax.additive_step_random_walk(
                                    logdensity_fn=evaluate_log_posterior,
                                    random_step=random_step_fn
                                ),
        num_steps=5_000,
        initial_position=extract_single_position(i),
        progress_bar=False,
    )

    return jax.vmap(run_inference_vmap)(jnp.arange(num_chains))
  
print('start running chains')
history = run_inference_vmap(jax.random.PRNGKey(42), num_chains=2)

def convert_multichain_to_dataframe_filtered(positions_dict, is_accepted_mask):
    """Convert dictionary of [num_chains, num_samples] arrays to DataFrame, filtering by acceptance."""
    
    all_samples = []
    num_chains = positions_dict['t_end'].shape[0]
    num_samples = positions_dict['t_end'].shape[1]
    
    print(f"Original samples: {num_chains} chains × {num_samples} samples = {num_chains * num_samples}")
    
    for chain_id in range(num_chains):
        # Get acceptance mask for this chain
        chain_accepted = is_accepted_mask[chain_id, :]  # Boolean mask for this chain
        
        # Extract samples for this chain
        chain_data = {}
        for param_name, values in positions_dict.items():
            # Only keep accepted samples
            chain_data[param_name] = values[chain_id, chain_accepted]
        
        # Count accepted samples for this chain
        n_accepted = jnp.sum(chain_accepted)
        print(f"  Chain {chain_id}: {n_accepted}/{num_samples} accepted ({n_accepted/num_samples*100:.1f}%)")
        
        # Create DataFrame for this chain (only accepted samples)
        if n_accepted > 0:  # Only add if there are accepted samples
            df_chain = pd.DataFrame(chain_data)
            df_chain['chain_id'] = chain_id
            df_chain['original_sample_id'] = jnp.where(chain_accepted)[0]  # Track original indices
            
            all_samples.append(df_chain)
    
    if not all_samples:
        print("⚠️ Warning: No accepted samples found!")
        return pd.DataFrame()
    
    # Combine all chains
    combined_df = pd.concat(all_samples, ignore_index=True)
    total_accepted = len(combined_df)
    total_original = num_chains * num_samples
    
    print(f"📊 Total accepted samples: {total_accepted}/{total_original} ({total_accepted/total_original*100:.1f}%)")
    
    return combined_df

# Use the filtered conversion
positions = history[1][0].position
is_accepted = history[1][1].is_accepted

# Convert with filtering
df_all_filtered = convert_multichain_to_dataframe_filtered(positions, is_accepted)
df_all_filtered.to_csv('mcmc_samples_filtered.csv', index=False)


for i in range(history[1][0].logdensity.shape[0]):
    plt.plot(history[1][0].logdensity[i], label=f'Chain {i}')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Log Density')
plt.savefig('log_density_chains.png')
plt.show()

np.save('mcmc_logdensity.npy', history[1][0].logdensity)