# cleaned_odisseo_run.py
# Cleaned / corrected version of the simulator + likelihood + NUTS setup you provided.

from autocvd import autocvd
autocvd(num_gpus=3, interval=1)

import time
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import jax
import jax.numpy as jnp
from jax import jit
import jax.scipy as jsp

# Odisseo imports (as in your original file)
from odisseo import construct_initial_state
from odisseo.dynamics import DIRECT_ACC_MATRIX
from odisseo.option_classes import (
    SimulationConfig, SimulationParams, MNParams, NFWParams, PlummerParams, PSPParams,
    MN_POTENTIAL, NFW_POTENTIAL, PSP_POTENTIAL
)
from odisseo.initial_condition import Plummer_sphere
from odisseo.time_integration import time_integration
from odisseo.units import CodeUnits
from odisseo.utils import projection_on_GD1

# probabilistic / inference imports
import numpyro
from numpyro.infer import MCMC, NUTS

# -------------------------
# Units / config (unchanged intent)
# -------------------------
code_length = 10.0  # kpc units for code_length as before (you previously used astropy conversions)
# NOTE: in your earlier script you created CodeUnits using astropy units. I keep the CodeUnits call,
# but ensure we pass scalars in the same style you used.
import astropy.units as u
code_mass = 1e4 * u.Msun
code_time = 3 * u.Gyr
code_units = CodeUnits(code_length * u.kpc, code_mass, G=1, unit_time=code_time)  # keep your CodeUnits usage

N_particles = 1000

# center-of-mass final position / velocity in code units (kept your values, converted)
pos_com_final = jnp.array([[11.8, 0.79, 6.4]]) * u.kpc.to(code_units.code_length)
vel_com_final = jnp.array([[109.5, -254.5, -90.3]]) * (u.km / u.s).to(code_units.code_velocity)

config_sim = SimulationConfig(
    N_particles=N_particles,
    return_snapshots=False,
    num_timesteps=1000,
    external_accelerations=(NFW_POTENTIAL, MN_POTENTIAL, PSP_POTENTIAL),
    acceleration_scheme=DIRECT_ACC_MATRIX,
    softening=(0.1 * u.pc).to(code_units.code_length).value,
)

config_com = config_sim._replace(N_particles=1)

# -------------------------
# Helper: log pdf for diagonal-covariance multivariate normal
# -------------------------
def log_diag_multivariate_normal(x, mean, sigma):
    """
    Log PDF of a multivariate normal with diagonal covariance.
    x, mean, sigma are 1D arrays of same size D.
    """
    diff = (x - mean) / sigma
    D = x.shape[-1]
    log_det = 2.0 * jnp.sum(jnp.log(sigma))
    norm_const = -0.5 * (D * jnp.log(2.0 * jnp.pi) + log_det)
    exponent = -0.5 * jnp.sum(diff ** 2)
    return norm_const + exponent

# -------------------------
# Simulation wrapper
# -------------------------
# NOTE: JIT'ing high-level functions that accept Python dicts and call other non-jittable code (file I/O, etc.)
# can be problematic. We JIT only the inner numeric pieces where safe. run_simulation remains a pure JAX-style function
# insofar as it only uses JAX arrays and your odisseo functions — however if those functions are not JAX-jittable,
# remove @jit below. Keep an eye on whether time_integration and Plummer_sphere are JAX-compiled / pure.
@jit
def run_simulation(rng_key, params_dict):
    """
    rng_key : jax PRNGKey
    params_dict : dict with keys matching {'t_end','Mtot','a_Plummer','M_NFW','r_s','M_MN','a_MN'}
                  values in *linear* units expected by your SimulationParams constructor.
    Returns:
        stream : array shape (N_model, D)  # D is dimension per-particle (expected 6 here)
    """
    # Build SimulationParams (convert to code units inside, keep consistent with prior code)
    params_samples = SimulationParams(
        t_end=params_dict['t_end'] * u.Gyr.to(code_units.code_time),
        Plummer_params=PlummerParams(
            Mtot=params_dict['Mtot'] * u.Msun.to(code_units.code_mass),
            a=params_dict['a_Plummer'] * u.kpc.to(code_units.code_length),
        ),
        NFW_params=NFWParams(
            Mvir=params_dict['M_NFW'] * u.Msun.to(code_units.code_mass),
            r_s=params_dict['r_s'] * u.kpc.to(code_units.code_length),
        ),
        MN_params=MNParams(
            M=params_dict['M_MN'] * u.Msun.to(code_units.code_mass),
            a=params_dict['a_MN'] * u.kpc.to(code_units.code_length),
            b=(0.280 * u.kpc).to(code_units.code_length).value
        ),
        PSP_params=PSPParams(
            M=4501365375.06545 * u.Msun.to(code_units.code_mass),
            alpha=1.8,
            r_c=(1.9 * u.kpc).to(code_units.code_length).value
        ),
        G=code_units.G,
    )

    # integrate COM backward in time
    params_com = params_samples._replace(t_end=-params_samples.t_end)
    mass_com = jnp.array([params_samples.Plummer_params.Mtot])

    initial_state_com = construct_initial_state(pos_com_final, vel_com_final)
    final_state_com = time_integration(initial_state_com, mass_com, config=config_com, params=params_com)
    pos_com = final_state_com[:, 0]
    vel_com = final_state_com[:, 1]

    # Plummer sphere initial conditions
    positions, velocities, mass = Plummer_sphere(key=rng_key, params=params_samples, config=config_sim)

    # add COM offset
    positions = positions + pos_com
    velocities = velocities + vel_com

    initial_state_stream = construct_initial_state(positions, velocities)
    final_state = time_integration(initial_state_stream, mass, config=config_sim, params=params_samples)

    stream = projection_on_GD1(final_state, code_units=code_units)
    return stream

# -------------------------
# Stream likelihood (diagonal covariance) - vectorized and numerically stable
# -------------------------
def stream_likelihood(model_stream, obs_stream, obs_errors):
    """
    model_stream : (N_model, D)
    obs_stream   : (N_obs, D)
    obs_errors   : (D,)  per-dimension standard deviations (same for all obs)
    Returns total log-likelihood (scalar).
    Mixture with background is not included here; we treat every observation as coming from the model mixture
    by marginalizing over model points (i.e., p(obs) = mean_i N(obs | model_i, Sigma_diag)).
    """

    N_model = model_stream.shape[0]

    # Compute log p(obs | model_i) for each model point and observation.
    # We'll vectorize: for a single obs (D,), produce an array (N_model,) of log-probs.
    def log_p_for_obs(obs):
        # vmapped over model entries to compute log_diag_multivariate_normal(obs, model_i, sigma)
        per_model_log = jax.vmap(lambda m: log_diag_multivariate_normal(obs, m, obs_errors))(model_stream)
        # stable log(mean(exp(per_model_log))) = logsumexp - log(N_model)
        return jsp.special.logsumexp(per_model_log) - jnp.log(N_model)

    # map over all observations
    per_obs_logs = jax.vmap(log_p_for_obs)(obs_stream)  # shape (N_obs,)
    return jnp.sum(per_obs_logs)

# -------------------------
# Data load: set expectations for shape
# -------------------------
# In your earlier script you loaded an .npz with key 'x' and reshaped to (1,1000,6).
# For clarity, we expect obs to be shape (N_obs, D). We'll load and reshape accordingly.
true_GD1_observation_path = '/export/data/vgiusepp/odisseo_data/data_fix_position/true.npz'
_loaded = np.load(true_GD1_observation_path)
# assume 'x' is (N_total, 6)
obs_array = jnp.array(_loaded['x'][:1000])  # shape (1000, 6)  -> (N_obs, D)
# optional: theta from file (if you need truth)
true_theta = jnp.array(_loaded.get('theta', np.zeros((1000, 7)))[:1000])

# -------------------------
# Prior bounds (linear) and transforms
# -------------------------
param_names = ("t_end", "Mtot", "a_Plummer", "M_NFW", "r_s", "M_MN", "a_MN")
D = len(param_names)

prior_bounds_linear = jnp.array([
    [0.5, 5.0],                                # t_end (Gyr)
    [10 ** 3, 10 ** 4.5],                      # Plummer Mtot
    [8.0e-3 * 0.25, 8.0e-3 * 2.0],             # Plummer a (kpc)
    [4.3683325e11 * 0.25, 4.3683325e11 * 2.0], # NFW Mvir
    [16.0 * 0.25, 16.0 * 2.0],                 # NFW r_s
    [68_193_902_782.346756 * 0.25, 68_193_902_782.346756 * 2.0],  # MN M
    [3.0 * 0.25, 3.0 * 2.0]                    # MN a
])

# which params are sampled in log-space (everything except t_end)
log_mask = jnp.array([False, True, True, True, True, True, True])

# bounds in transformed (sampled) space
prior_bounds_trans = jnp.where(log_mask[:, None], jnp.log(prior_bounds_linear), prior_bounds_linear)

# helper: transform a sampled (transformed) theta to linear dict for run_simulation
def theta_log_to_dict(theta_log):
    theta_lin = jnp.where(log_mask, jnp.exp(theta_log), theta_log)
    return {k: theta_lin[i] for i, k in enumerate(param_names)}


# -------------------------
# Priors & posterior utilities
# -------------------------
def log_prior_trans(theta_log):
    """Uniform in transformed space between prior_bounds_trans."""
    in_bounds = jnp.all((theta_log >= prior_bounds_trans[:, 0]) & (theta_log <= prior_bounds_trans[:, 1]))
    # log volume (transformed space): -sum(log(interval_length))
    interval_lengths = prior_bounds_trans[:, 1] - prior_bounds_trans[:, 0]
    # numeric safety: interval_lengths must be positive
    log_vol = jnp.sum(jnp.log(interval_lengths))
    log_density = -log_vol
    return jnp.where(in_bounds, log_density, -jnp.inf)

# Note: this inner function calls run_simulation which expects a PRNG key. For reproducibility
# we use a fixed key here. If you want stochasticity inside MCMC evaluation, you must thread RNG
# differently (e.g. a seed per chain / per evaluation).
def log_likelihood_trans(theta_log):
    theta_dict = theta_log_to_dict(theta_log)
    # deterministic key (can be changed to vary per-evaluation, but be careful)
    key = jax.random.PRNGKey(0)
    model_stream = run_simulation(key, theta_dict)  # shape (N_model, D)
    # obs_errors: choose reasonable per-dimension stds (same you had previously)
    obs_errors = jnp.array([0.25, 0.001, 0.15, 5.0, 0.1, 0.0001])
    ll = stream_likelihood(model_stream=model_stream, obs_stream=obs_array, obs_errors=obs_errors)
    return ll

def log_posterior_trans(theta_log):
    lp = log_prior_trans(theta_log)
    return jnp.where(lp == -jnp.inf, -jnp.inf, lp + log_likelihood_trans(theta_log))


# potential function for NUTS (NumPyro expects potential = -log_prob)
def potential_fn(theta_log):
    # keep this Python function (do NOT jit) to avoid complications with NumPyro/NUTS JIT
    val = log_posterior_trans(theta_log)
    return -val

# -------------------------
# Initialize / run NUTS (NumPyro)
# -------------------------
# d-dim transformed prior draw
def draw_from_prior_trans(key):
    u = jax.random.uniform(key, shape=(D,))
    return prior_bounds_trans[:, 0] + u * (prior_bounds_trans[:, 1] - prior_bounds_trans[:, 0])
    

num_chains = 3
num_warmup = 5000
num_samples = 5000
target_accept = 0.8

master_key = jax.random.PRNGKey(0)
init_keys = jax.random.split(master_key, num_chains)
init_positions = jax.vmap(draw_from_prior_trans)(init_keys)  # shape (num_chains, D)

kernel = NUTS(potential_fn=potential_fn, target_accept_prob=target_accept, max_tree_depth=5)
mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, progress_bar=True)

start_time = time.time()
# For MCMC.run: pass master_key and init_params matching shape (num_chains, D)
mcmc.run(master_key, init_params=init_positions)
elapsed = time.time() - start_time
print(f"MCMC finished in {elapsed:.1f} s")

# -------------------------
# Extract & postprocess
# -------------------------
samples = mcmc.get_samples(group_by_chain=True)
# handle various shapes
if isinstance(samples, dict):
    # typical key names: 'z' if using potential, else keys map to names - we fall back to the first value
    raw = next(iter(samples.values()))
else:
    raw = samples

raw = np.array(raw)  # ensure numpy array
print("raw sample shape:", raw.shape)  # expected (num_chains, num_samples, D)

raw_flat = raw.reshape((-1, D))  # (num_chains * num_samples, D)

# convert transformed samples back to linear space
def transform_samples_log_to_lin(samples_log):
    return np.where(np.array(log_mask), np.exp(samples_log), samples_log)

samples_lin = transform_samples_log_to_lin(raw_flat)
samples_df = pd.DataFrame(samples_lin, columns=list(param_names))
print(samples_df.describe(percentiles=[0.16, 0.5, 0.84]))

# Save samples if desired
samples_df.to_csv("mcmc_samples.csv", index=False)
print("Saved samples to mcmc_samples.csv")
