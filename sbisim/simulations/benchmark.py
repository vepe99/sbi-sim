from functools import partial

import jax.random as jr
import jax.numpy as jnp
from diffrax import PIDController, SaveAt, Dopri5, ODETerm, diffeqsolve, Euler, ConstantStepSize, Tsit5
from jax import jit

from ..strategy.distributions.base_distribution import sample_log_normal

import jax
import jax.numpy as jnp
from jax import jit, random

jax.config.update("jax_enable_x64", True)

from odisseo import construct_initial_state
from odisseo.dynamics import  DIRECT_ACC_MATRIX
from odisseo.option_classes import SimulationConfig, SimulationParams, MNParams, NFWParams, PlummerParams, PSPParams, MN_POTENTIAL, NFW_POTENTIAL, PSP_POTENTIAL
from odisseo.initial_condition import Plummer_sphere
from odisseo.time_integration import time_integration
from odisseo.units import CodeUnits
from odisseo.utils import projection_on_GD1
from astropy import units as u

class SBISimulator:

    def __init__(self):
        pass

    def __call__(self, x, num_simulations, rng):
        pass


class OdisseoSimulator(SBISimulator):

    code_length = 10.0 * u.kpc
    code_mass = 1e4 * u.Msun
    code_time = 3 * u.Gyr
    code_units = CodeUnits(code_length, code_mass, G=1, unit_time = code_time )  
    config = SimulationConfig(N_particles = 5_000,
                            return_snapshots = False, 
                            num_timesteps = 1000, 
                            external_accelerations=(NFW_POTENTIAL, MN_POTENTIAL, PSP_POTENTIAL), 
                            acceleration_scheme = DIRECT_ACC_MATRIX,
                            softening = (0.1 * u.pc).to(code_units.code_length).value,) #default values
    #the center of mass needs to be integrated backwards in time first 
    config_com = config._replace(N_particles=1,)

    pos_com_final = jnp.array([[11.8, 0.79, 6.4]]) * u.kpc.to(code_units.code_length)
    vel_com_final = jnp.array([[109.5,-254.5,-90.3]]) * (u.km/u.s).to(code_units.code_velocity)

    def run_simulation(self, 
                       rng_key, 
                       params):
        params_samples = SimulationParams(t_end = params[0],
                        Plummer_params = PlummerParams(Mtot=params[1] * u.Gyr.to(self.code_units.code_time)),
                        NFW_params = NFWParams(Mvir=params[2] * u.Msun.to(self.code_units.code_mass),
                                               r_s= (16.0 * u.kpc).to(self.code_units.code_length).value,),
                        MN_params = MNParams(M = params[3] * u.Msun.to(self.code_units.code_mass),
                                             a = (3.0 * u.kpc).to(self.code_units.code_length).value,
                                            b = (0.280 * u.kpc).to(self.code_units.code_length).value),
                        PSP_params= PSPParams(M = 4501365375.06545 * u.Msun.to(self.code_units.code_mass),
                                                alpha = 1.8, 
                                                r_c = (1.9*u.kpc).to(self.code_units.code_length).value),  
                        G = self.code_units.G, )

        params_com = params_samples._replace(t_end=-params_samples.t_end,)
        mass_com = jnp.array([params_samples.Plummer_params.Mtot]) 
        
        #we construmt the initial state of the com 
        initial_state_com = construct_initial_state(self.pos_com_final, self.vel_com_final,)
        #we run the simulation backwards in time for the center of mass
        final_state_com = time_integration(initial_state_com, mass_com, config=self.config_com, params=params_com)
        #we calculate the final position and velocity of the center of mass
        pos_com = final_state_com[:, 0]
        vel_com = final_state_com[:, 1]

        #we construct the initial state of the Plummer sphere
        positions, velocities, mass = Plummer_sphere(key=rng_key, params=params_samples, config=self.config)
        #we add the center of mass position and velocity to the Plummer sphere particles
        positions = positions + pos_com
        velocities = velocities + vel_com
        #initialize the initial state
        initial_state_stream = construct_initial_state(positions, velocities, )
        #run the simulation
        final_state = time_integration(initial_state_stream, mass, config=self.config, params=params_samples)

        #projection on the GD1 stream
        stream = projection_on_GD1(final_state, code_units=self.code_units,)

        return stream
    
    def add_noise(x,rng, noise_std=jnp.array([0.25, 0.001, 0.15, 5., 0.1, 1e-3])):
        noise_std = noise_std.to(x.device)
        x_noise = jax.random.multivariate_normal(mean=x, cov=noise_std, key=rng, shape=(x.shape[0]))
        return x_noise
    
    def transform_in_histogram(x, bins=[64, 32]):
        ph1_phi2, _, _ = jnp.histogram2d(x[:, 1], x[:, 2], bins = bins, range = [[-120., 70.], [-8, 2]] )
        R_vR, _, _ = jnp.histogram2d(x[:, 0], x[:, 3], bins = bins, range = [[6., 20.], [-250., 250.]] )
        vphicosphi2_vphi2, _, _ = jnp.histogram2d(x[:, 4], x[:, 5], bins = bins, range = [[-2., 1.], [-0.1, 0.1 ]] )
        return jnp.stack([ph1_phi2, R_vR, vphicosphi2_vphi2], axis=0)


    def __init__(self, ):
        super().__init__()

        self.code_units = self.code_units
        self.config = self.config._replace(code_units=self.code_units)
        self.config_com = self.config_com._replace(code_units=self.code_units)
        #this is the final position of the cluster, we need to integrate backwards in time 
        self.pos_com_final = self.pos_com_final
        self.vel_com_final = self.vel_com_final
   
        
    def __call__(self, params,  num_simulations, rng, 
                 normalize=True, deterministic=False, histogram=True):
            
        if normalize:
            #not sure about this
            params = params * self.std_X + self.mean_X

        batch_size = params.shape[0]

        X = jnp.repeat(params, batch_size)
        Y = self.run_simulation(rng, params)
        Y = jnp.repeat(Y, batch_size)

        if deterministic:
            pass
        else:
            Y =  self.add_noise(x=Y, rng=rng)

        if normalize:
            # Normalize the streams if required
            Y = (Y - jnp.mean(Y, axis=0)) / jnp.std(Y, axis=0)
        
        if histogram:
            Y = self.transform_in_histogram(Y, bins=[64, 32])
        
        X = jnp.repeat(X[:, :, None], num_simulations, axis=2)
        Y = jnp.repeat(Y[:, :, None], num_simulations )


        samples_x = X
        samples_y = Y

        samples = jnp.stack([samples_x, samples_y], axis=1)
        samples = samples.reshape(batch_size, -1)

        return samples, rng


class LotkaVolterraSimulator(SBISimulator):

    # dataset mean and std for the dataset generated with julia (lotka_volterra)
    mean_X_julia = jnp.array([0.9990237, 0.05643666, 1.0011334, 0.05642604])
    std_X_julia = jnp.array([0.52994794, 0.03009129, 0.53767025, 0.02997164])

    mean_Y_julia = jnp.array([30.141129, 46.83427, 8.7286, 11.021725, 21.564829,
                        26.050062, 24.92791, 23.482784, 22.80601, 22.81803,
                        1.0052602, 81.25922, 33.069664, 11.751602, 12.766676,
                        19.309868, 22.869844, 23.83543, 23.517775, 23.575367])

    std_Y_julia = jnp.array([3.0155041, 60.28853, 29.467123, 30.484344, 42.12618,
                       44.28906, 42.485302, 41.751675, 41.63403, 42.180115,
                       0.10088786, 92.0542, 35.28452, 24.845387, 39.192966,
                       48.260002, 51.37197, 52.59593, 50.4334, 50.454536])

    # dataset mean and std for the dataset generated with this solver (jax) (lotka_volterra_jax)
    mean_X_jax = jnp.array([1.0014604 , 0.0564126 , 1.0018734 , 0.05641408])
    std_X_jax = jnp.array([0.5388547 , 0.03008566, 0.53690755, 0.02992921])

    mean_Y_jax = jnp.array([30.141914 , 47.047108 ,  9.214024 , 11.343965 , 21.914902 ,
                      26.307745 , 25.160164 , 23.45332  , 23.16009  , 22.946615 ,
                      1.0055969, 81.01535  , 33.090443 , 11.814855 , 12.870098 ,
                      19.140581 , 23.004269 , 23.908941 , 23.642736 , 23.511082 ])

    std_Y_jax = jnp.array([3.0218716 , 60.507206 , 29.593065  , 30.618069, 42.391228,
                    44.15628, 42.36046, 41.378613 , 41.630035  , 41.85586,
                    0.10115808, 91.81441, 35.12128 , 25.497183  , 40.145065,
                    47.37003, 51.53941, 50.84498 , 50.40996 , 51.215176])

    def __init__(self, num_samples=10, stepsize='constant', type='julia'):

        super().__init__()

        self.stepsize = stepsize
        self.num_samples = num_samples

        if type=='julia':
            self.mean_X = self.mean_X_julia
            self.std_X = self.std_X_julia
            self.mean_Y = self.mean_Y_julia
            self.std_Y = self.std_Y_julia
        elif type=='jax':
            self.mean_X = self.mean_X_jax
            self.std_X = self.std_X_jax
            self.mean_Y = self.mean_Y_jax
            self.std_Y = self.std_Y_jax
        else:
            raise ValueError('type should be either julia or jax')


    @partial(jit, static_argnums=(0, 2, 5, 6))
    def __call__(self, params, num_simulations, rng, stepsize=0.2,
                 normalize=True, deterministic=False):

        if normalize:
            params = params * self.std_X + self.mean_X

        batch_size = params.shape[0]

        X = jnp.array([30.0])
        Y = jnp.array([1.0])

        X = jnp.repeat(X, batch_size)
        Y = jnp.repeat(Y, batch_size)

        T = 20

        def vector_field(t, y, args):
            a = args[:, 0]
            b = args[:, 1]
            c = args[:, 2]
            d = args[:, 3]
            x1, y1 = y
            e = jnp.array([jnp.multiply(a, x1) - jnp.multiply(b, jnp.multiply(x1, y1)),
                           - jnp.multiply(c, y1) + jnp.multiply(d, jnp.multiply(x1, y1))])

            return e

        term = ODETerm(vector_field)

        # solver = Euler()
        # solver = Dopri5()
        solver = Tsit5()

        # saveat_list = list(range(1, T+1))[::2]
        # saveat_list = list(jnp.linspace(0, 20, int(20 / 0.1) + 1))
        # saveat_list = list(jnp.linspace(0, T - 1, self.num_samples))

        saveat_list = list(jnp.linspace(0, T, int(T / 0.1) + 1))[::(T + 1)]

        saveat = SaveAt(ts=saveat_list)

        if self.stepsize == 'constant':
            stepsize_controller = ConstantStepSize()
        else:
            stepsize_controller = PIDController(rtol=1e-8, atol=1e-8)

        sol = diffeqsolve(term, solver, t0=0.0, t1=T, dt0=stepsize, y0=jnp.array([X, Y]), args=params,
                          saveat=saveat, stepsize_controller=stepsize_controller)

        X = sol.ys[:, 0].T
        Y = sol.ys[:, 1].T

        X = jnp.repeat(X[:, :, None], num_simulations, axis=2)
        Y = jnp.repeat(Y[:, :, None], num_simulations, axis=2)

        if deterministic:
            sample_x = X.clip(1e-10, 10000.0)
            sample_y = Y.clip(1e-10, 10000.0)
        else:
            sample_x, rng = sample_log_normal(rng, mean=jnp.log(X).clip(1e-10, 10000.0), std=0.1)
            sample_y, rng = sample_log_normal(rng, mean=jnp.log(Y).clip(1e-10, 10000.0), std=0.1)

        sample_x = jnp.mean(sample_x, axis=2, keepdims=True)
        sample_y = jnp.mean(sample_y, axis=2, keepdims=True)

        sample_x = jnp.transpose(sample_x, (0, 2, 1))
        sample_y = jnp.transpose(sample_y, (0, 2, 1))

        if normalize:
            sample_x = (sample_x - self.mean_Y[:self.num_samples]) / self.std_Y[:self.num_samples]
            sample_y = ((sample_y - self.mean_Y[self.num_samples:2 * self.num_samples]) /
                        self.std_Y[self.num_samples:2 * self.num_samples])

        samples = jnp.stack([sample_x, sample_y], axis=1)

        samples = samples.reshape(batch_size, -1)

        return samples, rng

class TwoMoonsSimulator(SBISimulator):

    mean_X = jnp.array([-0.00143059, -0.0003702])
    std_X = jnp.array([0.5778799, 0.57644737])

    mean_Y = jnp.array([-0.15700957, 0.00028384])

    std_Y = jnp.array([0.33428973, 0.5816402])

    def __init__(self, num_samples=10):

        super().__init__()

        self.num_samples = num_samples

    def __call__(self, x, num_simulations, rng, normalize=True, deterministic=True):

        if normalize:
            x = x * self.std_X + self.mean_X

        batch_size = x.shape[0]
        alpha = jr.uniform(rng, (batch_size * num_simulations),
                           minval=-jnp.pi / 2, maxval=jnp.pi / 2)

        rng = jr.split(rng)[0]

        r = jr.normal(rng, (batch_size * num_simulations)) * 0.01 + 0.1

        rng = jr.split(rng)[0]

        m_1 = - jnp.abs(x[:, 0] + x[:, 1]) / jnp.sqrt(2)
        m_2 = (-x[:, 0] + x[:, 1]) / jnp.sqrt(2)

        x_1 = r * jnp.cos(alpha) + 0.25
        x_2 = r * jnp.sin(alpha)

        m_1 = jnp.repeat(m_1, num_simulations)
        m_2 = jnp.repeat(m_2, num_simulations)

        s_1 = m_1 + x_1
        s_2 = m_2 + x_2

        s_1 = s_1.reshape(batch_size, num_simulations)
        s_2 = s_2.reshape(batch_size, num_simulations)

        s_1 = jnp.mean(s_1, axis=1)
        s_2 = jnp.mean(s_2, axis=1)

        s = jnp.stack([s_1, s_2], axis=1)

        if normalize:
            s = (s - self.mean_Y) / self.std_Y

        return s, rng

class SLCPSimulator(SBISimulator):

    mean_X = jnp.array([-0.00352043, 0.00952015, -0.00882017, 0.00077796, -0.00117648])
    std_X = jnp.array([1.7317334, 1.7290698, 1.7304472, 1.729663, 1.7304978])

    mean_Y = jnp.array(
        [-0.02332785, -0.00365681, 0.01568362, 0.01189906, -0.00670727, 0.01893608, 0.00081571, -0.00768246])

    std_Y = jnp.array([4.369495, 4.36282, 4.4007354, 4.3710537, 4.3577127, 4.3721476, 4.375734, 4.355193])

    def __init__(self, num_samples=10):

        super().__init__()

        self.num_samples = num_samples

    def __call__(self, theta, num_simulations, rng, normalize=True, deterministic=True):

        if normalize:
            theta = theta * self.std_X + self.mean_X

        batch_size = theta.shape[0]

        s1 = theta[:, 2] ** 2
        s2 = theta[:, 3] ** 2
        rho = jnp.tanh(theta[:, 4])

        m = theta[:, :2]

        cov_11 = s1 ** 2
        cov_22 = s2 ** 2
        cov_12 = s1 * s2 * rho
        cov_21 = cov_12

        cov = jnp.stack([jnp.stack([cov_11, cov_12], axis=1),
                         jnp.stack([cov_21, cov_22], axis=1)], axis=1)

        m = jnp.repeat(m, 4 * num_simulations, axis=0)
        cov = jnp.repeat(cov, 4 * num_simulations, axis=0)

        x = jr.multivariate_normal(rng, m, cov, shape=(batch_size * 4 * num_simulations,))
        rng = jr.split(rng)[0]

        x = jnp.reshape(x, (batch_size, num_simulations, 4, 2))

        x = jnp.mean(x, axis=1)

        x = jnp.reshape(x, (batch_size, 8))

        if normalize:
            x = (x - self.mean_Y) / self.std_Y

        return x, rng

class SIRSimulator(SBISimulator):

    mean_X = jnp.array([0.45238078, 0.12739643])
    std_X = jnp.array([0.24206053, 0.02570804])

    mean_Y = jnp.array([1.09999999e-03, 3.48769493e+01, 1.09684425e+02,
                        9.79240799e+01, 6.52471771e+01, 4.02428398e+01,
                        2.48261585e+01, 1.55915995e+01,
                        1.01330795e+01, 6.78710985e+00])

    std_Y = jnp.array([3.31479982e-02, 1.20636147e+02, 1.55277420e+02,
                       1.16854355e+02, 8.65967255e+01, 6.38980637e+01,
                       4.75755882e+01, 3.53401566e+01, 2.66629333e+01,
                       2.03939667e+01])

    def __init__(self, num_samples=10):

        super().__init__()

        self.num_samples = num_samples

    def __call__(self, params, num_simulations, rng, stepsize=5.0, normalize=True,
                 deterministic=True):

        if normalize:
            params = params * self.std_X + self.mean_X

        batch_size = params.shape[0]

        N = 1000000
        S = jnp.array([N - 1]).repeat(batch_size)
        I = jnp.array([1.0]).repeat(batch_size)
        R = jnp.array([0.0]).repeat(batch_size)

        T = 160

        def vector_field(t, y, args):
            beta = args[:, 0]
            gamma = args[:, 1]

            s, i, r = y

            e = jnp.array([- beta * s * i / N, beta * s * i / N - gamma * i, gamma * i])

            return e

        term = ODETerm(vector_field)

        solver = Tsit5()

        saveat_list = list(jnp.linspace(0, T, int(T / 1) + 1))[::17]

        saveat = SaveAt(ts=saveat_list)

        # stepsize_controller = ConstantStepSize()
        stepsize_controller = PIDController(rtol=1e-8, atol=1e-8)

        sol = diffeqsolve(term, solver, t0=0.0, t1=T, dt0=stepsize, y0=jnp.array([S, I, R]), args=params,
                          saveat=saveat, stepsize_controller=stepsize_controller)

        S = sol.ys[:, 0].T
        I = sol.ys[:, 1].T
        R = sol.ys[:, 2].T

        I = jnp.repeat(I[:, :, None], num_simulations, axis=2)
        prob = I / N

        if deterministic:
            sample = 1000 * prob
        else:
            sample = jr.binomial(rng, 1000, prob)
            rng = jr.split(rng)[0]

        sample = jnp.mean(sample, axis=2, keepdims=True)

        sample = jnp.transpose(sample, (0, 2, 1))

        sample = sample.reshape(batch_size, -1)

        if normalize:
            sample = (sample - self.mean_Y[:self.num_samples]) / self.std_Y[:self.num_samples]

        return sample, rng