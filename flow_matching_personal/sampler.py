from abc import abstractmethod, ABC
from functools import partial
from typing import Union, Tuple, Optional, Dict
from jax.scipy.stats.norm import logpdf
from diffrax import diffeqsolve, ODETerm, ConstantStepSize, PIDController, SaveAt
from flax.core import FrozenDict
import flax.linen as nn
import jax.random as jr
import jax.numpy as jnp
from jax import jit

import jax
import diffrax
from diffrax import PIDController, ODETerm, diffeqsolve, ConstantStepSize

from jaxtyping import PyTree


def get_solver(solver, scan_stages=False):

    # use scan https://github.com/patrick-kidger/diffrax/issues/94 for lower compile time
    # update: scan_stages is no longer supported in diffrax; removed scan_stages argument; 
    if solver == 'tsit5':
        return diffrax.Tsit5()
    elif solver == 'dopri5':
        return diffrax.Dopri5()
    elif solver == 'euler':
        return diffrax.Euler()
    else:
        raise ValueError(f"diffrax solver \"{solver}\" not recognized")

def approx_logp_wrapper(t, y, args):
    y, _ = y
    *args, eps, func, params, rngs = args
    fn = lambda y: func(t, y, args[0], params, rngs=rngs)
    f, vjp_fn = jax.vjp(fn, y)
    (eps_dfdy,) = vjp_fn(eps)
    logp = jnp.sum(eps_dfdy * eps, axis=-1)
    return f, logp

def wrap_expansion(y):
    if isinstance(y, tuple):
        return tuple([y_[None] for y_ in y])
    else:
        return y[None]

def get_stepsize_controller(stepsize_controller_name, rtol, atol):

        if stepsize_controller_name == 'constant':
            return ConstantStepSize()
        elif stepsize_controller_name == 'pid':
            return PIDController(rtol=rtol, atol=atol)
        else:
            raise ValueError(f"stepsize controller \"{stepsize_controller_name}\" not recognized")

def get_prob_wrapper(mode):

    if mode == 'approx':
        return approx_logp_wrapper
    elif mode == 'exact':
        return exact_logp_wrapper
    elif mode == 'none':
        return no_logp_wrapper
    else:
        raise ValueError(f"log p computation mode \"{mode}\" not recognized")

def no_logp_wrapper(t, y, args):

    y, _ = y
    *args, _, func, params, rngs = args

    if isinstance(y, tuple):
        t = jnp.repeat(t, y[0].shape[0])

    else:
        t = jnp.repeat(t, y.shape[0])

    out = func(t, y, args[0], params, rngs=rngs)
    return out, 0


def exact_logp_wrapper(t, y, args):
    y, _ = y
    *args, _, func, params, rngs = args

    def exact_logp_sample(y_sample, conditioning_sample):
        fn = lambda y_: func(t[None], wrap_expansion(y_), wrap_expansion(conditioning_sample), params, rngs=rngs)[0]
        f, vjp_fn = jax.vjp(fn, y_sample)
        (size,) = y_sample.shape  # only 1D input
        eye = jnp.eye(size)
        (dfdy,) = jax.vmap(vjp_fn)(eye)
        logp = jnp.trace(dfdy)
        return f, logp

    f, logp = jax.vmap(exact_logp_sample, in_axes=0)(y, args[0])

    return f, logp

def generate_apply_rngs(rng):
    s = jr.split(rng, 5)
    rngs = {'dropout': s[1],
            'drop_path': s[2],
            'dropout_probability': s[3],
            'simulator': s[4]}
    return rngs, s[0]

def get_stepsize_controller(stepsize_controller_name, rtol, atol):

        if stepsize_controller_name == 'constant':
            return ConstantStepSize()
        elif stepsize_controller_name == 'pid':
            return PIDController(rtol=rtol, atol=atol)
        else:
            raise ValueError(f"stepsize controller \"{stepsize_controller_name}\" not recognized")
        
def get_solver(solver, scan_stages=False):

    # use scan https://github.com/patrick-kidger/diffrax/issues/94 for lower compile time
    # update: scan_stages is no longer supported in diffrax; removed scan_stages argument; 
    if solver == 'tsit5':
        return diffrax.Tsit5()
    elif solver == 'dopri5':
        return diffrax.Dopri5()
    elif solver == 'euler':
        return diffrax.Euler()
    else:
        raise ValueError(f"diffrax solver \"{solver}\" not recognized")
    
class BaseSampler(ABC):

    @abstractmethod
    def sample(self, num_samples: int, dim: int, model: nn.Module, params: PyTree, rng: jr.PRNGKey,
               z_init: Optional[jnp.ndarray], conditioning: Union[Tuple[jnp.ndarray], PyTree]) -> Tuple[
        PyTree, jr.PRNGKey]:
        pass

    @abstractmethod
    def compute_likelihood(self, x: jnp.ndarray, model: nn.Module, params: PyTree, rng: jr.PRNGKey,
                           conditioning: Union[Tuple[jnp.ndarray], PyTree]) -> Tuple[PyTree, jr.PRNGKey]:
        pass

    @abstractmethod
    def forward(self, x: jnp.ndarray, model: nn.Module, params: PyTree, rng: jr.PRNGKey,
                conditioning: Union[Tuple[jnp.ndarray], PyTree]) -> Tuple[PyTree, jr.PRNGKey]:
        pass

class ODESolver(BaseSampler):

    def __init__(self, solver_name: str = 'euler', init_stepsize: float = 0.1, num_steps: int = 10,
                 stepsize_controller_name: str = 'constant', t_0: float = 0.0, t_1: float = 1.0, rtol: float = 1e-5,
                 atol: float = 1e-5, mode: str = 'none', sigma_init: float = 1.0, sigma_rescale: float = 0.0,):

        self.solver_name = solver_name
        self.init_stepsize = init_stepsize
        self.num_steps = num_steps
        self.t_0 = t_0
        self.t_1 = t_1
        self.sigma_rescale = sigma_rescale
        self.sigma_init = sigma_init
        self.solver = get_solver(self.solver_name)
        self.prob_wrapper = get_prob_wrapper(mode)
        self.term = ODETerm(self.prob_wrapper)
        self.stepsize_controller = get_stepsize_controller(stepsize_controller_name, rtol, atol)

    def rescale(self, x: jnp.ndarray, rng: jr.PRNGKey):
        x = x + jr.normal(rng, x.shape) * self.sigma_rescale
        rng = jr.split(rng)[0]
        return x, rng

    def compute_likelihood(self, x: jnp.ndarray, model: nn.Module, params: PyTree, rng: jr.PRNGKey,
                           conditioning: Union[Tuple[jnp.ndarray], PyTree]) -> PyTree:

        return self._inference(x, model, params, rng, conditioning, backward=False)

    def get_ode_fn(self, model):

        @jit
        def ode_fn(t, y, args, params, rngs=None):
            out = model.apply(params, t, y, args, rngs=rngs, train=False)

            return out

        return ode_fn

    def _inference(self, x: jnp.ndarray, model: nn.Module, params: PyTree, rng: jr.PRNGKey,
                   conditioning: Union[Tuple[jnp.ndarray], PyTree], backward=False) -> Tuple[PyTree, jr.PRNGKey]:

        data_dict = {
            'trajectory': [],
            'features': [],
            'gradient': [],
            'nll': []
        }

        if isinstance(x, tuple):
            eps = None
        else:
            eps = jr.normal(rng, x.shape)
            rng = jr.split(rng)[0]

        ode_fn = self.get_ode_fn(model)

        rng_apply, rng = generate_apply_rngs(rng)

        if backward:

            y = (x, jnp.zeros(x.shape[0]))

            saveat = SaveAt(ts=jnp.linspace(self.t_1, self.t_0, self.num_steps))

            sol = diffeqsolve(self.term, self.solver, t0=self.t_1, t1=self.t_0,
                              dt0= - self.init_stepsize, saveat=saveat,
                              y0=y, args=(conditioning, eps, ode_fn, params, rng_apply),
                              stepsize_controller=self.stepsize_controller)

        else:

            if isinstance(x, tuple):
                prob_x = 0
            else:
                prob_x = logpdf(x).sum(axis=1)

            y = (x, prob_x)

            saveat = SaveAt(ts=jnp.linspace(self.t_0, self.t_1, self.num_steps))

            sol = diffeqsolve(self.term, self.solver, t0=self.t_0, t1=self.t_1, dt0=self.init_stepsize,
                              saveat=saveat, y0=y, args=(conditioning, eps, ode_fn, params, rng_apply),
                              stepsize_controller=self.stepsize_controller)

        x, ldj_cnf = sol.ys

        data_dict['samples'] = x[-1]

        data_dict['trajectory'] = x

        data_dict = {k: jnp.array(v) for k, v in data_dict.items()}

        return data_dict, rng
    
    def sample(self, num_samples: int, dim: int, model: nn.Module, params: PyTree, rng: jr.PRNGKey,
               z_init: Optional[jnp.ndarray], conditioning: Union[Tuple[jnp.ndarray], PyTree], ) -> Tuple[
        PyTree, jr.PRNGKey]:

        if z_init is not None:
            x_init = z_init
        else:
            x_init = self.sigma_init * jr.normal(rng, (num_samples, dim))
            rng = jr.split(rng)[0]

        return self._inference(x_init, model, params, rng, conditioning)

    def forward(self, x: jnp.ndarray, model: nn.Module, params: PyTree, rng: jr.PRNGKey,
                conditioning: Union[Tuple[jnp.ndarray], PyTree]) -> Tuple[PyTree, jr.PRNGKey]:

        return self._inference(x, model, params, rng, conditioning)