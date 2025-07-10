import flax.linen as nn
from jax import vmap, value_and_grad
import jax
import jax.random as jr

from ...simulations import LotkaVolterraSimulator, SBISimulator, OdisseoSimulator
from ...utils import instantiate_from_config
import jax.numpy as jnp

from jax.lax import stop_gradient

class CorrectorSimulator(nn.Module):

    model: nn.Module
    simulator: dict
    controlled_flow: dict
    aggregation: dict
    freeze: bool = True
    layer_norm: bool = False
    start_time: float = 1.0
    num_simulations: int = 1
    clip_output: float = 10.0

    def forward_flow(self, t, theta, context, train=False):
        return self.model(t, theta, context, train=train)

    def setup(self):

        self.simulator_impl: SBISimulator = instantiate_from_config(self.simulator)
        self.controlled_flow_impl: nn.Module = instantiate_from_config(self.controlled_flow)
        self.aggregration_impl: nn.Module = instantiate_from_config(self.aggregation)

    def __call__(self, t, theta, context, train=True):

        return self.forward(t, theta, context, train=train)[0]

    def forward(self, t, theta, context, train=True):

        # predict flow
        flow_pred = self.model(t, theta, context, train=train)

        if self.freeze:
            flow_pred = stop_gradient(flow_pred)

        theta_1 = theta + jnp.einsum('ab,a->ab', flow_pred, 1 - t[:, 0])

        simulator_rng = self.make_rng('simulator')
        output, _ = self.simulator_impl(theta_1, num_simulations=self.num_simulations,
                                     rng = simulator_rng, deterministic=True) # noqa

        output = output * self.simulator_impl.std_Y + self.simulator_impl.mean_Y
        context = context * self.simulator_impl.std_Y + self.simulator_impl.mean_Y

        output = jnp.log(output)
        context = jnp.log(context)

        output = output - context

        output = jnp.nan_to_num(output).clip(-self.clip_output, self.clip_output)

        output = self.aggregration_impl(output)

        output = jnp.concatenate([flow_pred, t, output], axis=1)

        drift = flow_pred + self.controlled_flow_impl(output, context=None)  # noqa

        drift = (jnp.einsum('ab, a -> ab', drift, t[:, 0] > self.start_time) +
                 jnp.einsum('ab, a -> ab', flow_pred, t[:, 0] <= self.start_time))

        return drift, output


class CorrectorMock(nn.Module):
    """
    Corrector model for experiments: instead of using a simulator, just output 0, see if the improvements improve just from finetuning
    """

    model: nn.Module
    simulator: dict
    controlled_flow: dict
    aggregation: dict
    freeze: bool = True
    layer_norm: bool = False
    start_time: float = 1.0
    num_simulations: int = 1
    clip_output: float = 2.0

    def forward_flow(self, t, theta, context, train=False):
        return self.model(t, theta, context, train=train)

    def setup(self):

        self.simulator_impl: SBISimulator = instantiate_from_config(self.simulator)
        self.controlled_flow_impl: nn.Module = instantiate_from_config(self.controlled_flow)
        self.aggregration_impl: nn.Module = instantiate_from_config(self.aggregation)

    def __call__(self, t, theta, context, train=True):

        return self.forward(t, theta, context, train=train)[0]

    def forward(self, t, theta, context, train=True):

        # predict flow
        flow_pred = self.model(t, theta, context, train=train)

        if self.freeze:
            flow_pred = stop_gradient(flow_pred)

        output = jnp.zeros_like(flow_pred)

        output = jnp.concatenate([flow_pred, t, output], axis=1)

        drift = flow_pred + self.controlled_flow_impl(output, context=None)  # noqa

        drift = (jnp.einsum('ab, a -> ab', drift, t[:, 0] > self.start_time) +
                 jnp.einsum('ab, a -> ab', flow_pred, t[:, 0] <= self.start_time))

        return drift, output

class CorrectorDifferentiableSimulator(nn.Module):

    model: nn.Module
    simulator: dict
    controlled_flow: dict
    aggregation: dict
    freeze: bool = True
    layer_norm: bool = False
    start_time: float = 1.0
    num_simulations: int = 1
    clip_output: float = 10.0

    def setup(self):

        self.simulator_impl: SBISimulator = instantiate_from_config(self.simulator)
        self.controlled_flow_impl: nn.Module = instantiate_from_config(self.controlled_flow)
        self.aggregration_impl: nn.Module = instantiate_from_config(self.aggregation)

    def __call__(self, t, theta, context, train=True):

        return self.forward(t, theta, context, train=train)[0]

    def l2_distance(self, theta_1, target):

        simulator_rng = self.make_rng('simulator')
        output, _ = self.simulator_impl(theta_1[None], num_simulations=self.num_simulations,
                                        rng=simulator_rng, deterministic=False)  # noqa

        return jnp.mean(jnp.square(target - output[0]))

    def forward_flow(self, t, theta, context, train=False):
        # we need this because self.model was trained with stacked flow which has additional time dimensions

        return self.model(t, theta, context, train=train)

    def forward(self, t, theta, context, train=True):

        # predict flow
        flow_pred = self.model(t, theta, context, train=train)

        if self.freeze:
            flow_pred = stop_gradient(flow_pred)

        theta_1 = theta + jnp.einsum('ab,a->ab', flow_pred, 1 - t[:, 0])

        grad_fn = vmap(value_and_grad(self.l2_distance), in_axes=0)
        loss, grad = grad_fn(theta_1, context)

        loss = jnp.expand_dims(loss, axis=1)

        output = jnp.concatenate([loss, grad], axis=1)
        output = jnp.nan_to_num(output).clip(-self.clip_output, self.clip_output)

        output = jnp.concatenate([flow_pred, t, output], axis=1)
        drift = flow_pred + self.controlled_flow_impl(output, context=None) # noqa

        drift = (jnp.einsum('ab, a -> ab', drift, t[:, 0] > self.start_time) +
                 jnp.einsum('ab, a -> ab', flow_pred, t[:, 0] <= self.start_time))

        return drift, output

class CorrectorDifferentiableSimulatorLV(nn.Module):
    """
    Corrector model for Lotka-Volterra simulator, differentiable version.
    """

    model: nn.Module
    simulator: dict
    controlled_flow: dict
    aggregation: dict
    freeze: bool = True
    layer_norm: bool = False
    start_time: float = 1.0
    num_simulations: int = 1
    clip_output: float = 10.0

    def setup(self):

        self.simulator_impl: LotkaVolterraSimulator = instantiate_from_config(self.simulator)
        self.controlled_flow_impl: nn.Module = instantiate_from_config(self.controlled_flow)
        self.aggregration_impl: nn.Module = instantiate_from_config(self.aggregation)

    def __call__(self, t, theta, context, train=True):

        return self.forward(t, theta, context, train=train)[0]

    def l2_distance(self, theta_1, target):

        simulator_rng = self.make_rng('simulator')
        output, _ = self.simulator_impl(theta_1[None], num_simulations=self.num_simulations,
                                        rng=simulator_rng, deterministic=True)  # noqa

        output = output[0]

        output = output * self.simulator_impl.std_Y + self.simulator_impl.mean_Y
        target = target * self.simulator_impl.std_Y + self.simulator_impl.mean_Y

        output = jnp.log(output)
        target = jnp.log(target)

        return jnp.mean(jnp.square(target - output))

    def forward_flow(self, t, theta, context, train=False):

        # we need this because self.model was trained with stacked flow which has additional time dimensions
        return self.model(t, theta, context, train=train)

    def forward(self, t, theta, context, train=True):

        # predict flow
        flow_pred = self.model(t, theta, context, train=train)

        if self.freeze:
            flow_pred = stop_gradient(flow_pred)

        theta_1 = theta + jnp.einsum('ab,a->ab', flow_pred, 1 - t[:, 0])

        grad_fn = vmap(value_and_grad(self.l2_distance), in_axes=0)
        loss, grad = grad_fn(theta_1, context)

        loss = jnp.expand_dims(loss, axis=1)

        output = jnp.concatenate([loss, grad], axis=1)
        output = jnp.nan_to_num(output).clip(-self.clip_output, self.clip_output)

        output = jnp.concatenate([flow_pred, t, output], axis=1)
        drift = flow_pred + self.controlled_flow_impl(output, context=None) # noqa

        drift = (jnp.einsum('ab, a -> ab', drift, t[:, 0] > self.start_time) +
                 jnp.einsum('ab, a -> ab', flow_pred, t[:, 0] <= self.start_time))

        return drift, output
    

class CorrectorDifferentiableSimulatorOdisseo(nn.Module):
    """
    Corrector model for Lotka-Volterra simulator, differentiable version.
    """

    model: nn.Module
    simulator: dict
    controlled_flow: dict
    aggregation: dict
    freeze: bool = True
    layer_norm: bool = False
    start_time: float = 1.0
    num_simulations: int = 1
    clip_output: float = 10.0

    def setup(self):

        self.simulator_impl: OdisseoSimulator = instantiate_from_config(self.simulator)
        self.controlled_flow_impl: nn.Module = instantiate_from_config(self.controlled_flow)
        self.aggregration_impl: nn.Module = instantiate_from_config(self.aggregation)

    def __call__(self, t, theta, context, train=True):

        return self.forward(t, theta, context, train=train)[0]

    def mmd(self, theta_1, target):

        simulator_rng = self.make_rng('simulator')
        output, _ = self.simulator_impl(theta_1[None], num_simulations=self.num_simulations,
                                        rng=simulator_rng, deterministic=True)  # noqa

        output = output[0]

        output = output * self.simulator_impl.std_Y + self.simulator_impl.mean_Y
        target = target * self.simulator_impl.std_Y + self.simulator_impl.mean_Y
        
        return percintile_based_mmd(output, target) 
        

    def forward_flow(self, t, theta, context, train=False):

        # we need this because self.model was trained with stacked flow which has additional time dimensions
        return self.model(t, theta, context, train=train)

    def forward(self, t, theta, context, train=True):

        # predict flow
        flow_pred = self.model(t, theta, context, train=train)

        if self.freeze:
            flow_pred = stop_gradient(flow_pred)

        theta_1 = theta + jnp.einsum('ab,a->ab', flow_pred, 1 - t[:, 0])

        grad_fn = vmap(value_and_grad(self.mmd), in_axes=0)
        loss, grad = grad_fn(theta_1, context)

        loss = jnp.expand_dims(loss, axis=1)

        output = jnp.concatenate([loss, grad], axis=1)
        output = jnp.nan_to_num(output).clip(-self.clip_output, self.clip_output)

        output = jnp.concatenate([flow_pred, t, output], axis=1)
        drift = flow_pred + self.controlled_flow_impl(output, context=None) # noqa

        drift = (jnp.einsum('ab, a -> ab', drift, t[:, 0] > self.start_time) +
                 jnp.einsum('ab, a -> ab', flow_pred, t[:, 0] <= self.start_time))

        return drift, output


def rbf_kernel(x, y, sigma):
    """RBF kernel optimized for 6D astronomical data"""
    return jnp.exp(-jnp.sum((x - y)**2) / (2 * sigma**2))

def compute_mmd(sim_norm, target_norm, sigmas):
    xx = jnp.mean(jax.vmap(lambda xi: jax.vmap(lambda xj: rbf_kernel(xi, xj, sigmas))(sim_norm))(sim_norm))
    yy = jnp.mean(jax.vmap(lambda yi: jax.vmap(lambda yj: rbf_kernel(yi, yj, sigmas))(target_norm))(target_norm))
    xy = jnp.mean(jax.vmap(lambda xi: jax.vmap(lambda yj: rbf_kernel(xi, yj, sigmas))(target_norm))(sim_norm))
    return xx + yy - 2 * xy


def percintile_based_mmd(sim_norm, target_norm, scale_weights = jnp.array([0.1, 0.1, 0.3, 0.25, 0.25])):
    """MMD using percentiles as natural scales"""
    distances = jax.vmap(lambda x: jax.vmap(lambda y: jnp.linalg.norm(x - y))(target_norm))(sim_norm)
    distance_flat = distances.flatten()

    # Use percentiles as natural scales
    sigmas = jnp.array([
        jnp.percentile(distance_flat, 10),   # Fine scale
        jnp.percentile(distance_flat, 25),   # Small scale  
        jnp.percentile(distance_flat, 50),   # Medium scale (median)
        jnp.percentile(distance_flat, 75),   # Large scale
        jnp.percentile(distance_flat, 90),   # Very large scale
    ])
    
    mmd = jnp.sum(scale_weights * jax.vmap(lambda sigma: compute_mmd(sim_norm, target_norm, sigma))(sigmas))/len(sigmas)
    return mmd 