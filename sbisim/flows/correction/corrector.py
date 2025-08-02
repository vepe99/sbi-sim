import flax.linen as nn
from jax import vmap, value_and_grad
import jax
import jax.random as jr
from functools import partial

from ...simulations import LotkaVolterraSimulator, SBISimulator, OdisseoSimulator
from ...utils import instantiate_from_config
import jax.numpy as jnp
import numpy as np

from jax.lax import stop_gradient
from jax.sharding import Mesh, PartitionSpec, NamedSharding

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
        # jax.debug.print("{theta_1_shape}", theta_1.shape)
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
    Corrector model for Odisseo simulator, differentiable version.
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
    sharding: bool = False

    def setup(self):

        self.simulator_impl: OdisseoSimulator = instantiate_from_config(self.simulator)
        self.controlled_flow_impl: nn.Module = instantiate_from_config(self.controlled_flow)
        self.aggregration_impl: nn.Module = instantiate_from_config(self.aggregation)

        if self.sharding:
            # Create a pmapped version of mmd computation
            self.pmapped_mmd = jax.pmap(
                lambda theta_batch, context_batch: vmap(value_and_grad(self.mmd), in_axes=0)(theta_batch, context_batch),
                axis_name='devices'
            )

    def __call__(self, t, theta, context, train=True):

        return self.forward(t, theta, context, train=train)[0]

    def mmd(self, theta_1, target):

        simulator_rng = self.make_rng('simulator')
        # print(f"In the corrector: theta_1 shape: {theta_1.shape}, target shape: {target.shape}")

        # @partial(jax.checkpoint, static_argnums=(1, 2, 3))
        # def run_simulator(theta_1, num_simulations=self.num_simulations,
        #                                 rng=simulator_rng, deterministic=False,):
        #     return self.simulator_impl(theta_1, num_simulations=num_simulations,
        #                                 rng=rng, deterministic=deterministic)
        # output, _ = run_simulator(theta_1, num_simulations=self.num_simulations,
        #                                 rng=simulator_rng, deterministic=False, )
        output, _ = self.simulator_impl(theta_1, num_simulations=self.num_simulations,
                                        rng=simulator_rng, deterministic=False, )  # noqa
        

        # output = output[0]

        # output = output * self.simulator_impl.std_Y + self.simulator_impl.mean_Y
        # target = target * self.simulator_impl.std_Y + self.simulator_impl.mean_Y
        # print(f" In the corrector: output shape: {output.shape}, target shape: {target.shape}")
        
        return percintile_based_mmd(output, target) 

    def mmd_laxmap(self, theta_and_context):
        theta_1, context = theta_and_context
        return self.mmd(theta_1, context)
        

    def forward_flow(self, t, theta, context, train=False):

        # we need this because self.model was trained with stacked flow which has additional time dimensions
        return self.model(t, theta, context, train=train)

    def forward(self, t, theta, context, train=True):

        # predict flow
        flow_pred = self.model(t, theta, context, train=train)

        if self.freeze:
            flow_pred = stop_gradient(flow_pred)

        theta_1 = theta + jnp.einsum('ab,a->ab', flow_pred, 1 - t[:, 0])

        # print(f"In the corrector (should have a batch_dimension): theta_1 shape: {theta_1.shape}, context shape: {context.shape}")
        grad_fn = vmap(value_and_grad(self.mmd), in_axes=0)

        if self.sharding == True:
            mesh = Mesh(np.array(jax.devices()), ("i",))
            theta_1 = jax.device_put(theta_1, NamedSharding(mesh, PartitionSpec("i")))
            context = jax.device_put(context, NamedSharding(mesh, PartitionSpec("i")))
            loss, grad = grad_fn(theta_1, context)
            # loss = jax.device_put(loss, self.controlled_flow_impl['blocks_0']['layer1']['bias'].device())
            # grad = jax.device_put(grad, self.controlled_flow_impl['blocks_0']['layer1']['bias'].device())
            loss = jax.device_put(loss, np.array(jax.devices())[0])
            grad = jax.device_put(grad, np.array(jax.devices())[0])
        else:
            loss, grad = grad_fn(theta_1, context)
        
        ##LAXMPA
        ## loss, grad = jax.lax.map(value_and_grad(self.mmd_laxmap), batch_size=6, xs=(theta_1, context))

        loss = jnp.expand_dims(loss, axis=1)

        output = jnp.concatenate([loss, grad], axis=1)
        output = jnp.nan_to_num(output).clip(-self.clip_output, self.clip_output)

        output = jnp.concatenate([flow_pred, t, output], axis=1)
        drift = flow_pred + self.controlled_flow_impl(output, context=None) # noqa
        # drift = flow_pred + self.controlled_flow_impl(sample=flow_pred, timesteps=t, encoder_hidden_states=context, loss_grad=output, train=train)  #this is for the conv_2d_cross_attention 

        drift = (jnp.einsum('ab, a -> ab', drift, t[:, 0] > self.start_time) +
                 jnp.einsum('ab, a -> ab', flow_pred, t[:, 0] <= self.start_time))

        return drift, output


def rbf_kernel(x, y, sigma):
    """RBF kernel optimized for 6D astronomical data"""
    return jnp.exp(-jnp.sum((x - y)**2) / (2 * sigma**2))

def compute_mmd(sim_norm, target_norm, sigma):
    xx = jnp.mean(jax.vmap(lambda xi: jax.vmap(lambda xj: rbf_kernel(xi, xj, sigma))(sim_norm))(sim_norm))
    yy = jnp.mean(jax.vmap(lambda yi: jax.vmap(lambda yj: rbf_kernel(yi, yj, sigma))(target_norm))(target_norm))
    xy = jnp.mean(jax.vmap(lambda xi: jax.vmap(lambda yj: rbf_kernel(xi, yj, sigma))(target_norm))(sim_norm))
    return (1/len(sim_norm)**2)*xx + (1/len(target_norm)**2)*yy - 2/(len(sim_norm)*len(target_norm)) * xy


def percintile_based_mmd(sim_norm, target_norm, scale_weights = jnp.array([0.25, 0.25, 0.3, 0.1, 0.1])):
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
    
    mmd = jnp.sum(scale_weights * jax.vmap(lambda sigmas: compute_mmd(sim_norm, target_norm, sigmas))(sigmas))/len(sigmas)
    return mmd 

class CorrectorDifferentiableSimulatorOdisseoAggregation(nn.Module):
    """
    Corrector model for Odisseo simulator, differentiable version.
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
    sharding: bool = False

    def setup(self):

        self.simulator_impl: OdisseoSimulator = instantiate_from_config(self.simulator)
        self.controlled_flow_impl: nn.Module = instantiate_from_config(self.controlled_flow)
        self.aggregration_impl: nn.Module = instantiate_from_config(self.aggregation)

        if self.sharding:
            # Create a pmapped version of mmd computation
            self.pmapped_mmd = jax.pmap(
                lambda theta_batch, context_batch: vmap(value_and_grad(self.mmd), in_axes=0)(theta_batch, context_batch),
                axis_name='devices'
            )

    def __call__(self, t, theta, context, train=True):

        return self.forward(t, theta, context, train=train)[0]

    def mmd(self, theta_1, target):

        simulator_rng = self.make_rng('simulator')
        # print(f"In the corrector: theta_1 shape: {theta_1.shape}, target shape: {target.shape}")
        output, _ = self.simulator_impl(theta_1, num_simulations=self.num_simulations,
                                        rng=simulator_rng, deterministic=False, )  # noqa

        # output = output[0]

        # output = output * self.simulator_impl.std_Y + self.simulator_impl.mean_Y
        # target = target * self.simulator_impl.std_Y + self.simulator_impl.mean_Y
        # print(f" In the corrector: output shape: {output.shape}, target shape: {target.shape}")
        
        return percintile_based_mmd(output, target) 

    def mmd_laxmap(self, theta_and_context):
        theta_1, context = theta_and_context
        return self.mmd(theta_1, context)
        

    def forward_flow(self, t, theta, context, train=False):

        # we need this because self.model was trained with stacked flow which has additional time dimensions
        return self.model(t, theta, context, train=train)

    def forward(self, t, theta, context, train=True):

        # predict flow
        flow_pred = self.model(t, theta, context, train=train)

        if self.freeze:
            flow_pred = stop_gradient(flow_pred)

        theta_1 = theta + jnp.einsum('ab,a->ab', flow_pred, 1 - t[:, 0])

        # print(f"In the corrector (should have a batch_dimension): theta_1 shape: {theta_1.shape}, context shape: {context.shape}")
        grad_fn = vmap(value_and_grad(self.mmd), in_axes=0)

        if self.sharding == True:
            mesh = Mesh(np.array(jax.devices()), ("i",))
            theta_1 = jax.device_put(theta_1, NamedSharding(mesh, PartitionSpec("i")))
            context = jax.device_put(context, NamedSharding(mesh, PartitionSpec("i")))
            loss, grad = grad_fn(theta_1, context)
            # loss = jax.device_put(loss, self.controlled_flow_impl['blocks_0']['layer1']['bias'].device())
            # grad = jax.device_put(grad, self.controlled_flow_impl['blocks_0']['layer1']['bias'].device())
            loss = jax.device_put(loss, np.array(jax.devices())[0])
            grad = jax.device_put(grad, np.array(jax.devices())[0])
        else:
            loss, grad = grad_fn(theta_1, context)
        
        ##LAXMPA
        ## loss, grad = jax.lax.map(value_and_grad(self.mmd_laxmap), batch_size=6, xs=(theta_1, context))

        loss = jnp.expand_dims(loss, axis=1)

        output = jnp.concatenate([loss, grad], axis=1)
        output = jnp.nan_to_num(output).clip(-self.clip_output, self.clip_output)

        output = jnp.concatenate([flow_pred, t, output], axis=1)
        drift = flow_pred + self.controlled_flow_impl(output, context=None) # noqa
        # drift = flow_pred + self.controlled_flow_impl(sample=flow_pred, timesteps=t, encoder_hidden_states=context, loss_grad=output, train=train)  #this is for the conv_2d_cross_attention 

        drift = (jnp.einsum('ab, a -> ab', drift, t[:, 0] > self.start_time) +
                 jnp.einsum('ab, a -> ab', flow_pred, t[:, 0] <= self.start_time))

        return drift, output


def rbf_kernel(x, y, sigma):
    """RBF kernel optimized for 6D astronomical data"""
    return jnp.exp(-jnp.sum((x - y)**2) / (2 * sigma**2))

def compute_mmd(sim_norm, target_norm, sigma):
    xx = jnp.mean(jax.vmap(lambda xi: jax.vmap(lambda xj: rbf_kernel(xi, xj, sigma))(sim_norm))(sim_norm))
    yy = jnp.mean(jax.vmap(lambda yi: jax.vmap(lambda yj: rbf_kernel(yi, yj, sigma))(target_norm))(target_norm))
    xy = jnp.mean(jax.vmap(lambda xi: jax.vmap(lambda yj: rbf_kernel(xi, yj, sigma))(target_norm))(sim_norm))
    return (1/len(sim_norm)**2)*xx + (1/len(target_norm)**2)*yy - 2/(len(sim_norm)*len(target_norm)) * xy


def percintile_based_mmd(sim_norm, target_norm, scale_weights = jnp.array([0.25, 0.25, 0.3, 0.1, 0.1])):
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
    
    mmd = jnp.sum(scale_weights * jax.vmap(lambda sigmas: compute_mmd(sim_norm, target_norm, sigmas))(sigmas))/len(sigmas)
    return mmd 