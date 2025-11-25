import flax.linen as nn
from jax import vmap, value_and_grad
import jax
import jax.random as jr
from functools import partial

from ...simulations import LotkaVolterraSimulator, SBISimulator, OdisseoSimulator
from ...utils import instantiate_from_config
import jax.numpy as jnp
from jax.scipy.special import logsumexp

import numpy as np

from jax.lax import stop_gradient
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.scipy.stats import norm
from astropy import units as u
from odisseo.units import CodeUnits
from jax.experimental import checkify

from optimistix import least_squares
import optimistix
from optax import adamw


from math import log10

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
        print('loss:', loss, 
              'grad:', grad)

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

    def __call__(self, t, theta, context, train=True):

        return self.forward(t, theta, context, train=train)[0]

    def mmd(self, theta_1, target):

        simulator_rng = self.make_rng('simulator')
        # print(f"In the corrector: theta_1 shape: {theta_1.shape}, target shape: {target.shape}")
        output, _ = self.simulator_impl(theta_1, num_simulations=self.num_simulations,
                                        rng=simulator_rng, deterministic=False, )  # noqa
        
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

        # print(f"In the corrector (should have a batch_dimension): theta_1 shape: {theta_1.shape}, context shape: {context.shape}")
        grad_fn = vmap(value_and_grad(self.mmd), in_axes=0)
        loss, grad = grad_fn(theta_1, context)
        
        #LAXMAP
        # def mmd_laxmap(theta):
        #     return self.mmd(theta, context)
        
        # loss, grad = jax.lax.map(value_and_grad(mmd_laxmap), batch_size=1, xs=(theta_1))
        

        loss = jnp.expand_dims(loss, axis=1)

        output = jnp.concatenate([loss, grad], axis=1)
        output = jnp.nan_to_num(output).clip(-self.clip_output, self.clip_output)

        output = jnp.concatenate([flow_pred, t, output], axis=1)
        drift = flow_pred + self.controlled_flow_impl(output, context=None) # noqa
        # drift = flow_pred + self.controlled_flow_impl(sample=flow_pred, timesteps=t, encoder_hidden_states=context, loss_grad=output, train=train)  #this is for the conv_2d_cross_attention 

        drift = (jnp.einsum('ab, a -> ab', drift, t[:, 0] > self.start_time) +
                 jnp.einsum('ab, a -> ab', flow_pred, t[:, 0] <= self.start_time))

        return drift, output
    

class CorrectorDifferentiableSimulatorOdisseoNewLoss(nn.Module):
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

    def __call__(self, t, theta, context, train=True):

        return self.forward(t, theta, context, train=train)[0]

    def NLL(self, theta_1, target):

        simulator_rng = self.make_rng('simulator')
        # print(f"In the corrector: theta_1 shape: {theta_1.shape}, target shape: {target.shape}")
        output, _ = self.simulator_impl(theta_1, num_simulations=self.num_simulations,
                                        rng=simulator_rng, deterministic=True, )  # noqa

        return -1 * stream_likelihood(model_stream=output, obs_stream=target, obs_errors=jnp.array([0.25, 0.001, 0.15, 5., 0.1, 0.0001]))


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
        grad_fn = vmap(value_and_grad(self.NLL), in_axes=0)
        loss, grad = grad_fn(theta_1, context)
        

        loss = jnp.expand_dims(loss, axis=1)

        output = jnp.concatenate([loss, grad], axis=1)
        output = jnp.nan_to_num(output).clip(-self.clip_output, self.clip_output)

        # output = jnp.concatenate([flow_pred, t, output], axis=1) #baseline

        drift = flow_pred + self.controlled_flow_impl(output, context=None) # noqa, baseline 
        # drift = flow_pred + self.controlled_flow_impl(sample=flow_pred, timesteps=t, encoder_hidden_states=context, loss_grad=output, train=train)  #this is for the conv_2d_cross_attention
        # drift = flow_pred + self.controlled_flow_impl(timesteps=t, theta=theta, y=jnp.concatenate([flow_pred, output], axis=1), ) # noqa
        drift = (jnp.einsum('ab, a -> ab', drift, t[:, 0] > self.start_time) +
                 jnp.einsum('ab, a -> ab', flow_pred, t[:, 0] <= self.start_time))

        return drift, output
    

class CorrectorDifferentiableSimulatorOdisseoAggregationNewLoss(nn.Module):
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

    def __call__(self, t, theta, context, train=True):

        return self.forward(t, theta, context, train=train)[0]

    def NLL(self, theta_1, target):

        simulator_rng = self.make_rng('simulator')
        # print(f"In the corrector: theta_1 shape: {theta_1.shape}, target shape: {target.shape}")
        output, _ = self.simulator_impl(theta_1, num_simulations=self.num_simulations,
                                        rng=simulator_rng, deterministic=True, )  # noqa

        return -1 * stream_likelihood(model_stream=output, obs_stream=target, obs_errors=jnp.array([0.25, 0.001, 0.15, 5., 0.1, 0.0001]))


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
        grad_fn = vmap(value_and_grad(self.NLL), in_axes=0)
        loss, grad = grad_fn(theta_1, context)
        print('loss:', loss)
        print('grad:', grad)

        loss = jnp.expand_dims(loss, axis=1)

        output = jnp.concatenate([loss, grad], axis=1)
        output = jnp.nan_to_num(output).clip(-self.clip_output, self.clip_output)

        output = self.aggregration_impl(output)

        output = jnp.concatenate([flow_pred, t, output], axis=1)
        drift = flow_pred + self.controlled_flow_impl(output, context=None) # noqa
        # drift = flow_pred + self.controlled_flow_impl(sample=flow_pred, timesteps=t, encoder_hidden_states=context, loss_grad=output, train=train)  #this is for the conv_2d_cross_attention
        
        drift = (jnp.einsum('ab, a -> ab', drift, t[:, 0] > self.start_time) +
                 jnp.einsum('ab, a -> ab', flow_pred, t[:, 0] <= self.start_time))

        return drift, output
    
class CorrectorDifferentiableSimulatorOdisseoAggregationNewLoss_noclipping(nn.Module):
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

    def __call__(self, t, theta, context, train=True):

        return self.forward(t, theta, context, train=train)[0]

    def NLL(self, theta_1, target):

        simulator_rng = self.make_rng('simulator')
        # print(f"In the corrector: theta_1 shape: {theta_1.shape}, target shape: {target.shape}")
        output, _ = self.simulator_impl(theta_1, num_simulations=self.num_simulations,
                                        rng=simulator_rng, deterministic=True, )  # noqa

        return -1 * stream_likelihood(model_stream=output, obs_stream=target, obs_errors=jnp.array([0.25, 0.001, 0.15, 5., 0.1, 0.0001]))


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
        grad_fn = vmap(value_and_grad(self.NLL), in_axes=0)
        loss, grad = grad_fn(theta_1, context)
        print('loss:', loss)
        print('grad:', grad)

        loss = jnp.expand_dims(loss, axis=1)/10_000
        grad = grad/jnp.linalg.norm(grad, axis=1, keepdims=True)

        output = jnp.concatenate([loss, grad], axis=1)
        output = jnp.nan_to_num(output).clip(-self.clip_output, self.clip_output)

        output = self.aggregration_impl(output)

        output = jnp.concatenate([flow_pred, t, output], axis=1)
        drift = flow_pred + self.controlled_flow_impl(output, context=None) # noqa
        # drift = flow_pred + self.controlled_flow_impl(sample=flow_pred, timesteps=t, encoder_hidden_states=context, loss_grad=output, train=train)  #this is for the conv_2d_cross_attention
        
        drift = (jnp.einsum('ab, a -> ab', drift, t[:, 0] > self.start_time) +
                 jnp.einsum('ab, a -> ab', flow_pred, t[:, 0] <= self.start_time))

        return drift, output
    
class CorrectorDifferentiableSimulatorOdisseoAggregationNewLoss_transformer_noclipping(nn.Module):
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

    def __call__(self, t, theta, context, train=True):

        return self.forward(t, theta, context, train=train)[0]

    def NLL(self, theta_1, target):

        simulator_rng = self.make_rng('simulator')
        # print(f"In the corrector: theta_1 shape: {theta_1.shape}, target shape: {target.shape}")
        output, _ = self.simulator_impl(theta_1, num_simulations=self.num_simulations,
                                        rng=simulator_rng, deterministic=True, )  # noqa

        return -1 * stream_likelihood(model_stream=output, obs_stream=target, obs_errors=jnp.array([0.25, 0.001, 0.15, 5., 0.1, 0.0001]))


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
        grad_fn = vmap(value_and_grad(self.NLL), in_axes=0)
        loss, grad = grad_fn(theta_1, context)
        print('loss:', loss)
        print('grad:', grad)

        loss = jnp.expand_dims(loss, axis=1)/20_000
        grad = grad/jnp.linalg.norm(grad, axis=1, keepdims=True)

        output = jnp.concatenate([loss, grad], axis=1)
        output = jnp.nan_to_num(output).clip(-self.clip_output, self.clip_output)

        # output = self.aggregration_impl(output)

        output = jnp.concatenate([flow_pred, t, output], axis=1)
        # drift = flow_pred + self.controlled_flow_impl(theta=flow_pred, context=output, t=t) # noqa
        drift = flow_pred + self.controlled_flow_impl(sample=flow_pred, timesteps=t, encoder_hidden_states=context, loss_grad=output, train=train)  #this is for the conv_2d_cross_attention
        # drift = flow_pred + self.controlled_flow_impl(output, context=None) # noqa, baseline
        drift = (jnp.einsum('ab, a -> ab', drift, t[:, 0] > self.start_time) +
                 jnp.einsum('ab, a -> ab', flow_pred, t[:, 0] <= self.start_time))

        return drift, output


class CorrectorDifferentiableSimulatorOdisseoAggregationMMD_transformer_noclipping(nn.Module):
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

    def __call__(self, t, theta, context, train=True):

        return self.forward(t, theta, context, train=train)[0]

    def NLL(self, theta_1, target):

        simulator_rng = self.make_rng('simulator')
        # print(f"In the corrector: theta_1 shape: {theta_1.shape}, target shape: {target.shape}")
        output, _ = self.simulator_impl(theta_1, num_simulations=self.num_simulations,
                                        rng=simulator_rng, deterministic=True, )  # noqa

        return percintile_based_mmd(sim_norm=output, target_norm=target, )/10


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
        grad_fn = vmap(value_and_grad(self.NLL), in_axes=0)
        loss, grad = grad_fn(theta_1, context)
        print('loss:', loss)
        print('grad:', grad)

        loss = jnp.expand_dims(loss, axis=1)/20_000
        grad = grad/jnp.linalg.norm(grad, axis=1, keepdims=True)

        output = jnp.concatenate([loss, grad], axis=1)
        output = jnp.nan_to_num(output).clip(-self.clip_output, self.clip_output)

        # output = self.aggregration_impl(output)

        output = jnp.concatenate([flow_pred, t, output], axis=1)
        drift = flow_pred + self.controlled_flow_impl(theta=flow_pred, context=output, t=t) # noqa
        # drift = flow_pred + self.controlled_flow_impl(sample=flow_pred, timesteps=t, encoder_hidden_states=context, loss_grad=output, train=train)  #this is for the conv_2d_cross_attention
        # drift = flow_pred + self.controlled_flow_impl(output, context=None) # noqa, baseline
        drift = (jnp.einsum('ab, a -> ab', drift, t[:, 0] > self.start_time) +
                 jnp.einsum('ab, a -> ab', flow_pred, t[:, 0] <= self.start_time))

        return drift, output

class CorrectorDifferentiableSimulatorOdisseoAggregationNewLoss_transformfirst(nn.Module):
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
        self.low=jnp.array([0.5,
                            10**3., 
                            1/4 * 0.008,
                            10**log10(1/4 * 4.3683325e11), 
                            1/4 * 16,
                            10**log10(1/4 *68_193_902_782.346756),
                             1/4 * 3,
                              ])
        self.high=jnp.array([5, 
                             10**4.5, 
                             2 * 0.008, 
                             10**log10(2 * 4.3683325e11),
                             2 * 16,
                             10**log10(2 * 68_193_902_782.346756),
                             2 * 3,
                             ])
        code_length = 10.0 * u.kpc
        code_mass = 1e4 * u.Msun
        code_time = 3 * u.Gyr
        self.code_units = CodeUnits(code_length, code_mass, G=1, unit_time = code_time )  
        self.mean_pointcloud = jnp.load('/export/data/vgiusepp/odisseo_data/data_fix_position/preprocess/mean_std_1e5_pointcloud.npz')['mean_x']
        self.std_pointcloud = jnp.load('/export/data/vgiusepp/odisseo_data/data_fix_position/preprocess/mean_std_1e5_pointcloud.npz')['std_x']

    def __call__(self, t, theta, context, train=True):

        return self.forward(t, theta, context, train=train)[0]

    def NLL(self, theta_1, target):

        simulator_rng = self.make_rng('simulator')
        # print(f"In the corrector: theta_1 shape: {theta_1.shape}, target shape: {target.shape}")
        output, _ = self.simulator_impl(theta_1, num_simulations=self.num_simulations,
                                        rng=simulator_rng, deterministic=True, )  # noqa

        return -1 * stream_likelihood(model_stream=output, obs_stream=target, obs_errors=jnp.array([0.25, 0.001, 0.15, 5., 0.1, 0.0001]))/1000.0


    def forward_flow(self, t, theta, context, train=False):

        # we need this because self.model was trained with stacked flow which has additional time dimensions
        return self.model(t, theta, context, train=train)

    def forward(self, t, theta, context, train=True):

        # predict flow
        flow_pred = self.model(t, theta, context, train=train)

        if self.freeze:
            flow_pred = stop_gradient(flow_pred)

        theta_1 = theta + jnp.einsum('ab,a->ab', flow_pred, 1 - t[:, 0])
        print(theta_1.shape)

        theta_1 = norm.cdf(theta_1) * (self.high-self.low) + self.low
        theta_1 = theta_1.at[:, 0].set(theta_1[:, 0] * u.Gyr.to(self.code_units.code_time))
        theta_1 = theta_1.at[:, 1].set(theta_1[:, 1] * u.Msun.to(self.code_units.code_mass))
        theta_1 = theta_1.at[:, 2].set(theta_1[:, 2] * u.kpc.to(self.code_units.code_length))
        theta_1 = theta_1.at[:, 3].set(theta_1[:, 3] * u.Msun.to(self.code_units.code_mass))
        theta_1 = theta_1.at[:, 4].set(theta_1[:, 4] * u.kpc.to(self.code_units.code_length))
        theta_1 = theta_1.at[:, 5].set(theta_1[:, 5] * u.Msun.to(self.code_units.code_mass))
        theta_1 = theta_1.at[:, 6].set(theta_1[:, 6] * u.kpc.to(self.code_units.code_length))

        # context = context * self.std_pointcloud + self.mean_pointcloud

        # print(f"In the corrector (should have a batch_dimension): theta_1 shape: {theta_1.shape}, context shape: {context.shape}")
        grad_fn = vmap(value_and_grad(self.NLL), in_axes=0)
        loss, grad = grad_fn(theta_1, context)
        print('loss:', loss)
        print('grad:', grad)

        loss = jnp.expand_dims(loss, axis=1)

        output = jnp.concatenate([loss, grad], axis=1)
        output = jnp.nan_to_num(output).clip(-self.clip_output, self.clip_output)

        output = self.aggregration_impl(output)

        output = jnp.concatenate([flow_pred, t, output], axis=1)
        drift = flow_pred + self.controlled_flow_impl(output, context=None) # noqa
        # drift = flow_pred + self.controlled_flow_impl(sample=flow_pred, timesteps=t, encoder_hidden_states=context, loss_grad=output, train=train)  #this is for the conv_2d_cross_attention
        
        drift = (jnp.einsum('ab, a -> ab', drift, t[:, 0] > self.start_time) +
                 jnp.einsum('ab, a -> ab', flow_pred, t[:, 0] <= self.start_time))

        return drift, output

class CorrectorDifferentiableSimulatorOdisseoAggregationNormalizedNewLoss_transformfirst(nn.Module):
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
        self.low=jnp.array([0.5,
                            10**3., 
                            1/4 * 0.008,
                            10**log10(1/4 * 4.3683325e11), 
                            1/4 * 16,
                            10**log10(1/4 *68_193_902_782.346756),
                            1/4 * 3,
                            10.0, #x
                            0.1, #y
                            6.0, #z
                            90.0, #vx
                            -280.0, #vy
                            -120.0]) #vz
        self.high=jnp.array([5, 
                             10**4.5, 
                             2 * 0.008, 
                             10**log10(2 * 4.3683325e11),
                             2 * 16,
                             10**log10(2 * 68_193_902_782.346756),
                             2 * 3,
                             14.0, #x
                             2.5,  #y
                             8.0,  #z
                             115.0, #vx
                             -230.0, #vy
                             -80.0]) #vz
        code_length = 10.0 * u.kpc
        code_mass = 1e4 * u.Msun
        code_time = 3 * u.Gyr
        self.code_units = CodeUnits(code_length, code_mass, G=1, unit_time = code_time )  
        self.mean_pointcloud = jnp.load('/export/data/vgiusepp/odisseo_data/data_fix_position/preprocess/mean_std_1e5_pointcloud.npz')['mean_x']
        self.std_pointcloud = jnp.load('/export/data/vgiusepp/odisseo_data/data_fix_position/preprocess/mean_std_1e5_pointcloud.npz')['std_x']

    def __call__(self, t, theta, context, train=True):

        return self.forward(t, theta, context, train=train)[0]

    def NLL(self, theta_1, target):

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

        simulator_rng = self.make_rng('simulator')
        # print(f"In the corrector: theta_1 shape: {theta_1.shape}, target shape: {target.shape}")
        stream, _ = self.simulator_impl(theta_1, num_simulations=self.num_simulations,
                                        rng=simulator_rng, deterministic=True, )  # noqa

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

        dens_phi_target = kde2d_on_grid(target[:, [1,2]], PHI1, PHI2, bw_phi)
        dens_v_target   = kde2d_on_grid(target[:, [4,5]], V1, V2, bw_v)
        dens_R_target   = kde2d_on_grid(target[:, [0,3]], R_GRID, VR_GRID, bw_R)

        return jnp.exp(-0.1 * (loss_js(dens_phi_target, dens_phi) +
                loss_js(dens_v_target, dens_v) +
                loss_js(dens_R_target, dens_R)))


    def forward_flow(self, t, theta, context, train=False):

        # we need this because self.model was trained with stacked flow which has additional time dimensions
        return self.model(t, theta, context, train=train)

    def forward(self, t, theta, context, train=True):

        # predict flow
        flow_pred = self.model(t, theta, context, train=train)

        if self.freeze:
            flow_pred = stop_gradient(flow_pred)

        # print(theta.shape)

        theta_1 = theta + jnp.einsum('ab,a->ab', flow_pred, 1 - t[:, 0])
        

        theta_1 = norm.cdf(theta_1) * (self.high-self.low) + self.low
        theta_1 = theta_1.at[:, 0].set(theta_1[:, 0] * u.Gyr.to(self.code_units.code_time))
        theta_1 = theta_1.at[:, 1].set(jnp.log10(theta_1[:, 1] * u.Msun.to(self.code_units.code_mass)))
        theta_1 = theta_1.at[:, 2].set(theta_1[:, 2] * u.kpc.to(self.code_units.code_length))
        theta_1 = theta_1.at[:, 3].set(jnp.log10(theta_1[:, 3] * u.Msun.to(self.code_units.code_mass)))
        theta_1 = theta_1.at[:, 4].set(theta_1[:, 4] * u.kpc.to(self.code_units.code_length))
        theta_1 = theta_1.at[:, 5].set(theta_1[:, 5] * u.Msun.to(self.code_units.code_mass))
        theta_1 = theta_1.at[:, 6].set(theta_1[:, 6] * u.kpc.to(self.code_units.code_length))

        # context = context * self.std_pointcloud + self.mean_pointcloud

        # print(f"In the corrector (should have a batch_dimension): theta_1 shape: {theta_1.shape}, context shape: {context.shape}")
        grad_fn = vmap(value_and_grad(self.NLL), in_axes=0)
        loss, grad = grad_fn(theta_1, context)
        # print('loss:', loss)
        # print('grad:', grad)

        # grad = grad / jnp.linalg.norm(grad, axis=1, keepdims=True)

        # print('normalized grad:', grad)

        loss = jnp.expand_dims(loss, axis=1)

        output = jnp.concatenate([loss, grad], axis=1)
        output = jnp.nan_to_num(output).clip(-self.clip_output, self.clip_output)

        output = self.aggregration_impl(output)

        # output = jnp.concatenate([flow_pred, t, output], axis=1)
        drift = flow_pred + self.controlled_flow_impl(theta=flow_pred, context=output, t=t) # noqa
        # drift = flow_pred + self.controlled_flow_impl(output, context=None) # noqa
        # drift = flow_pred + self.controlled_flow_impl(sample=flow_pred, timesteps=t, encoder_hidden_states=context, loss_grad=output, train=train)  #this is for the conv_2d_cross_attention
        
        drift = (jnp.einsum('ab, a -> ab', drift, t[:, 0] > self.start_time) +
                 jnp.einsum('ab, a -> ab', flow_pred, t[:, 0] <= self.start_time))

        return drift, output
    

class CorrectorDifferentiableSimulatorOdisseo_uniformprior_jsloss(nn.Module):
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
    
        code_length = 10.0 * u.kpc
        code_mass = 1e4 * u.Msun
        code_time = 3 * u.Gyr
        self.code_units = CodeUnits(code_length, code_mass, G=1, unit_time = code_time )  

        # Define bin edges and create meshgrids
        phi1_bins = jnp.linspace(-120, 70, 65)    # 64 bins
        phi2_bins = jnp.linspace(-8, 2, 33)       # 32 bins
        v1_bins = jnp.linspace(-2., 1.0, 65)      # 64 bins  
        v2_bins = jnp.linspace(-0.10, 0.10, 33)   # 32 bins
        R_bins = jnp.linspace(6, 20, 65)          # 64 bins
        vR_bins = jnp.linspace(-250, 250, 33)     # 32 bins

        # Create meshgrids for bin edges (not centers)
        self.PHI1, self.PHI2 = jnp.meshgrid(phi1_bins, phi2_bins, indexing='ij')
        self.V1, self.V2 = jnp.meshgrid(v1_bins, v2_bins, indexing='ij')
        self.R_GRID, self.VR_GRID = jnp.meshgrid(R_bins, vR_bins, indexing='ij')


    def __call__(self, t, theta, context, train=True):

        return self.forward(t, theta, context, train=train)[0]

    def NLL(self, theta_1, target):


        simulator_rng = self.make_rng('simulator')
        # print(f"In the corrector: theta_1 shape: {theta_1.shape}, target shape: {target.shape}")
        stream, _ = self.simulator_impl(theta_1, num_simulations=self.num_simulations,
                                        rng=simulator_rng, deterministic=True, )  # noqa

        # take relevant projections from simulated stream
        x_phi = stream[:, [1,2]]   # phi1, phi2
        x_v   = stream[:, [4,5]]   # vphi1, vphi2
        x_R   = stream[:, [0,3]]   # R, v_radial

        # choose bandwidths (tune or use Silverman's rule)
        bw_phi = jnp.array([2.0, 0.5])     # example: phi1=2deg, phi2=0.5deg
        bw_v   = jnp.array([0.1, 0.01])    # example velocities
        bw_R   = jnp.array([0.5, 20.0])    # example R and vR

        # KDE densities on each meshgrid
        dens_phi = kde2d_on_grid(x_phi, self.PHI1, self.PHI2, bw_phi)
        dens_v   = kde2d_on_grid(x_v, self.V1, self.V2, bw_v)
        dens_R   = kde2d_on_grid(x_R, self.R_GRID, self.VR_GRID, bw_R)

        dens_phi_target = kde2d_on_grid(target[:, [1,2]], self.PHI1, self.PHI2, bw_phi)
        dens_v_target   = kde2d_on_grid(target[:, [4,5]], self.V1, self.V2, bw_v)
        dens_R_target   = kde2d_on_grid(target[:, [0,3]], self.R_GRID, self.VR_GRID, bw_R)

        return jnp.exp(-0.1 * (loss_js(dens_phi_target, dens_phi) +
                loss_js(dens_v_target, dens_v) +
                loss_js(dens_R_target, dens_R)))


    def forward_flow(self, t, theta, context, train=False):

        # we need this because self.model was trained with stacked flow which has additional time dimensions
        return self.model(t, theta, context, train=train)

    def forward(self, t, theta, context, train=True):

        # predict flow
        flow_pred = self.model(t, theta, context, train=train)

        if self.freeze:
            flow_pred = stop_gradient(flow_pred)

        # print(theta.shape)
        theta_1 = theta + jnp.einsum('ab,a->ab', flow_pred, 1 - t[:, 0])

        # print(f"In the corrector (should have a batch_dimension): theta_1 shape: {theta_1.shape}, context shape: {context.shape}")
        grad_fn = vmap(value_and_grad(self.NLL), in_axes=0)
        loss, grad = grad_fn(theta_1, context)
        print('loss:', loss)
        print('grad:', grad)

        # grad = grad / jnp.linalg.norm(grad, axis=1, keepdims=True)

        loss = jnp.expand_dims(loss, axis=1)

        output = jnp.concatenate([loss, grad], axis=1)
        output = jnp.nan_to_num(output).clip(-self.clip_output, self.clip_output)

        output = self.aggregration_impl(output)

        output = jnp.concatenate([flow_pred, t, output], axis=1)
        # drift = flow_pred + self.controlled_flow_impl(theta=flow_pred, context=output, t=t) # noqa
        drift = flow_pred + self.controlled_flow_impl(output, context=None) # noqa
        
        drift = (jnp.einsum('ab, a -> ab', drift, t[:, 0] > self.start_time) +
                 jnp.einsum('ab, a -> ab', flow_pred, t[:, 0] <= self.start_time))

        return drift, output

class CorrectorDifferentiableSimulatorOdisseoAggregationNormalizedNewLoss_transformfirst_nstep(nn.Module):
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
        self.low=jnp.array([0.5,
                            10**3., 
                            1/4 * 0.008,
                            10**log10(1/4 * 4.3683325e11), 
                            1/4 * 16,
                            10**log10(1/4 *68_193_902_782.346756),
                            1/4 * 3,
                            10.0, #x
                            0.1, #y
                            6.0, #z
                            90.0, #vx
                            -280.0, #vy
                            -120.0]) #vz
        self.high=jnp.array([5, 
                             10**4.5, 
                             2 * 0.008, 
                             10**log10(2 * 4.3683325e11),
                             2 * 16,
                             10**log10(2 * 68_193_902_782.346756),
                             2 * 3,
                             14.0, #x
                             2.5,  #y
                             8.0,  #z
                             115.0, #vx
                             -230.0, #vy
                             -80.0]) #vz
        code_length = 10.0 * u.kpc
        code_mass = 1e4 * u.Msun
        code_time = 3 * u.Gyr
        self.code_units = CodeUnits(code_length, code_mass, G=1, unit_time = code_time )  
        self.mean_pointcloud = jnp.load('/export/data/vgiusepp/odisseo_data/data_fix_position/preprocess/mean_std_1e5_pointcloud.npz')['mean_x']
        self.std_pointcloud = jnp.load('/export/data/vgiusepp/odisseo_data/data_fix_position/preprocess/mean_std_1e5_pointcloud.npz')['std_x']
        self.mean_X = jnp.load('/export/data/vgiusepp/odisseo_data/data_varying_position_newprior/preprocess/mean_std_1e5_parameter.npz')['mean_theta']
        self.std_X = jnp.load('/export/data/vgiusepp/odisseo_data/data_varying_position_newprior/preprocess/mean_std_1e5_parameter.npz')['std_theta']

    def __call__(self, t, theta, context, train=True):

        return self.forward(t, theta, context, train=train)[0]

    def NLL(self, theta_1, target):

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

        simulator_rng = self.make_rng('simulator')
        # print(f"In the corrector: theta_1 shape: {theta_1.shape}, target shape: {target.shape}")
        stream, _ = self.simulator_impl(theta_1, num_simulations=self.num_simulations,
                                        rng=simulator_rng, deterministic=True, )  # noqa

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

        dens_phi_target = kde2d_on_grid(target[:, [1,2]], PHI1, PHI2, bw_phi)
        dens_v_target   = kde2d_on_grid(target[:, [4,5]], V1, V2, bw_v)
        dens_R_target   = kde2d_on_grid(target[:, [0,3]], R_GRID, VR_GRID, bw_R)

        return jnp.exp(-0.1 * (loss_js(dens_phi_target, dens_phi) +
                loss_js(dens_v_target, dens_v) +
                loss_js(dens_R_target, dens_R)))


    def forward_flow(self, t, theta, context, train=False):

        # we need this because self.model was trained with stacked flow which has additional time dimensions
        return self.model(t, theta, context, train=train)

    def forward(self, t, theta, context, train=True):

        # predict flow
        flow_pred = self.model(t, theta, context, train=train)

        if self.freeze:
            flow_pred = stop_gradient(flow_pred)

        n_steps = 3
        dt = (1 - t[:, 0]) / n_steps
        
    
        # def transform_to_simulation_units(theta_normalized):
        #     """Transform normalized theta to simulation units"""
        #     theta_sim = theta_normalized * self.std_X + self.mean_X
        #     theta_sim = theta_sim.at[:, 0].set(theta_sim[:, 0] * u.Gyr.to(self.code_units.code_time))
        #     theta_sim = theta_sim.at[:, 1].set(jnp.log10(theta_sim[:, 1] * u.Msun.to(self.code_units.code_mass)))
        #     theta_sim = theta_sim.at[:, 2].set(theta_sim[:, 2] * u.kpc.to(self.code_units.code_length))
        #     theta_sim = theta_sim.at[:, 3].set(jnp.log10(theta_sim[:, 3] * u.Msun.to(self.code_units.code_mass)))
        #     theta_sim = theta_sim.at[:, 4].set(theta_sim[:, 4] * u.kpc.to(self.code_units.code_length))
        #     theta_sim = theta_sim.at[:, 5].set(theta_sim[:, 5] * u.Msun.to(self.code_units.code_mass))
        #     theta_sim = theta_sim.at[:, 6].set(theta_sim[:, 6] * u.kpc.to(self.code_units.code_length))
        #     return theta_sim

        # context = context * self.std_pointcloud + self.mean_pointcloud

        def integration_step(carry, step_idx):
            theta_current, t_current = carry
            
            # Get flow prediction
            flow_step = self.model(t_current, theta_current, context, train=train)
            if self.freeze:
                flow_step = stop_gradient(flow_step)
            
            # Euler step
            theta_next = theta_current + jnp.einsum('ab,a->ab', flow_step, dt)
            # jax.debug.print('{theta_next}', theta_next=theta_next)
            
            # Transform for gradient calculation
            # theta_sim = transform_to_simulation_units(theta_next)
            theta_sim = theta_next
            # jax.debug.print('{theta_sim}', theta_sim=theta_sim)
            
            # Calculate gradient at this step
            grad_fn = vmap(value_and_grad(self.NLL), in_axes=0)
            loss_step, grad_step = grad_fn(theta_sim, context)

            # jax.debug.print('{loss_step}', loss_step=loss_step)
            # jax.debug.print('{grad_step}', grad_step=grad_step)

            # Update time
            t_next = t_current.at[:, 0].add(dt)
            
            return (theta_next, t_next), (loss_step, grad_step)

        # Initialize and run integration
        initial_carry = (theta, t)
        final_carry, (all_losses, all_gradients) = jax.lax.scan(
            integration_step, 
            initial_carry, 
            jnp.arange(n_steps)
        )

        # Calculate mean gradient and loss
        grad = jnp.mean(all_gradients, axis=0)  # Mean over steps
        loss = jnp.mean(all_losses, axis=0)     # Mean over steps
        # jax.debug.print('{loss}', loss=loss)
        # jax.debug.print('{grad}', grad=grad)

        loss = jnp.expand_dims(loss, axis=1)

        output = jnp.concatenate([loss, grad], axis=1)
        output = jnp.nan_to_num(output).clip(-self.clip_output, self.clip_output)

        output = self.aggregration_impl(output)

        output = jnp.concatenate([flow_pred, t, output], axis=1)
        drift = flow_pred + self.controlled_flow_impl(output, context=None) # noqa


        drift = (jnp.einsum('ab, a -> ab', drift, t[:, 0] > self.start_time) +
                 jnp.einsum('ab, a -> ab', flow_pred, t[:, 0] <= self.start_time))

        return drift, output
    

class CorrectorDifferentiableSimulatorOdisseoTest(nn.Module):
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

    def __call__(self, t, theta, context, train=True):

        return self.forward(t, theta, context, train=train)[0]

    def NLL(self, theta_1, target):

        simulator_rng = self.make_rng('simulator')
        # print(f"In the corrector: theta_1 shape: {theta_1.shape}, target shape: {target.shape}")
        output, _ = self.simulator_impl(theta_1, num_simulations=self.num_simulations,
                                        rng=simulator_rng, deterministic=True, )  # noqa

        return  stream_likelihood_test(model_stream=output, obs_stream=target, obs_errors=jnp.array([0.25, 0.001, 0.15, 5., 0.1, 0.0001]))


    def forward_flow(self, t, theta, context, train=False):

        # we need this because self.model was trained with stacked flow which has additional time dimensions
        return self.model(t, theta, context, train=train)

    def forward(self, t, theta, context, train=True):

        # predict flow
        flow_pred = self.model(t, theta, context, train=train)

        if self.freeze:
            flow_pred = stop_gradient(flow_pred)

        theta_1 = theta + jnp.einsum('ab,a->ab', flow_pred, 1 - t[:, 0])
        print(theta_1)

        # print(f"In the corrector (should have a batch_dimension): theta_1 shape: {theta_1.shape}, context shape: {context.shape}")
        grad_fn = vmap(value_and_grad(self.NLL), in_axes=0)
        loss, grad = grad_fn(theta_1, context)
        print(loss, grad)

        loss = jnp.expand_dims(loss, axis=1)

        output = jnp.concatenate([loss, grad], axis=1)
        output = jnp.nan_to_num(output).clip(-self.clip_output, self.clip_output)

        output = self.aggregration_impl(output)

        output = jnp.concatenate([flow_pred, t, output], axis=1)
        drift = flow_pred + self.controlled_flow_impl(output, context=None) # noqa
        # drift = flow_pred + self.controlled_flow_impl(sample=flow_pred, timesteps=t, encoder_hidden_states=context, loss_grad=output, train=train)  #this is for the conv_2d_cross_attention
        
        drift = (jnp.einsum('ab, a -> ab', drift, t[:, 0] > self.start_time) +
                 jnp.einsum('ab, a -> ab', flow_pred, t[:, 0] <= self.start_time))

        return drift, output

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


    def __call__(self, t, theta, context, train=True):

        return self.forward(t, theta, context, train=train)[0]

    def mmd(self, theta_1, target):

        simulator_rng = self.make_rng('simulator')
        # print(f"In the corrector: theta_1 shape: {theta_1.shape}, target shape: {target.shape}")
        output, _ = self.simulator_impl(theta_1, num_simulations=self.num_simulations,
                                        rng=simulator_rng, deterministic=False, )  # noqa

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

        
        loss, grad = grad_fn(theta_1, context)
        
        ##LAXMPA
        ## loss, grad = jax.lax.map(value_and_grad(self.mmd_laxmap), batch_size=6, xs=(theta_1, context))

        loss = jnp.expand_dims(loss, axis=1)

        output = jnp.concatenate([loss, grad], axis=1)
        output = jnp.nan_to_num(output).clip(-self.clip_output, self.clip_output)

        output = self.aggregration_impl(output)

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


def percintile_based_mmd(sim_norm, target_norm, ):
    """MMD using percentiles as natural scales"""
    distances = jax.vmap(lambda x: jax.vmap(lambda y: jnp.linalg.norm(x - y))(target_norm))(sim_norm)
    distance_flat = distances.flatten()

    # Use percentiles as natural scales
    # # Use percentiles as natural scales
    sigmas = jnp.array([
        jnp.percentile(distance_flat, 10),   # Fine scale
        jnp.percentile(distance_flat, 25),   # Small scale  
        jnp.percentile(distance_flat, 50),   # Medium scale (median)
        # jnp.percentile(distance_flat, 75),   # Large scale
        # jnp.percentile(distance_flat, 90),   # Very large scale
    ])
    scale_weights = jnp.ones_like(sigmas)
    mmd = jnp.sum(scale_weights * jax.vmap(lambda sigmas: compute_mmd(sim_norm, target_norm, sigmas))(sigmas)) / len(sigmas)
    return jnp.log(mmd) 


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

def stream_likelihood_mean(model_stream, obs_stream, obs_errors, ):
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
        return log_p_stream

    # Vectorize over observations
    logL_values = jax.vmap(obs_log_prob)(obs_stream, jnp.repeat(obs_errors, obs_stream.shape[0]).reshape(-1, 6))
    return jnp.mean(logL_values)



def stream_likelihood_test(model_stream, obs_stream, obs_errors, ):
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
    return jnp.mean(logL_values)


# Utility: ensure non-neg and avoid zeros
def _safe(d, eps=1e-12):
    return jnp.clip(d, a_min=eps)


def loss_js(d_target, d_sim):
    p = _safe(d_target) / jnp.sum(_safe(d_target))
    q = _safe(d_sim)   / jnp.sum(_safe(d_sim))
    m = 0.5 * (p + q)
    return 0.5 * (jnp.sum(p * (jnp.log(p) - jnp.log(m))) + jnp.sum(q * (jnp.log(q) - jnp.log(m))))


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



class CorrectorDifferentiableSimulatorOdisseo_fixposition_orbit_fitting(nn.Module):
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
        code_length = 10.0 * u.kpc
        code_mass = 1e4 * u.Msun
        code_time = 3 * u.Gyr
        self.code_units = CodeUnits(code_length, code_mass, G=1, unit_time = code_time )  
        self.mean_X = jnp.load('/export/data/vgiusepp/odisseo_data/data_fix_position_uniform_prior_TSTIT5/preprocess/mean_std_1e5_parameter.npz')['mean_theta']
        self.std_X = jnp.load('/export/data/vgiusepp/odisseo_data/data_fix_position_uniform_prior_TSTIT5/preprocess/mean_std_1e5_parameter.npz')['std_theta']

    def __call__(self, t, theta, context, train=True):

        return self.forward(t, theta, context, train=train)[0]

    def NLL(self, theta_1, target):
        stream_data = target

        # stream_data = target[(target[:, :, 1] > -100) & (target[:, :, 1] < 25)]
        coord_indices = jnp.array([2, 3, 4, 5])


        phi1_min, phi1_max = -100, 25
        phi2_min, phi2_max = -8, 2
        simulator_rng = self.make_rng('simulator')
        # print(f"In the corrector: theta_1 shape: {theta_1.shape}, target shape: {target.shape}")
        stream_coordinate_com, _ = self.simulator_impl(theta_1, num_simulations=self.num_simulations,
                                        rng=simulator_rng, deterministic=True, )  # noqa
        
        stream_coordinate_com_backward, stream_coordinate_com_forward = stream_coordinate_com[0], stream_coordinate_com[1]
        
        # Create masks for valid time steps
        mask_window_backward = (stream_coordinate_com_backward[:, 0, 1] < phi1_max) & \
                            (stream_coordinate_com_backward[:, 0, 1] > phi1_min) & \
                            (stream_coordinate_com_backward[:, 0, 2] < phi2_max) & \
                            (stream_coordinate_com_backward[:, 0, 2] > phi2_min)
        
        mask_diff_backward = jnp.ediff1d(stream_coordinate_com_backward[:, 0, 1], to_begin=1) > 0
        # New mask - True until first False appears
        mask_diff_backward = jnp.cumprod(mask_diff_backward, dtype=bool)


        mask_window_forward = (stream_coordinate_com_forward[:, 0, 1] < phi1_max) & \
                            (stream_coordinate_com_forward[:, 0, 1] > phi1_min) & \
                            (stream_coordinate_com_forward[:, 0, 2] < phi2_max) & \
                            (stream_coordinate_com_forward[:, 0, 2] > phi2_min)
        
        mask_diff_forward = jnp.ediff1d(stream_coordinate_com_forward[:, 0, 1], to_begin=-1) < 0
        mask_diff_forward = jnp.cumprod(mask_diff_forward, dtype=bool)

        mask_backward = mask_window_backward & mask_diff_backward
        mask_forward = mask_window_forward & mask_diff_forward


        def coord_backward_fill(arr_phi1, arr_coord, mask):
            arr_phi1_masked = jnp.where(mask, arr_phi1, 0.0)
            filled = jnp.where(arr_phi1_masked == 0., jnp.max(arr_phi1_masked), arr_phi1_masked)
            arr_coord_masked = jnp.where(mask, arr_coord, 0.0)
            filled_coord = jnp.where(arr_coord_masked == 0., arr_coord_masked[jnp.argmax(filled)], arr_coord_masked)
            return filled, filled_coord

        def coord_forward_fill(arr_phi1, arr_coord, mask):
            arr_phi1_masked = jnp.where(mask, arr_phi1, 0.0)
            filled = jnp.where(arr_phi1_masked == 0., jnp.min(arr_phi1_masked), arr_phi1_masked)
            arr_coord_masked = jnp.where(mask, arr_coord, 0.0)
            filled_coord = jnp.where(arr_coord_masked == 0., arr_coord_masked[jnp.argmin(filled)], arr_coord_masked)
            return filled, filled_coord

        phi1_backward_valid, coord_backward_valid = jax.vmap(lambda coordinate: coord_backward_fill(stream_coordinate_com_backward[:, 0, 1], stream_coordinate_com_backward[:, 0, coordinate], mask_backward))(coordinate=coord_indices)
        phi1_forward_valid, coord_forw_valid = jax.vmap(lambda coordinate: coord_forward_fill(stream_coordinate_com_forward[:, 0, 1], stream_coordinate_com_forward[:, 0, coordinate], mask_forward))(coordinate=coord_indices)


        def interpolate_coord_backward(coord):
            return jnp.interp(stream_data[:, 1], phi1_backward_valid[0], coord)
        
        def interpolate_coord_forward(coord):
            return jnp.interp(stream_data[:, 1], phi1_forward_valid[0][::-1], coord[::-1])
            

        # Apply interpolation to all coordinates
        interp_tracks_backward = jax.vmap(interpolate_coord_backward)(coord_backward_valid)  # Shape: (n_coords, n_data)
        interp_tracks_forward = jax.vmap(interpolate_coord_forward)(coord_forw_valid)  # Shape: (n_coords, n_data)

        # Calculate residuals for all coordinates
        data_coords = stream_data[:, coord_indices].T  # Shape: (n_coords, n_data)
        sigma = jnp.array([0.5, 10., 2., 2. ])
        # sigma = jnp.array([0.15, 5., 0.1, 0.0001]) #from albatross

        mask_correct_interpolation_backward = stream_data[:, 1] < 25
        mask_correct_interpolation_forward = stream_data[:, 1] > - 100

        # Stream data masks - which data points to use for each direction
        mask_stream_backward = stream_data[:, 1] > stream_coordinate_com_backward[0, 0, 1]
        mask_stream_forward = stream_data[:, 1] < stream_coordinate_com_forward[0, 0, 1]

        mask_evaluate_inside_track_backward = (stream_data[:, 1] < jnp.max(phi1_backward_valid)) & (stream_data[:, 1] < phi1_max)
        mask_evaluate_inside_track_forward = (stream_data[:, 1] > jnp.min(phi1_forward_valid)) & (stream_data[:, 1] > phi1_min)

        # Calculate chi2 using only the appropriate data points for each direction
        residuals_backward = jnp.where(mask_stream_backward & mask_evaluate_inside_track_backward & mask_correct_interpolation_backward, 
                                    (data_coords - interp_tracks_backward)/sigma[:, None],
                                    1.)
        residuals_forward = jnp.where(mask_stream_forward & mask_evaluate_inside_track_forward & mask_correct_interpolation_forward, 
                                    (data_coords - interp_tracks_forward)/sigma[:, None],
                                    1.)
        residuals = jnp.where(mask_stream_backward,
                            residuals_backward,
                            residuals_forward)
        # Masks for valid residuals
        mask_backward_full = mask_stream_backward & mask_evaluate_inside_track_backward & mask_correct_interpolation_backward
        mask_forward_full = mask_stream_forward & mask_evaluate_inside_track_forward & mask_correct_interpolation_forward
        
        # Count valid data points PER COORDINATE
        mask_used = jnp.where(mask_stream_backward[:, None],
                            mask_backward_full[:, None],  # Broadcast to (n_data, 1)
                            mask_forward_full[:, None])
        
        # Total number of valid measurements across all coordinates
        n_valid_measurements = jnp.sum(mask_used)  # Counts True values in (n_data, 4) array
        
        # Number of free parameters being fit
        n_params = theta_1.shape[-1]  # Should be 7 for your model
        
        # Degrees of freedom
        dof = n_valid_measurements - n_params
        
        # Ensure DOF is positive (if not enough data, penalize heavily)
        dof = jnp.maximum(dof, 1.0)  # Avoid division by zero
        
        # Chi-squared (sum of squared residuals)
        chi2 = jnp.sum(residuals**2)
        
        # Reduced chi-squared
        chi2_reduced = chi2 / dof
        
        return chi2_reduced
       


    def forward_flow(self, t, theta, context, train=False):

        # we need this because self.model was trained with stacked flow which has additional time dimensions
        return self.model(t, theta, context, train=train)

    def forward(self, t, theta, context, train=True):

        # predict flow
        flow_pred = self.model(t, theta, context, train=train)

        if self.freeze:
            flow_pred = stop_gradient(flow_pred)

        # print(theta.shape)

        theta_1 = theta + jnp.einsum('ab,a->ab', flow_pred, 1 - t[:, 0])
        

        theta_1 = theta_1 * self.std_X + self.mean_X

        # print(f"In the corrector (should have a batch_dimension): theta_1 shape: {theta_1.shape}, context shape: {context.shape}")
        grad_fn = vmap(value_and_grad(self.NLL), in_axes=0)
        # context = context[(context[:, :, 1] > -100) & (context[:, :, 1] < 25)]
        loss, grad = grad_fn(theta_1, context)

        loss = jnp.expand_dims(loss, axis=1)
        loss = jnp.log10(loss/1e3)
        print('loss:', loss)
        print('grad:', grad)

        output = jnp.concatenate([loss, grad], axis=1)
        output = jnp.nan_to_num(output).clip(-self.clip_output, self.clip_output)

        output = self.aggregration_impl(output)

        output = jnp.concatenate([flow_pred, t, output], axis=1)
        # drift = flow_pred + self.controlled_flow_impl(theta=flow_pred, context=output, t=t) # noqa
        drift = flow_pred + self.controlled_flow_impl(output, context=None) # noqa
        # drift = flow_pred + self.controlled_flow_impl(sample=flow_pred, timesteps=t, encoder_hidden_states=context, loss_grad=output, train=train)  #this is for the conv_2d_cross_attention
        
        drift = (jnp.einsum('ab, a -> ab', drift, t[:, 0] > self.start_time) +
                 jnp.einsum('ab, a -> ab', flow_pred, t[:, 0] <= self.start_time))

        return drift, output


# class CorrectorDifferentiableSimulatorOdisseo_fixposition_orbit_fitting_andpointcloud(nn.Module):
#     """
#     Corrector model for Odisseo simulator, differentiable version.
#     """

#     model: nn.Module
#     simulator: dict
#     controlled_flow: dict
#     aggregation: dict
#     freeze: bool = True
#     layer_norm: bool = False
#     start_time: float = 1.0
#     num_simulations: int = 1
#     clip_output: float = 10.0
#     sharding: bool = False

#     def setup(self):

#         self.simulator_impl: OdisseoSimulator = instantiate_from_config(self.simulator)
#         self.controlled_flow_impl: nn.Module = instantiate_from_config(self.controlled_flow)
#         self.aggregration_impl: nn.Module = instantiate_from_config(self.aggregation)
#         code_length = 10.0 * u.kpc
#         code_mass = 1e4 * u.Msun
#         code_time = 3 * u.Gyr
#         self.code_units = CodeUnits(code_length, code_mass, G=1, unit_time = code_time )  
#         self.mean_X = jnp.load('/export/data/vgiusepp/odisseo_data/data_fix_position_uniform_prior_TSTIT5/preprocess/mean_std_1e5_parameter.npz')['mean_theta']
#         self.std_X = jnp.load('/export/data/vgiusepp/odisseo_data/data_fix_position_uniform_prior_TSTIT5/preprocess/mean_std_1e5_parameter.npz')['std_theta']

#     def __call__(self, t, theta, context, train=True):

#         return self.forward(t, theta, context, train=train)[0]

#     def NLL(self, theta_1, target):
#         theta_1 = 10**theta_1
#         theta_1 = theta_1.at[0].set(theta_1[0] * u.Msun.to(self.code_units.code_mass))
#         theta_1 = theta_1.at[1].set(theta_1[1] * u.kpc.to(self.code_units.code_length))
#         theta_1 = theta_1.at[2].set(theta_1[2] * u.Msun.to(self.code_units.code_mass))
#         theta_1 = theta_1.at[3].set(theta_1[3] * u.kpc.to(self.code_units.code_length))
#         theta_1 = theta_1.at[4].set(theta_1[4] * u.kpc.to(self.code_units.code_length))
#         theta_1 = theta_1.at[5].set(theta_1[5] * u.Msun.to(self.code_units.code_mass))
#         theta_1 = theta_1.at[6].set(theta_1[6] * u.kpc.to(self.code_units.code_length))
#         stream_data = target
#         coord_indices = jnp.array([2, 3, 4, 5])

#         phi1_min, phi1_max = -100, 25
#         phi2_min, phi2_max = -8, 2
#         simulator_rng = self.make_rng('simulator')
#         # print(f"In the corrector: theta_1 shape: {theta_1.shape}, target shape: {target.shape}")
#         stream_coordinate_com, _ = self.simulator_impl(theta_1, num_simulations=self.num_simulations,
#                                         rng=simulator_rng, deterministic=True, stream=False)  
        
#         stream_coordinate_com_backward, stream_coordinate_com_forward = stream_coordinate_com[0], stream_coordinate_com[1]
        
#         # Create masks for valid time steps
#         mask_window_backward = (stream_coordinate_com_backward[:, 0, 1] < phi1_max) & \
#                             (stream_coordinate_com_backward[:, 0, 1] > phi1_min) & \
#                             (stream_coordinate_com_backward[:, 0, 2] < phi2_max) & \
#                             (stream_coordinate_com_backward[:, 0, 2] > phi2_min)
        
#         mask_diff_backward = jnp.ediff1d(stream_coordinate_com_backward[:, 0, 1], to_begin=1) > 0
#         # New mask - True until first False appears
#         mask_diff_backward = jnp.cumprod(mask_diff_backward, dtype=bool)


#         mask_window_forward = (stream_coordinate_com_forward[:, 0, 1] < phi1_max) & \
#                             (stream_coordinate_com_forward[:, 0, 1] > phi1_min) & \
#                             (stream_coordinate_com_forward[:, 0, 2] < phi2_max) & \
#                             (stream_coordinate_com_forward[:, 0, 2] > phi2_min)
        
#         mask_diff_forward = jnp.ediff1d(stream_coordinate_com_forward[:, 0, 1], to_begin=-1) < 0
#         mask_diff_forward = jnp.cumprod(mask_diff_forward, dtype=bool)

#         mask_backward = mask_window_backward & mask_diff_backward
#         mask_forward = mask_window_forward & mask_diff_forward


#         def coord_backward_fill(arr_phi1, arr_coord, mask):
#             arr_phi1_masked = jnp.where(mask, arr_phi1, 0.0)
#             filled = jnp.where(arr_phi1_masked == 0., jnp.max(arr_phi1_masked), arr_phi1_masked)
#             arr_coord_masked = jnp.where(mask, arr_coord, 0.0)
#             filled_coord = jnp.where(arr_coord_masked == 0., arr_coord_masked[jnp.argmax(filled)], arr_coord_masked)
#             return filled, filled_coord

#         def coord_forward_fill(arr_phi1, arr_coord, mask):
#             arr_phi1_masked = jnp.where(mask, arr_phi1, 0.0)
#             filled = jnp.where(arr_phi1_masked == 0., jnp.min(arr_phi1_masked), arr_phi1_masked)
#             arr_coord_masked = jnp.where(mask, arr_coord, 0.0)
#             filled_coord = jnp.where(arr_coord_masked == 0., arr_coord_masked[jnp.argmin(filled)], arr_coord_masked)
#             return filled, filled_coord

#         phi1_backward_valid, coord_backward_valid = jax.vmap(lambda coordinate: coord_backward_fill(stream_coordinate_com_backward[:, 0, 1], stream_coordinate_com_backward[:, 0, coordinate], mask_backward))(coordinate=coord_indices)
#         phi1_forward_valid, coord_forw_valid = jax.vmap(lambda coordinate: coord_forward_fill(stream_coordinate_com_forward[:, 0, 1], stream_coordinate_com_forward[:, 0, coordinate], mask_forward))(coordinate=coord_indices)


#         def interpolate_coord_backward(coord):
#             return jnp.interp(stream_data[:, 1], phi1_backward_valid[0], coord)
        
#         def interpolate_coord_forward(coord):
#             return jnp.interp(stream_data[:, 1], phi1_forward_valid[0][::-1], coord[::-1])
            

#         # Apply interpolation to all coordinates
#         interp_tracks_backward = jax.vmap(interpolate_coord_backward)(coord_backward_valid)  # Shape: (n_coords, n_data)
#         interp_tracks_forward = jax.vmap(interpolate_coord_forward)(coord_forw_valid)  # Shape: (n_coords, n_data)

#         # Calculate residuals for all coordinates
#         data_coords = stream_data[:, coord_indices].T  # Shape: (n_coords, n_data)
#         sigma = jnp.array([0.5, 10., 2., 2. ])
#         # sigma = jnp.array([0.15, 5., 0.1, 0.0001]) #from albatross

#         mask_correct_interpolation_backward = stream_data[:, 1] < 25
#         mask_correct_interpolation_forward = stream_data[:, 1] > - 100

#         # Stream data masks - which data points to use for each direction
#         mask_stream_backward = stream_data[:, 1] > stream_coordinate_com_backward[0, 0, 1]
#         mask_stream_forward = stream_data[:, 1] < stream_coordinate_com_forward[0, 0, 1]

#         mask_evaluate_inside_track_backward = (stream_data[:, 1] < jnp.max(phi1_backward_valid)) & (stream_data[:, 1] < phi1_max)
#         mask_evaluate_inside_track_forward = (stream_data[:, 1] > jnp.min(phi1_forward_valid)) & (stream_data[:, 1] > phi1_min)

#         # Calculate chi2 using only the appropriate data points for each direction
#         residuals_backward = jnp.where(mask_stream_backward & mask_evaluate_inside_track_backward & mask_correct_interpolation_backward, 
#                                     (data_coords - interp_tracks_backward)/sigma[:, None],
#                                     1.)
#         residuals_forward = jnp.where(mask_stream_forward & mask_evaluate_inside_track_forward & mask_correct_interpolation_forward, 
#                                     (data_coords - interp_tracks_forward)/sigma[:, None],
#                                     1.)
#         residuals = jnp.where(mask_stream_backward,
#                             residuals_backward,
#                             residuals_forward)
#         # Masks for valid residuals
#         mask_backward_full = mask_stream_backward & mask_evaluate_inside_track_backward & mask_correct_interpolation_backward
#         mask_forward_full = mask_stream_forward & mask_evaluate_inside_track_forward & mask_correct_interpolation_forward
        
#         # Count valid data points PER COORDINATE
#         mask_used = jnp.where(mask_stream_backward[:, None],
#                             mask_backward_full[:, None],  # Broadcast to (n_data, 1)
#                             mask_forward_full[:, None])
        
#         # Total number of valid measurements across all coordinates
#         n_valid_measurements = jnp.sum(mask_used)  # Counts True values in (n_data, 4) array
        
#         # Number of free parameters being fit
#         n_params = theta_1.shape[-1]  # Should be 7 for your model
        
#         # Degrees of freedom
#         dof = n_valid_measurements  #- n_params
        
#         # Ensure DOF is positive (if not enough data, penalize heavily)
#         dof = jnp.maximum(dof, 1.0)  # Avoid division by zero
        
#         # Chi-squared (sum of squared residuals)
#         chi2 = jnp.sum(residuals**2)
        
#         # Reduced chi-squared
#         chi2_reduced = chi2 / dof
        
#         return jnp.log10(chi2)
       


#     def forward_flow(self, t, theta, context, train=False):

#         # we need this because self.model was trained with stacked flow which has additional time dimensions
#         return self.model(t, theta, context, train=train)

#     def forward(self, t, theta, context, train=True):

#         # predict flow
#         flow_pred = self.model(t, theta, context, train=train)

#         if self.freeze:
#             flow_pred = stop_gradient(flow_pred)

#         # print(theta.shape)

#         theta_1 = theta + jnp.einsum('ab,a->ab', flow_pred, 1 - t[:, 0])
        

#         theta_1_sim = theta_1 * self.std_X + self.mean_X

#         # print(f"In the corrector (should have a batch_dimension): theta_1 shape: {theta_1.shape}, context shape: {context.shape}")
#         grad_fn = vmap(value_and_grad(self.NLL), in_axes=0)
#         loss, grad = grad_fn(theta_1_sim, context)

#         loss = jnp.expand_dims(loss, axis=1)
#         print(loss)
#         print(grad)

#         # jax.debug.print('loss: {value}', value = jnp.isnan(loss).sum())
#         # jax.debug.print('grad: {value}', value = jnp.isnan(grad).sum())

#         output = jnp.concatenate([loss, grad], axis=1)
#         output = jnp.nan_to_num(output).clip(-self.clip_output, self.clip_output)


#         simulator_rng = self.make_rng('simulator')
#         stream, _ = jax.vmap(lambda theta: self.simulator_impl(theta, num_simulations=self.num_simulations,
#                                         rng=simulator_rng, deterministic=True, stream=True))(theta_1_sim)     
#         # jax.debug.print('stream: {value}', value = jnp.isnan(stream).sum())

#         drift = flow_pred + self.controlled_flow_impl(sample=theta_1, timesteps=t, encoder_hidden_states=stream, loss_grad=output, ) 
#         # errors = checkify.user_checks | checkify.index_checks | checkify.float_checks | checkify.nan_checks
#         # controlled_flow_to_be_checked = lambda  flow_pred, t, stream, output: self.controlled_flow_impl(sample=flow_pred, timesteps=t, encoder_hidden_states=stream, loss_grad=output, )
        
#         # checkify.checkify(controlled_flow_to_be_checked, errors=errors)(flow_pred, t, stream, output)
#         # drift = flow_pred + checkify.checkify(controlled_flow_to_be_checked, errors=errors)(flow_pred, t, stream, output)[1]


#         drift = (jnp.einsum('ab, a -> ab', drift, t[:, 0] > self.start_time) +
#                  jnp.einsum('ab, a -> ab', flow_pred, t[:, 0] <= self.start_time))

#         return drift, output
    


class CorrectorDifferentiableSimulatorOdisseo_fixposition_orbit_fitting_andpointcloud(nn.Module):
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
        code_length = 10.0 * u.kpc
        code_mass = 1e4 * u.Msun
        code_time = 3 * u.Gyr
        self.code_units = CodeUnits(code_length, code_mass, G=1, unit_time = code_time )  
        self.mean_X = jnp.load('/export/data/vgiusepp/odisseo_data/data_fix_position_uniform_prior_TSTIT5/preprocess/mean_std_1e5_parameter.npz')['mean_theta']
        self.std_X = jnp.load('/export/data/vgiusepp/odisseo_data/data_fix_position_uniform_prior_TSTIT5/preprocess/mean_std_1e5_parameter.npz')['std_theta']

    def __call__(self, t, theta, context, train=True):

        return self.forward(t, theta, context, train=train)[0]

    def NLL(self, theta_1, target):
        theta_1 = 10**theta_1
        theta_1 = theta_1.at[0].set(theta_1[0] * u.Msun.to(self.code_units.code_mass))
        theta_1 = theta_1.at[1].set(theta_1[1] * u.kpc.to(self.code_units.code_length))
        theta_1 = theta_1.at[2].set(theta_1[2] * u.Msun.to(self.code_units.code_mass))
        theta_1 = theta_1.at[3].set(theta_1[3] * u.kpc.to(self.code_units.code_length))
        theta_1 = theta_1.at[4].set(theta_1[4] * u.kpc.to(self.code_units.code_length))
        theta_1 = theta_1.at[5].set(theta_1[5] * u.Msun.to(self.code_units.code_mass))
        theta_1 = theta_1.at[6].set(theta_1[6] * u.kpc.to(self.code_units.code_length))
        stream_data = target
        coord_indices = jnp.array([2, 3, 4, 5])

        phi1_min, phi1_max = -90, 10
        phi2_min, phi2_max = -8, 2
        simulator_rng = self.make_rng('simulator')
        # print(f"In the corrector: theta_1 shape: {theta_1.shape}, target shape: {target.shape}")
        stream_coordinate_com, _ = self.simulator_impl(theta_1, num_simulations=self.num_simulations,
                                        rng=simulator_rng, deterministic=True, stream=False)  
        
        stream_coordinate_com_backward, stream_coordinate_com_forward = stream_coordinate_com[0], stream_coordinate_com[1]
        
        # Create masks for valid time steps
        mask_window_backward = (stream_coordinate_com_backward[:, 0, 1] < phi1_max) & \
                            (stream_coordinate_com_backward[:, 0, 1] > phi1_min) & \
                            (stream_coordinate_com_backward[:, 0, 2] < phi2_max) & \
                            (stream_coordinate_com_backward[:, 0, 2] > phi2_min)
        
        mask_diff_backward = jnp.ediff1d(stream_coordinate_com_backward[:, 0, 1], to_begin=1) > 0
        # New mask - True until first False appears
        mask_diff_backward = jnp.cumprod(mask_diff_backward, dtype=bool)


        mask_window_forward = (stream_coordinate_com_forward[:, 0, 1] < phi1_max) & \
                            (stream_coordinate_com_forward[:, 0, 1] > phi1_min) & \
                            (stream_coordinate_com_forward[:, 0, 2] < phi2_max) & \
                            (stream_coordinate_com_forward[:, 0, 2] > phi2_min)
        
        mask_diff_forward = jnp.ediff1d(stream_coordinate_com_forward[:, 0, 1], to_begin=-1) < 0
        mask_diff_forward = jnp.cumprod(mask_diff_forward, dtype=bool)

        mask_backward = mask_window_backward & mask_diff_backward
        mask_forward = mask_window_forward & mask_diff_forward


        def coord_backward_fill(arr_phi1, arr_coord, mask):
            arr_phi1_masked = jnp.where(mask, arr_phi1, 0.0)
            filled = jnp.where(arr_phi1_masked == 0., jnp.max(arr_phi1_masked), arr_phi1_masked)
            arr_coord_masked = jnp.where(mask, arr_coord, 0.0)
            filled_coord = jnp.where(arr_coord_masked == 0., arr_coord_masked[jnp.argmax(filled)], arr_coord_masked)
            return filled, filled_coord

        def coord_forward_fill(arr_phi1, arr_coord, mask):
            arr_phi1_masked = jnp.where(mask, arr_phi1, 0.0)
            filled = jnp.where(arr_phi1_masked == 0., jnp.min(arr_phi1_masked), arr_phi1_masked)
            arr_coord_masked = jnp.where(mask, arr_coord, 0.0)
            filled_coord = jnp.where(arr_coord_masked == 0., arr_coord_masked[jnp.argmin(filled)], arr_coord_masked)
            return filled, filled_coord

        phi1_backward_valid, coord_backward_valid = jax.vmap(lambda coordinate: coord_backward_fill(stream_coordinate_com_backward[:, 0, 1], stream_coordinate_com_backward[:, 0, coordinate], mask_backward))(coordinate=coord_indices)
        phi1_forward_valid, coord_forw_valid = jax.vmap(lambda coordinate: coord_forward_fill(stream_coordinate_com_forward[:, 0, 1], stream_coordinate_com_forward[:, 0, coordinate], mask_forward))(coordinate=coord_indices)


        def interpolate_coord_backward(coord):
            return jnp.interp(stream_data[:, 1], phi1_backward_valid[0], coord)
        
        def interpolate_coord_forward(coord):
            return jnp.interp(stream_data[:, 1], phi1_forward_valid[0][::-1], coord[::-1])
            

        # Apply interpolation to all coordinates
        interp_tracks_backward = jax.vmap(interpolate_coord_backward)(coord_backward_valid)  # Shape: (n_coords, n_data)
        interp_tracks_forward = jax.vmap(interpolate_coord_forward)(coord_forw_valid)  # Shape: (n_coords, n_data)

        # Calculate residuals for all coordinates
        data_coords = stream_data[:, coord_indices].T  # Shape: (n_coords, n_data)
        sigma = jnp.array([0.5, 10., 2., 2. ])
        # sigma = jnp.array([0.15, 5., 0.1, 0.0001]) #from albatross

        mask_correct_interpolation_backward = stream_data[:, 1] < 10
        mask_correct_interpolation_forward = stream_data[:, 1] > - 90

        # Stream data masks - which data points to use for each direction
        mask_stream_backward = stream_data[:, 1] > stream_coordinate_com_backward[0, 0, 1]
        mask_stream_forward = stream_data[:, 1] < stream_coordinate_com_forward[0, 0, 1]

        mask_evaluate_inside_track_backward = (stream_data[:, 1] < jnp.max(phi1_backward_valid)) & (stream_data[:, 1] < phi1_max)
        mask_evaluate_inside_track_forward = (stream_data[:, 1] > jnp.min(phi1_forward_valid)) & (stream_data[:, 1] > phi1_min)

        # Calculate chi2 using only the appropriate data points for each direction
        residuals_backward = jnp.where(mask_stream_backward & mask_evaluate_inside_track_backward & mask_correct_interpolation_backward, 
                                    (data_coords - interp_tracks_backward)/sigma[:, None],
                                    0.)
        residuals_forward = jnp.where(mask_stream_forward & mask_evaluate_inside_track_forward & mask_correct_interpolation_forward, 
                                    (data_coords - interp_tracks_forward)/sigma[:, None],
                                    0.)
        residuals = jnp.where(mask_stream_backward,
                            residuals_backward,
                            residuals_forward)
        # Masks for valid residuals
        mask_backward_full = mask_stream_backward & mask_evaluate_inside_track_backward & mask_correct_interpolation_backward
        mask_forward_full = mask_stream_forward & mask_evaluate_inside_track_forward & mask_correct_interpolation_forward
        
        # Count valid data points PER COORDINATE
        mask_used = jnp.where(mask_stream_backward[:, None],
                            mask_backward_full[:, None],  # Broadcast to (n_data, 1)
                            mask_forward_full[:, None])
        
        
        
        # Chi-squared (sum of squared residuals)
        chi2 = jnp.sum(residuals**2)
        
        # Reduced chi-squared
        
        return chi2
       


    def forward_flow(self, t, theta, context, train=False):

        # we need this because self.model was trained with stacked flow which has additional time dimensions
        return self.model(t, theta, context, train=train)

    def forward(self, t, theta, context, train=True):

        # predict flow
        flow_pred = self.model(t, theta, context, train=train)

        if self.freeze:
            flow_pred = stop_gradient(flow_pred)

        # print(theta.shape)

        theta_1 = theta + jnp.einsum('ab,a->ab', flow_pred, 1 - t[:, 0])
        

        theta_1_sim = theta_1 * self.std_X + self.mean_X

        # print(f"In the corrector (should have a batch_dimension): theta_1 shape: {theta_1.shape}, context shape: {context.shape}")
        grad_fn = vmap(value_and_grad(self.NLL), in_axes=0)
        loss, grad = grad_fn(theta_1_sim, context)

        loss = jnp.expand_dims(loss, axis=1)
        # print(loss)
        # print(grad)

        # jax.debug.print('loss: {value}', value = jnp.isnan(loss).sum())
        # jax.debug.print('grad: {value}', value = jnp.isnan(grad).sum())

        output = jnp.concatenate([loss, grad], axis=1)
        output = jnp.nan_to_num(output).clip(-self.clip_output, self.clip_output)


        simulator_rng = self.make_rng('simulator')
        stream, _ = jax.vmap(lambda theta: self.simulator_impl(theta, num_simulations=self.num_simulations,
                                        rng=simulator_rng, deterministic=True, stream=True))(theta_1_sim)     
        # jax.debug.print('stream: {value}', value = jnp.isnan(stream).sum())

        drift = flow_pred + self.controlled_flow_impl(sample=flow_pred, timesteps=t, encoder_hidden_states=stream, loss_grad=output, ) 
        # errors = checkify.user_checks | checkify.index_checks | checkify.float_checks | checkify.nan_checks
        # controlled_flow_to_be_checked = lambda  flow_pred, t, stream, output: self.controlled_flow_impl(sample=flow_pred, timesteps=t, encoder_hidden_states=stream, loss_grad=output, )
        
        # checkify.checkify(controlled_flow_to_be_checked, errors=errors)(flow_pred, t, stream, output)
        # drift = flow_pred + checkify.checkify(controlled_flow_to_be_checked, errors=errors)(flow_pred, t, stream, output)[1]


        drift = (jnp.einsum('ab, a -> ab', drift, t[:, 0] > self.start_time) +
                 jnp.einsum('ab, a -> ab', flow_pred, t[:, 0] <= self.start_time))

        return drift, output
    
class CorrectorDifferentiableSimulatorOdisseoPositions_fixtime_orbit_fitting_andpointcloud(nn.Module):
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
        code_length = 10.0 * u.kpc
        code_mass = 1e4 * u.Msun
        code_time = 3 * u.Gyr
        self.code_units = CodeUnits(code_length, code_mass, G=1, unit_time = code_time )  
        self.mean_X = jnp.load('/export/data/vgiusepp/odisseo_data/data_varying_position_fixed_time_uniform_prior_TSTIT5/preprocess/mean_std_1e5_parameter.npz')['mean_theta']
        self.std_X = jnp.load('/export/data/vgiusepp/odisseo_data/data_varying_position_fixed_time_uniform_prior_TSTIT5/preprocess/mean_std_1e5_parameter.npz')['std_theta']

    def __call__(self, t, theta, context, train=True):

        return self.forward(t, theta, context, train=train)[0]

    def NLL(self, theta_1, target):
        theta_1 = theta_1.at[0].set(10**theta_1[0] * u.Msun.to(self.code_units.code_mass))
        theta_1 = theta_1.at[1].set(10**theta_1[1] * u.kpc.to(self.code_units.code_length))
        theta_1 = theta_1.at[2].set(10**theta_1[2] * u.Msun.to(self.code_units.code_mass))
        theta_1 = theta_1.at[3].set(10**theta_1[3] * u.kpc.to(self.code_units.code_length))
        theta_1 = theta_1.at[4].set(10**theta_1[4] * u.kpc.to(self.code_units.code_length))
        theta_1 = theta_1.at[5].set(10**theta_1[5] * u.Msun.to(self.code_units.code_mass))
        theta_1 = theta_1.at[6].set(10**theta_1[6] * u.kpc.to(self.code_units.code_length))
        stream_data = target
        coord_indices = jnp.array([2, 3, 4, 5])

        phi1_min, phi1_max = -90, 10
        phi2_min, phi2_max = -8, 2
        simulator_rng = self.make_rng('simulator')
        # print(f"In the corrector: theta_1 shape: {theta_1.shape}, target shape: {target.shape}")
        stream_coordinate_com, _ = self.simulator_impl(theta_1, num_simulations=self.num_simulations,
                                        rng=simulator_rng, deterministic=True, stream=False)  
        
        stream_coordinate_com_backward, stream_coordinate_com_forward = stream_coordinate_com[0], stream_coordinate_com[1]
        
        # Create masks for valid time steps
        mask_window_backward = (stream_coordinate_com_backward[:, 0, 1] < phi1_max) & \
                            (stream_coordinate_com_backward[:, 0, 1] > phi1_min) & \
                            (stream_coordinate_com_backward[:, 0, 2] < phi2_max) & \
                            (stream_coordinate_com_backward[:, 0, 2] > phi2_min)
        
        mask_diff_backward = jnp.ediff1d(stream_coordinate_com_backward[:, 0, 1], to_begin=1) > 0
        # New mask - True until first False appears
        mask_diff_backward = jnp.cumprod(mask_diff_backward, dtype=bool)


        mask_window_forward = (stream_coordinate_com_forward[:, 0, 1] < phi1_max) & \
                            (stream_coordinate_com_forward[:, 0, 1] > phi1_min) & \
                            (stream_coordinate_com_forward[:, 0, 2] < phi2_max) & \
                            (stream_coordinate_com_forward[:, 0, 2] > phi2_min)
        
        mask_diff_forward = jnp.ediff1d(stream_coordinate_com_forward[:, 0, 1], to_begin=-1) < 0
        mask_diff_forward = jnp.cumprod(mask_diff_forward, dtype=bool)

        mask_backward = mask_window_backward & mask_diff_backward
        mask_forward = mask_window_forward & mask_diff_forward


        def coord_backward_fill(arr_phi1, arr_coord, mask):
            arr_phi1_masked = jnp.where(mask, arr_phi1, 0.0)
            filled = jnp.where(arr_phi1_masked == 0., jnp.max(arr_phi1_masked), arr_phi1_masked)
            arr_coord_masked = jnp.where(mask, arr_coord, 0.0)
            filled_coord = jnp.where(arr_coord_masked == 0., arr_coord_masked[jnp.argmax(filled)], arr_coord_masked)
            return filled, filled_coord

        def coord_forward_fill(arr_phi1, arr_coord, mask):
            arr_phi1_masked = jnp.where(mask, arr_phi1, 0.0)
            filled = jnp.where(arr_phi1_masked == 0., jnp.min(arr_phi1_masked), arr_phi1_masked)
            arr_coord_masked = jnp.where(mask, arr_coord, 0.0)
            filled_coord = jnp.where(arr_coord_masked == 0., arr_coord_masked[jnp.argmin(filled)], arr_coord_masked)
            return filled, filled_coord

        phi1_backward_valid, coord_backward_valid = jax.vmap(lambda coordinate: coord_backward_fill(stream_coordinate_com_backward[:, 0, 1], stream_coordinate_com_backward[:, 0, coordinate], mask_backward))(coordinate=coord_indices)
        phi1_forward_valid, coord_forw_valid = jax.vmap(lambda coordinate: coord_forward_fill(stream_coordinate_com_forward[:, 0, 1], stream_coordinate_com_forward[:, 0, coordinate], mask_forward))(coordinate=coord_indices)


        def interpolate_coord_backward(coord):
            return jnp.interp(stream_data[:, 1], phi1_backward_valid[0], coord)
        
        def interpolate_coord_forward(coord):
            return jnp.interp(stream_data[:, 1], phi1_forward_valid[0][::-1], coord[::-1])
            

        # Apply interpolation to all coordinates
        interp_tracks_backward = jax.vmap(interpolate_coord_backward)(coord_backward_valid)  # Shape: (n_coords, n_data)
        interp_tracks_forward = jax.vmap(interpolate_coord_forward)(coord_forw_valid)  # Shape: (n_coords, n_data)

        # Calculate residuals for all coordinates
        data_coords = stream_data[:, coord_indices].T  # Shape: (n_coords, n_data)
        sigma = jnp.array([0.5, 10., 2., 2. ])
        # sigma = jnp.array([0.15, 5., 0.1, 0.0001]) #from albatross

        mask_correct_interpolation_backward = stream_data[:, 1] < 10
        mask_correct_interpolation_forward = stream_data[:, 1] > - 90

        # Stream data masks - which data points to use for each direction
        mask_stream_backward = stream_data[:, 1] > stream_coordinate_com_backward[0, 0, 1]
        mask_stream_forward = stream_data[:, 1] < stream_coordinate_com_forward[0, 0, 1]

        mask_evaluate_inside_track_backward = (stream_data[:, 1] < jnp.max(phi1_backward_valid)) & (stream_data[:, 1] < phi1_max)
        mask_evaluate_inside_track_forward = (stream_data[:, 1] > jnp.min(phi1_forward_valid)) & (stream_data[:, 1] > phi1_min)

        # Calculate chi2 using only the appropriate data points for each direction
        residuals_backward = jnp.where(mask_stream_backward & mask_evaluate_inside_track_backward & mask_correct_interpolation_backward, 
                                    (data_coords - interp_tracks_backward)/sigma[:, None],
                                    0.)
        residuals_forward = jnp.where(mask_stream_forward & mask_evaluate_inside_track_forward & mask_correct_interpolation_forward, 
                                    (data_coords - interp_tracks_forward)/sigma[:, None],
                                    0.)
        residuals = jnp.where(mask_stream_backward,
                            residuals_backward,
                            residuals_forward)
        # Masks for valid residuals
        mask_backward_full = mask_stream_backward & mask_evaluate_inside_track_backward & mask_correct_interpolation_backward
        mask_forward_full = mask_stream_forward & mask_evaluate_inside_track_forward & mask_correct_interpolation_forward
        
        # Count valid data points PER COORDINATE
        mask_used = jnp.where(mask_stream_backward[:, None],
                            mask_backward_full[:, None],  # Broadcast to (n_data, 1)
                            mask_forward_full[:, None])
        
        
        
        # Chi-squared (sum of squared residuals)
        chi2 = jnp.sum(residuals**2)
        
        # Reduced chi-squared
        
        return chi2
       


    def forward_flow(self, t, theta, context, train=False):

        # we need this because self.model was trained with stacked flow which has additional time dimensions
        return self.model(t, theta, context, train=train)

    def forward(self, t, theta, context, train=True):

        # predict flow
        flow_pred = self.model(t, theta, context, train=train)

        if self.freeze:
            flow_pred = stop_gradient(flow_pred)

        # print(theta.shape)

        theta_1 = theta + jnp.einsum('ab,a->ab', flow_pred, 1 - t[:, 0])
        

        theta_1_sim = theta_1 * self.std_X + self.mean_X

        # print(f"In the corrector (should have a batch_dimension): theta_1 shape: {theta_1.shape}, context shape: {context.shape}")
        grad_fn = vmap(value_and_grad(self.NLL), in_axes=0)
        loss, grad = grad_fn(theta_1_sim, context)

        loss = jnp.expand_dims(loss, axis=1)
        # print(loss)
        # print(grad)

        # jax.debug.print('loss: {value}', value = jnp.isnan(loss).sum())
        # jax.debug.print('grad: {value}', value = jnp.isnan(grad).sum())

        output = jnp.concatenate([loss, grad], axis=1)
        output = jnp.nan_to_num(output).clip(-self.clip_output, self.clip_output)


        simulator_rng = self.make_rng('simulator')
        stream, _ = jax.vmap(lambda theta: self.simulator_impl(theta, num_simulations=self.num_simulations,
                                        rng=simulator_rng, deterministic=True, stream=True))(theta_1_sim)     
        # jax.debug.print('stream: {value}', value = jnp.isnan(stream).sum())

        drift = flow_pred + self.controlled_flow_impl(sample=flow_pred, timesteps=t, encoder_hidden_states=stream, loss_grad=output, ) 
        # errors = checkify.user_checks | checkify.index_checks | checkify.float_checks | checkify.nan_checks
        # controlled_flow_to_be_checked = lambda  flow_pred, t, stream, output: self.controlled_flow_impl(sample=flow_pred, timesteps=t, encoder_hidden_states=stream, loss_grad=output, )
        
        # checkify.checkify(controlled_flow_to_be_checked, errors=errors)(flow_pred, t, stream, output)
        # drift = flow_pred + checkify.checkify(controlled_flow_to_be_checked, errors=errors)(flow_pred, t, stream, output)[1]


        drift = (jnp.einsum('ab, a -> ab', drift, t[:, 0] > self.start_time) +
                 jnp.einsum('ab, a -> ab', flow_pred, t[:, 0] <= self.start_time))

        return drift, output
    
class CorrectorDifferentiableSimulatorOdisseoPositions_fixtime_orbit_fitting(nn.Module):
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
        code_length = 10.0 * u.kpc
        code_mass = 1e4 * u.Msun
        code_time = 3 * u.Gyr
        self.code_units = CodeUnits(code_length, code_mass, G=1, unit_time = code_time )  
        self.mean_X = jnp.load('/export/data/vgiusepp/odisseo_data/data_varying_position_fixed_time_uniform_prior_TSTIT5/preprocess/mean_std_1e5_parameter.npz')['mean_theta']
        self.std_X = jnp.load('/export/data/vgiusepp/odisseo_data/data_varying_position_fixed_time_uniform_prior_TSTIT5/preprocess/mean_std_1e5_parameter.npz')['std_theta']

    def __call__(self, t, theta, context, train=True):

        return self.forward(t, theta, context, train=train)[0]

    def NLL(self, theta_1, target):
        theta_1 = theta_1.at[0].set(10**theta_1[0] * u.Msun.to(self.code_units.code_mass))
        theta_1 = theta_1.at[1].set(10**theta_1[1] * u.kpc.to(self.code_units.code_length))
        theta_1 = theta_1.at[2].set(10**theta_1[2] * u.Msun.to(self.code_units.code_mass))
        theta_1 = theta_1.at[3].set(10**theta_1[3] * u.kpc.to(self.code_units.code_length))
        theta_1 = theta_1.at[4].set(10**theta_1[4] * u.kpc.to(self.code_units.code_length))
        theta_1 = theta_1.at[5].set(10**theta_1[5] * u.Msun.to(self.code_units.code_mass))
        theta_1 = theta_1.at[6].set(10**theta_1[6] * u.kpc.to(self.code_units.code_length))
        stream_data = target
        coord_indices = jnp.array([2, 3, 4, 5])

        phi1_min, phi1_max = -90, 10
        phi2_min, phi2_max = -8, 2
        simulator_rng = self.make_rng('simulator')
        # print(f"In the corrector: theta_1 shape: {theta_1.shape}, target shape: {target.shape}")
        stream_coordinate_com, _ = self.simulator_impl(theta_1, num_simulations=self.num_simulations,
                                        rng=simulator_rng, deterministic=True, stream=False)  
        
        stream_coordinate_com_backward, stream_coordinate_com_forward = stream_coordinate_com[0], stream_coordinate_com[1]
        
        # Create masks for valid time steps
        mask_window_backward = (stream_coordinate_com_backward[:, 0, 1] < phi1_max) & \
                            (stream_coordinate_com_backward[:, 0, 1] > phi1_min) & \
                            (stream_coordinate_com_backward[:, 0, 2] < phi2_max) & \
                            (stream_coordinate_com_backward[:, 0, 2] > phi2_min)
        
        mask_diff_backward = jnp.ediff1d(stream_coordinate_com_backward[:, 0, 1], to_begin=1) > 0
        # New mask - True until first False appears
        mask_diff_backward = jnp.cumprod(mask_diff_backward, dtype=bool)


        mask_window_forward = (stream_coordinate_com_forward[:, 0, 1] < phi1_max) & \
                            (stream_coordinate_com_forward[:, 0, 1] > phi1_min) & \
                            (stream_coordinate_com_forward[:, 0, 2] < phi2_max) & \
                            (stream_coordinate_com_forward[:, 0, 2] > phi2_min)
        
        mask_diff_forward = jnp.ediff1d(stream_coordinate_com_forward[:, 0, 1], to_begin=-1) < 0
        mask_diff_forward = jnp.cumprod(mask_diff_forward, dtype=bool)

        mask_backward = mask_window_backward & mask_diff_backward
        mask_forward = mask_window_forward & mask_diff_forward


        def coord_backward_fill(arr_phi1, arr_coord, mask):
            arr_phi1_masked = jnp.where(mask, arr_phi1, 0.0)
            filled = jnp.where(arr_phi1_masked == 0., jnp.max(arr_phi1_masked), arr_phi1_masked)
            arr_coord_masked = jnp.where(mask, arr_coord, 0.0)
            filled_coord = jnp.where(arr_coord_masked == 0., arr_coord_masked[jnp.argmax(filled)], arr_coord_masked)
            return filled, filled_coord

        def coord_forward_fill(arr_phi1, arr_coord, mask):
            arr_phi1_masked = jnp.where(mask, arr_phi1, 0.0)
            filled = jnp.where(arr_phi1_masked == 0., jnp.min(arr_phi1_masked), arr_phi1_masked)
            arr_coord_masked = jnp.where(mask, arr_coord, 0.0)
            filled_coord = jnp.where(arr_coord_masked == 0., arr_coord_masked[jnp.argmin(filled)], arr_coord_masked)
            return filled, filled_coord

        phi1_backward_valid, coord_backward_valid = jax.vmap(lambda coordinate: coord_backward_fill(stream_coordinate_com_backward[:, 0, 1], stream_coordinate_com_backward[:, 0, coordinate], mask_backward))(coordinate=coord_indices)
        phi1_forward_valid, coord_forw_valid = jax.vmap(lambda coordinate: coord_forward_fill(stream_coordinate_com_forward[:, 0, 1], stream_coordinate_com_forward[:, 0, coordinate], mask_forward))(coordinate=coord_indices)


        def interpolate_coord_backward(coord):
            return jnp.interp(stream_data[:, 1], phi1_backward_valid[0], coord)
        
        def interpolate_coord_forward(coord):
            return jnp.interp(stream_data[:, 1], phi1_forward_valid[0][::-1], coord[::-1])
            

        # Apply interpolation to all coordinates
        interp_tracks_backward = jax.vmap(interpolate_coord_backward)(coord_backward_valid)  # Shape: (n_coords, n_data)
        interp_tracks_forward = jax.vmap(interpolate_coord_forward)(coord_forw_valid)  # Shape: (n_coords, n_data)

        # Calculate residuals for all coordinates
        data_coords = stream_data[:, coord_indices].T  # Shape: (n_coords, n_data)
        sigma = jnp.array([0.5, 10., 2., 2. ])
        # sigma = jnp.array([0.15, 5., 0.1, 0.0001]) #from albatross

        mask_correct_interpolation_backward = stream_data[:, 1] < 10
        mask_correct_interpolation_forward = stream_data[:, 1] > - 90

        # Stream data masks - which data points to use for each direction
        mask_stream_backward = stream_data[:, 1] > stream_coordinate_com_backward[0, 0, 1]
        mask_stream_forward = stream_data[:, 1] < stream_coordinate_com_forward[0, 0, 1]

        mask_evaluate_inside_track_backward = (stream_data[:, 1] < jnp.max(phi1_backward_valid)) & (stream_data[:, 1] < phi1_max)
        mask_evaluate_inside_track_forward = (stream_data[:, 1] > jnp.min(phi1_forward_valid)) & (stream_data[:, 1] > phi1_min)

        # Calculate chi2 using only the appropriate data points for each direction
        residuals_backward = jnp.where(mask_stream_backward & mask_evaluate_inside_track_backward & mask_correct_interpolation_backward, 
                                    (data_coords - interp_tracks_backward)/sigma[:, None],
                                    0.)
        residuals_forward = jnp.where(mask_stream_forward & mask_evaluate_inside_track_forward & mask_correct_interpolation_forward, 
                                    (data_coords - interp_tracks_forward)/sigma[:, None],
                                    0.)
        residuals = jnp.where(mask_stream_backward,
                            residuals_backward,
                            residuals_forward)
        # Masks for valid residuals
        mask_backward_full = mask_stream_backward & mask_evaluate_inside_track_backward & mask_correct_interpolation_backward
        mask_forward_full = mask_stream_forward & mask_evaluate_inside_track_forward & mask_correct_interpolation_forward
        
        # Count valid data points PER COORDINATE
        mask_used = jnp.where(mask_stream_backward[:, None],
                            mask_backward_full[:, None],  # Broadcast to (n_data, 1)
                            mask_forward_full[:, None])
        
        
        
        # Chi-squared (sum of squared residuals)
        chi2 = jnp.sum(residuals**2)
        
        # Reduced chi-squared
        
        return chi2
       


    def forward_flow(self, t, theta, context, train=False):

        # we need this because self.model was trained with stacked flow which has additional time dimensions
        return self.model(t, theta, context, train=train)

    def forward(self, t, theta, context, train=True):

        # predict flow
        flow_pred = self.model(t, theta, context, train=train)

        if self.freeze:
            flow_pred = stop_gradient(flow_pred)

        # print(theta.shape)

        theta_1 = theta + jnp.einsum('ab,a->ab', flow_pred, 1 - t[:, 0])
        

        theta_1 = theta_1 * self.std_X + self.mean_X

        # print(f"In the corrector (should have a batch_dimension): theta_1 shape: {theta_1.shape}, context shape: {context.shape}")
        grad_fn = vmap(value_and_grad(self.NLL), in_axes=0)
        
        loss, grad = grad_fn(theta_1, context)

        loss = jnp.expand_dims(loss, axis=1)


        output = jnp.concatenate([loss, grad], axis=1)
        # output = jnp.nan_to_num(output).clip(-self.clip_output, self.clip_output)
        output = jnp.nan_to_num(output)

        output = self.aggregration_impl(output, train=train)

        output = jnp.concatenate([flow_pred, t, output], axis=1)
        # drift = flow_pred + self.controlled_flow_impl(theta=flow_pred, context=output, t=t) # noqa
        drift = flow_pred + self.controlled_flow_impl(output, context=None) # noqa


        drift = (jnp.einsum('ab, a -> ab', drift, t[:, 0] > self.start_time) +
                 jnp.einsum('ab, a -> ab', flow_pred, t[:, 0] <= self.start_time))

        return drift, output
    

class CorrectorDifferentiableSimulatorOdisseoPositions_fixtime_orbit_fitting_leastsquare(nn.Module):
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
        code_length = 10.0 * u.kpc
        code_mass = 1e4 * u.Msun
        code_time = 3 * u.Gyr
        self.code_units = CodeUnits(code_length, code_mass, G=1, unit_time = code_time )  
        self.mean_X = jnp.load('/export/data/vgiusepp/odisseo_data/data_varying_position_fixed_time_uniform_prior_TSTIT5/preprocess/mean_std_1e5_parameter_nolog.npz')['mean_theta']
        self.std_X = jnp.load('/export/data/vgiusepp/odisseo_data/data_varying_position_fixed_time_uniform_prior_TSTIT5/preprocess/mean_std_1e5_parameter_nolog.npz')['std_theta']

    def __call__(self, t, theta, context, train=True):

        return self.forward(t, theta, context, train=train)[0]

    def NLL(self, theta_1, target):
        theta_1 = theta_1.at[0].set(theta_1[0] * u.Msun.to(self.code_units.code_mass))
        theta_1 = theta_1.at[1].set(theta_1[1] * u.kpc.to(self.code_units.code_length))
        theta_1 = theta_1.at[2].set(theta_1[2] * u.Msun.to(self.code_units.code_mass))
        theta_1 = theta_1.at[3].set(theta_1[3] * u.kpc.to(self.code_units.code_length))
        theta_1 = theta_1.at[4].set(theta_1[4] * u.kpc.to(self.code_units.code_length))
        theta_1 = theta_1.at[5].set(theta_1[5] * u.Msun.to(self.code_units.code_mass))
        theta_1 = theta_1.at[6].set(theta_1[6] * u.kpc.to(self.code_units.code_length))
        stream_data = target
        coord_indices = jnp.array([2, 3, 4, 5])

        phi1_min, phi1_max = -90, 10
        phi2_min, phi2_max = -8, 2
        simulator_rng = self.make_rng('simulator')
        # print(f"In the corrector: theta_1 shape: {theta_1.shape}, target shape: {target.shape}")
        stream_coordinate_com, _ = self.simulator_impl(theta_1, num_simulations=self.num_simulations,
                                        rng=simulator_rng, deterministic=True, stream=False)  
        
        stream_coordinate_com_backward, stream_coordinate_com_forward = stream_coordinate_com[0], stream_coordinate_com[1]
        
        # Create masks for valid time steps
        mask_window_backward = (stream_coordinate_com_backward[:, 0, 1] < phi1_max) & \
                            (stream_coordinate_com_backward[:, 0, 1] > phi1_min) & \
                            (stream_coordinate_com_backward[:, 0, 2] < phi2_max) & \
                            (stream_coordinate_com_backward[:, 0, 2] > phi2_min)
        
        mask_diff_backward = jnp.ediff1d(stream_coordinate_com_backward[:, 0, 1], to_begin=1) > 0
        # New mask - True until first False appears
        mask_diff_backward = jnp.cumprod(mask_diff_backward, dtype=bool)


        mask_window_forward = (stream_coordinate_com_forward[:, 0, 1] < phi1_max) & \
                            (stream_coordinate_com_forward[:, 0, 1] > phi1_min) & \
                            (stream_coordinate_com_forward[:, 0, 2] < phi2_max) & \
                            (stream_coordinate_com_forward[:, 0, 2] > phi2_min)
        
        mask_diff_forward = jnp.ediff1d(stream_coordinate_com_forward[:, 0, 1], to_begin=-1) < 0
        mask_diff_forward = jnp.cumprod(mask_diff_forward, dtype=bool)

        mask_backward = mask_window_backward & mask_diff_backward
        mask_forward = mask_window_forward & mask_diff_forward


        def coord_backward_fill(arr_phi1, arr_coord, mask):
            arr_phi1_masked = jnp.where(mask, arr_phi1, 0.0)
            filled = jnp.where(arr_phi1_masked == 0., jnp.max(arr_phi1_masked), arr_phi1_masked)
            arr_coord_masked = jnp.where(mask, arr_coord, 0.0)
            filled_coord = jnp.where(arr_coord_masked == 0., arr_coord_masked[jnp.argmax(filled)], arr_coord_masked)
            return filled, filled_coord

        def coord_forward_fill(arr_phi1, arr_coord, mask):
            arr_phi1_masked = jnp.where(mask, arr_phi1, 0.0)
            filled = jnp.where(arr_phi1_masked == 0., jnp.min(arr_phi1_masked), arr_phi1_masked)
            arr_coord_masked = jnp.where(mask, arr_coord, 0.0)
            filled_coord = jnp.where(arr_coord_masked == 0., arr_coord_masked[jnp.argmin(filled)], arr_coord_masked)
            return filled, filled_coord

        phi1_backward_valid, coord_backward_valid = jax.vmap(lambda coordinate: coord_backward_fill(stream_coordinate_com_backward[:, 0, 1], stream_coordinate_com_backward[:, 0, coordinate], mask_backward))(coordinate=coord_indices)
        phi1_forward_valid, coord_forw_valid = jax.vmap(lambda coordinate: coord_forward_fill(stream_coordinate_com_forward[:, 0, 1], stream_coordinate_com_forward[:, 0, coordinate], mask_forward))(coordinate=coord_indices)


        def interpolate_coord_backward(coord):
            return jnp.interp(stream_data[:, 1], phi1_backward_valid[0], coord)
        
        def interpolate_coord_forward(coord):
            return jnp.interp(stream_data[:, 1], phi1_forward_valid[0][::-1], coord[::-1])
            

        # Apply interpolation to all coordinates
        interp_tracks_backward = jax.vmap(interpolate_coord_backward)(coord_backward_valid)  # Shape: (n_coords, n_data)
        interp_tracks_forward = jax.vmap(interpolate_coord_forward)(coord_forw_valid)  # Shape: (n_coords, n_data)

        # Calculate residuals for all coordinates
        data_coords = stream_data[:, coord_indices].T  # Shape: (n_coords, n_data)
        sigma = jnp.array([0.5, 10., 2., 2. ])
        # sigma = jnp.array([0.15, 5., 0.1, 0.0001]) #from albatross

        mask_correct_interpolation_backward = stream_data[:, 1] < 10
        mask_correct_interpolation_forward = stream_data[:, 1] > - 90

        # Stream data masks - which data points to use for each direction
        mask_stream_backward = stream_data[:, 1] > stream_coordinate_com_backward[0, 0, 1]
        mask_stream_forward = stream_data[:, 1] < stream_coordinate_com_forward[0, 0, 1]

        mask_evaluate_inside_track_backward = (stream_data[:, 1] < jnp.max(phi1_backward_valid)) & (stream_data[:, 1] < phi1_max)
        mask_evaluate_inside_track_forward = (stream_data[:, 1] > jnp.min(phi1_forward_valid)) & (stream_data[:, 1] > phi1_min)

        # Calculate chi2 using only the appropriate data points for each direction
        residuals_backward = jnp.where(mask_stream_backward & mask_evaluate_inside_track_backward & mask_correct_interpolation_backward, 
                                    (data_coords - interp_tracks_backward)/sigma[:, None],
                                    0.)
        residuals_forward = jnp.where(mask_stream_forward & mask_evaluate_inside_track_forward & mask_correct_interpolation_forward, 
                                    (data_coords - interp_tracks_forward)/sigma[:, None],
                                    0.)
        residuals = jnp.where(mask_stream_backward,
                            residuals_backward,
                            residuals_forward)
        # Masks for valid residuals
        mask_backward_full = mask_stream_backward & mask_evaluate_inside_track_backward & mask_correct_interpolation_backward
        mask_forward_full = mask_stream_forward & mask_evaluate_inside_track_forward & mask_correct_interpolation_forward
        
        # Count valid data points PER COORDINATE
        mask_used = jnp.where(mask_stream_backward[:, None],
                            mask_backward_full[:, None],  # Broadcast to (n_data, 1)
                            mask_forward_full[:, None])
        
        
        
        # Chi-squared (sum of squared residuals)
        chi2 = jnp.sum(residuals**2)
        
        # Reduced chi-squared
        
        return chi2

    def residuals(self, theta_1, args):
        target = args[0]
        theta_1 = theta_1.at[0].set(theta_1[0] * u.Msun.to(self.code_units.code_mass))
        theta_1 = theta_1.at[1].set(theta_1[1] * u.kpc.to(self.code_units.code_length))
        theta_1 = theta_1.at[2].set(theta_1[2] * u.Msun.to(self.code_units.code_mass))
        theta_1 = theta_1.at[3].set(theta_1[3] * u.kpc.to(self.code_units.code_length))
        theta_1 = theta_1.at[4].set(theta_1[4] * u.kpc.to(self.code_units.code_length))
        theta_1 = theta_1.at[5].set(theta_1[5] * u.Msun.to(self.code_units.code_mass))
        theta_1 = theta_1.at[6].set(theta_1[6] * u.kpc.to(self.code_units.code_length))
        stream_data = target
        coord_indices = jnp.array([2, 3, 4, 5])

        phi1_min, phi1_max = -90, 10
        phi2_min, phi2_max = -8, 2
        simulator_rng = self.make_rng('simulator')
        # print(f"In the corrector: theta_1 shape: {theta_1.shape}, target shape: {target.shape}")
        stream_coordinate_com, _ = self.simulator_impl(theta_1, num_simulations=self.num_simulations,
                                        rng=simulator_rng, deterministic=True, stream=False, forward_diff=True, )  
        
        stream_coordinate_com_backward, stream_coordinate_com_forward = stream_coordinate_com[0], stream_coordinate_com[1]
        
        # Create masks for valid time steps
        mask_window_backward = (stream_coordinate_com_backward[:, 0, 1] < phi1_max) & \
                            (stream_coordinate_com_backward[:, 0, 1] > phi1_min) & \
                            (stream_coordinate_com_backward[:, 0, 2] < phi2_max) & \
                            (stream_coordinate_com_backward[:, 0, 2] > phi2_min)
        
        mask_diff_backward = jnp.ediff1d(stream_coordinate_com_backward[:, 0, 1], to_begin=1) > 0
        # New mask - True until first False appears
        mask_diff_backward = jnp.cumprod(mask_diff_backward, dtype=bool)


        mask_window_forward = (stream_coordinate_com_forward[:, 0, 1] < phi1_max) & \
                            (stream_coordinate_com_forward[:, 0, 1] > phi1_min) & \
                            (stream_coordinate_com_forward[:, 0, 2] < phi2_max) & \
                            (stream_coordinate_com_forward[:, 0, 2] > phi2_min)
        
        mask_diff_forward = jnp.ediff1d(stream_coordinate_com_forward[:, 0, 1], to_begin=-1) < 0
        mask_diff_forward = jnp.cumprod(mask_diff_forward, dtype=bool)

        mask_backward = mask_window_backward & mask_diff_backward
        mask_forward = mask_window_forward & mask_diff_forward


        def coord_backward_fill(arr_phi1, arr_coord, mask):
            arr_phi1_masked = jnp.where(mask, arr_phi1, 0.0)
            filled = jnp.where(arr_phi1_masked == 0., jnp.max(arr_phi1_masked), arr_phi1_masked)
            arr_coord_masked = jnp.where(mask, arr_coord, 0.0)
            filled_coord = jnp.where(arr_coord_masked == 0., arr_coord_masked[jnp.argmax(filled)], arr_coord_masked)
            return filled, filled_coord

        def coord_forward_fill(arr_phi1, arr_coord, mask):
            arr_phi1_masked = jnp.where(mask, arr_phi1, 0.0)
            filled = jnp.where(arr_phi1_masked == 0., jnp.min(arr_phi1_masked), arr_phi1_masked)
            arr_coord_masked = jnp.where(mask, arr_coord, 0.0)
            filled_coord = jnp.where(arr_coord_masked == 0., arr_coord_masked[jnp.argmin(filled)], arr_coord_masked)
            return filled, filled_coord

        phi1_backward_valid, coord_backward_valid = jax.vmap(lambda coordinate: coord_backward_fill(stream_coordinate_com_backward[:, 0, 1], stream_coordinate_com_backward[:, 0, coordinate], mask_backward))(coordinate=coord_indices)
        phi1_forward_valid, coord_forw_valid = jax.vmap(lambda coordinate: coord_forward_fill(stream_coordinate_com_forward[:, 0, 1], stream_coordinate_com_forward[:, 0, coordinate], mask_forward))(coordinate=coord_indices)


        def interpolate_coord_backward(coord):
            return jnp.interp(stream_data[:, 1], phi1_backward_valid[0], coord)
        
        def interpolate_coord_forward(coord):
            return jnp.interp(stream_data[:, 1], phi1_forward_valid[0][::-1], coord[::-1])
            

        # Apply interpolation to all coordinates
        interp_tracks_backward = jax.vmap(interpolate_coord_backward)(coord_backward_valid)  # Shape: (n_coords, n_data)
        interp_tracks_forward = jax.vmap(interpolate_coord_forward)(coord_forw_valid)  # Shape: (n_coords, n_data)

        # Calculate residuals for all coordinates
        data_coords = stream_data[:, coord_indices].T  # Shape: (n_coords, n_data)
        sigma = jnp.array([0.5, 10., 2., 2. ])
        # sigma = jnp.array([0.15, 5., 0.1, 0.0001]) #from albatross

        mask_correct_interpolation_backward = stream_data[:, 1] < 10
        mask_correct_interpolation_forward = stream_data[:, 1] > - 90

        # Stream data masks - which data points to use for each direction
        mask_stream_backward = stream_data[:, 1] > stream_coordinate_com_backward[0, 0, 1]
        mask_stream_forward = stream_data[:, 1] < stream_coordinate_com_forward[0, 0, 1]

        mask_evaluate_inside_track_backward = (stream_data[:, 1] < jnp.max(phi1_backward_valid)) & (stream_data[:, 1] < phi1_max)
        mask_evaluate_inside_track_forward = (stream_data[:, 1] > jnp.min(phi1_forward_valid)) & (stream_data[:, 1] > phi1_min)

        # Calculate chi2 using only the appropriate data points for each direction
        residuals_backward = jnp.where(mask_stream_backward & mask_evaluate_inside_track_backward & mask_correct_interpolation_backward, 
                                    (data_coords - interp_tracks_backward)/sigma[:, None],
                                    0.)
        residuals_forward = jnp.where(mask_stream_forward & mask_evaluate_inside_track_forward & mask_correct_interpolation_forward, 
                                    (data_coords - interp_tracks_forward)/sigma[:, None],
                                    0.)
        residuals = jnp.where(mask_stream_backward,
                            residuals_backward,
                            residuals_forward)
        
        return residuals

    def minimization_vmap(self, y0, stream_data):
        return least_squares(
            fn=self.residuals,
            solver=optimistix.LevenbergMarquardt(rtol=1e-5, atol=1e-5),
            y0=y0,
            args=(stream_data,)
        ).value

    def gradient_descendt_vmap(self, y0, stream_data):
        res = optimistix.minimise(
            fn=self.NLL,
            solver = optimistix.OptaxMinimiser(optim = adamw(learning_rate=1e-4,),  rtol=1e-4, atol=1e-4),
            # solver = optimistix.LBFGS(rtol=1e-5, atol=1e-5,),
            y0=y0,
            args=stream_data,
            # max_steps = 100,
            ).value
        return res

       


    def forward_flow(self, t, theta, context, train=False):

        # we need this because self.model was trained with stacked flow which has additional time dimensions
        return self.model(t, theta, context, train=train)

    def forward(self, t, theta, context, train=True):

        # predict flow
        flow_pred = self.model(t, theta, context, train=train)

        if self.freeze:
            flow_pred = stop_gradient(flow_pred)

        # print(theta.shape)

        theta_1 = theta + jnp.einsum('ab,a->ab', flow_pred, 1 - t[:, 0])
  
        theta_1 = theta_1 * self.std_X + self.mean_X
        
        # theta_1 = theta_1 * self.std_X + self.mean_X
        print(theta_1)

        # theta_1 = jax.vmap(self.minimization_vmap)(theta_1, context )
        theta_1 = jax.vmap(self.minimization_vmap)(theta_1, context )

        # print(f"In the corrector (should have a batch_dimension): theta_1 shape: {theta_1.shape}, context shape: {context.shape}")
        grad_fn = vmap(value_and_grad(self.NLL), in_axes=0)
        
        loss, grad = grad_fn(theta_1, context)

        loss = jnp.expand_dims(loss, axis=1)


        output = jnp.concatenate([loss, grad], axis=1)
        # output = jnp.nan_to_num(output).clip(-self.clip_output, self.clip_output)
        output = jnp.nan_to_num(output)

        output = self.aggregration_impl(output, train=train)

        output = jnp.concatenate([flow_pred, t, output], axis=1)
        # drift = flow_pred + self.controlled_flow_impl(theta=flow_pred, context=output, t=t) # noqa
        drift = flow_pred + self.controlled_flow_impl(output, context=None) # noqa


        drift = (jnp.einsum('ab, a -> ab', drift, t[:, 0] > self.start_time) +
                 jnp.einsum('ab, a -> ab', flow_pred, t[:, 0] <= self.start_time))

        return drift, output
    


@jax.jit
def halo_to_sun(Xhalo: jnp.ndarray) -> jnp.ndarray:
    """
    Conversion from simulation frame to cartesian frame centred at Sun
    Args:
    Xhalo: 3d position (x [kpc], y [kpc], z [kpc]) in simulation frame
    Returns:
    3d position (x_s [kpc], y_s [kpc], z_s [kpc]) in Sun frame
    Examples
    --------
    >>> halo_to_sun(jnp.array([1.0, 2.0, 3.0]))
    """
    sunx = 8.0
    xsun = sunx - Xhalo[0]
    ysun = Xhalo[1]
    zsun = Xhalo[2]
    return jnp.array([xsun, ysun, zsun])


@jax.jit
def sun_to_gal(Xsun: jnp.ndarray) -> jnp.ndarray:
    """
    Conversion from sun cartesian frame to galactic co-ordinates
    Args:
    Xsun: 3d position (x_s [kpc], y_s [kpc], z_s [kpc]) in Sun frame
    Returns:
    3d position (r [kpc], b [rad], l [rad]) in galactic frame
    Examples
    --------
    >>> sun_to_gal(jnp.array([1.0, 2.0, 3.0]))
    """
    r = jnp.linalg.norm(Xsun)
    b = jnp.arcsin(Xsun[2] / r)
    l = jnp.arctan2(Xsun[1], Xsun[0])
    return jnp.array([r, b, l])


@jax.jit
def gal_to_equat(Xgal: jnp.ndarray) -> jnp.ndarray:
    """
    Conversion from galactic co-ordinates to equatorial co-ordinates
    Args:
    Xgal: 3d position (r [kpc], b [rad], l [rad]) in galactic frame
    Returns:
    3d position (r [kpc], alpha [rad], delta [rad]) in equatorial frame
    Examples
    --------
    >>> gal_to_equat(jnp.array([1.0, 2.0, 3.0]))
    """
    dNGPdeg = 27.12825118085622
    lNGPdeg = 122.9319185680026
    aNGPdeg = 192.85948
    dNGP = dNGPdeg * jnp.pi / 180.0
    lNGP = lNGPdeg * jnp.pi / 180.0
    aNGP = aNGPdeg * jnp.pi / 180.0
    r = Xgal[0]
    b = Xgal[1]
    l = Xgal[2]
    sb = jnp.sin(b)
    cb = jnp.cos(b)
    sl = jnp.sin(lNGP - l)
    cl = jnp.cos(lNGP - l)
    cs = cb * sl
    cc = jnp.cos(dNGP) * sb - jnp.sin(dNGP) * cb * cl
    alpha = jnp.arctan(cs / cc) + aNGP
    delta = jnp.arcsin(jnp.sin(dNGP) * sb + jnp.cos(dNGP) * cb * cl)
    return jnp.array([r, alpha, delta])

@partial(jax.jit, static_argnames=['transform_fn'])
def transform_velocity(transform_fn, X, V):
    """
    Generic velocity transformation through coordinate mapping.

    Args:
    transform_fn: function R^3 → R^3 mapping positions to new coordinates
    X: position vector in original coordinates (3,)
    V: velocity vector in original coordinates (3,)

    Returns:
    velocity vector in transformed coordinates (3,)
    """
    J = jax.jacobian(transform_fn)(X)  # (3,3) Jacobian
    return J @ V

@jax.jit
def halo_to_equatorial(Xhalo):
    Xsun = halo_to_sun(Xhalo)
    Xgal = sun_to_gal(Xsun)
    Xeq  = gal_to_equat(Xgal)
    return Xeq

#vmap functions
def halo_to_equatorial_batch(Xhalo):
    # Use partial to bind self, then vmap over the first positional arg
    return jax.vmap(lambda x: halo_to_equatorial(Xhalo=x))(Xhalo)

@partial(jax.jit, static_argnames=['transform_fn'])
def transform_velocity_batch(transform_fn, X, V):
    # Create a lambda that captures transform_fn and vmaps over X and V
    return jax.vmap(lambda x, v: transform_velocity(transform_fn=transform_fn, X=x, V=v))(X, V)

def stream_to_array(stream):
    pos = jnp.array([stream.q.x.to('kpc').value, stream.q.y.to('kpc').value, stream.q.z.to('kpc').value])
    vel = jnp.array([stream.p.x.to('km/s').value, stream.p.y.to('km/s').value, stream.p.z.to('km/s').value])
    return pos.T, vel.T


   

def log_prior_loguniform_logsigma(log_sigma, log_sigma_min, log_sigma_max):
    # uniform prior on log(sigma) between bounds -> p(log_sigma)=const inside
    # returns log p(sigma) up to additive constant
    in_bounds = (log_sigma >= log_sigma_min) & (log_sigma <= log_sigma_max)
    # if outside, return -inf
    return jnp.where(in_bounds, -jnp.log(log_sigma_max - log_sigma_min), -jnp.inf)


def log_diag_multivariate_normal(x, mean, sigma_eff):
    """
    Log PDF of a multivariate Gaussian with diagonal covariance.
    sigma_eff : (D,)  # effective standard deviation per dimension
    """
    diff = (x - mean) / sigma_eff
    D = x.shape[0]
    log_det = 2.0 * jnp.sum(jnp.log(sigma_eff))
    norm_const = -0.5 * (D * jnp.log(2 * jnp.pi) + log_det)
    exponent = -0.5 * jnp.sum(diff**2)
    return norm_const + exponent

def stream_likelihood_diag(model_stream, obs_stream, obs_errors, smooth_sigma):
    """
    Log-likelihood of observed stars given simulated stream (diagonal covariance),
    including model smoothing variance term Σ_k^2.
    
    Parameters
    ----------
    model_stream : (K, D)
    obs_stream : (N, D)
    obs_errors : (D,) or (N, D)
    smooth_sigma : (D,)  # per-dimension smoothing std deviation
    """
    sigma_eff = jnp.sqrt(obs_errors**2 + smooth_sigma**2)

    def obs_log_prob(obs):
        def model_log_prob(model_point):
            return log_diag_multivariate_normal(obs, model_point, sigma_eff)
        log_probs = jax.vmap(model_log_prob)(model_stream)
        log_p_stream = jax.scipy.special.logsumexp(log_probs) - jnp.log(model_stream.shape[0])
        return log_p_stream

    logL_values = jax.vmap(obs_log_prob)(obs_stream)
    return jnp.sum(logL_values)  # sum, not mean

@partial(jax.jit, static_argnames=['prior_fn', 'stream_loglikelihood_fn', 'n_grid'])
def marginalize_sigma_grid(stream_loglikelihood_fn,
                        model_stream, obs_stream, obs_errors,
                        log_sigma_min=-6.0, log_sigma_max=1.0, n_grid=128,
                        prior_fn=log_prior_loguniform_logsigma):
    """
    Numerically marginalize the likelihood over sigma using a log-grid.

    Arguments
    ---------
    stream_loglikelihood_fn: function(model_stream, obs_stream, obs_errors, smooth_sigma) -> log-likelihood scalar
    model_stream, obs_stream, obs_errors: as in your likelihood
    log_sigma_min, log_sigma_max: bounds in log10(sigma) **for the unit system of sigma**
                                (we'll work in natural log inside)
    n_grid: number of grid points in log10-space
    prior_fn: function(log_sigma_nat) -> log prior density (natural-log)
            The function receives natural-log sigma (ln(s)), not log10.
    Returns
    -------
    log_marginal_likelihood: scalar (natural log)
    """

    # grid in log10 space, but do arithmetic in natural log for accuracy
    log10_grid = jnp.linspace(log_sigma_min, log_sigma_max, n_grid)  # base-10 exponents
    # convert to natural log of sigma
    ln10 = jnp.log(10.0)
    log_sigma_grid_nat = log10_grid * ln10  # ln(sigma)
    sigma_grid = jnp.exp(log_sigma_grid_nat)  # sigma values

    # vectorized log-likelihood evaluation over grid of sigma
    # we expect stream_loglikelihood_fn to accept sigma either scalar or array
    batched_ll = jax.vmap(lambda s: stream_loglikelihood_fn(model_stream,
                                                            obs_stream,
                                                            obs_errors,
                                                            s))(sigma_grid)  # shape (n_grid,)

    # compute log prior for each grid point (prior on ln(sigma))
    log_prior_vals = jax.vmap(lambda ln_s: prior_fn(ln_s, log_sigma_min * ln10, log_sigma_max * ln10))(log_sigma_grid_nat)
    # Note: prior_fn expects natural-log bounds if you implemented it that way.

    # integration weight: dx = delta(ln sigma) when integrating over ln(sigma).
    # We're integrating p(data|s) p(s) ds. If prior is on ln(s), integral = ∫ p(data|s) p(ln s) e^{ln s} d(ln s)
    # Simpler: do the integral over ln(s): ∫ p(data|s(lns)) p(s(lns)) * exp(lns) dlns
    # If prior_fn returns log p(ln s) (i.e. prior on lns), adjust accordingly.
    # For a log-uniform prior p(s) ∝ 1/s -> p(ln s) is constant and p(s) = exp(-ln s + const), but easier:
    # We'll compute log integrand as: log p(data|s) + log p(s) and convert ds via trapezoid in ln(s).

    # Here we will assume prior_fn returns log p(ln s) (i.e., prior density per d(ln s)). If your prior is
    # uniform in ln s, log_p(ln s) is constant. For convenience, implement prior on ln(s) (natural log).
    # So integrand in d(ln s) is: p(data | s(lns)) * p(ln s)  (and integral over dlns)
    # That avoids extra Jacobian factors.

    # If prior_fn is log p(ln s), we can compute:
    log_integrand = batched_ll + log_prior_vals  # log of the integrand over d(ln s)

    # integrate over ln(s) with the trapezoid rule in log-space for numerical stability:
    # convert to linear weights with exp and trapezoid spacing, but do it with logsumexp stabilization
    # using the log-sum-exp plus log(dx) approach.

    # dx in ln(s) space (natural log)
    dx = (log_sigma_grid_nat[1] - log_sigma_grid_nat[0])
    # log of integrand plus log(dx)
    log_integrand_plus_dx = log_integrand + jnp.log(dx)

    # stable sum in log: log ∑ exp(log_integrand_plus_dx)
    log_marginal = jax.scipy.special.logsumexp(log_integrand_plus_dx)

    return log_marginal

@jax.jit
def log_multivariate_normal(x, mean, cov):
    """
    Log PDF of a multivariate Gaussian with full covariance.

    Parameters
    ----------
    x : (D,)
    mean : (D,)
    cov : (D, D)  # covariance matrix (must be symmetric positive definite)
    """
    D = x.shape[0]
    L = jnp.linalg.cholesky(cov)
    diff = x - mean
    solve = jax.scipy.linalg.solve_triangular(L, diff, lower=True)
    mahal = jnp.sum(solve**2)
    log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    norm_const = -0.5 * (D * jnp.log(2 * jnp.pi) + log_det)
    return norm_const - 0.5 * mahal


@jax.jit
def stream_likelihood_fullcov(model_stream, obs_stream, obs_errors, smooth_sigma, ):
    """
    Log-likelihood of observed stars given simulated stream (full covariance version).
    """
    cov = smooth_sigma

    def obs_log_prob(obs):
        def model_log_prob(model_point):
            return log_multivariate_normal(obs, model_point, cov)
        log_probs = jax.vmap(model_log_prob)(model_stream)
        return jax.scipy.special.logsumexp(log_probs) - jnp.log(model_stream.shape[0])

    logL_values = jax.vmap(obs_log_prob)(obs_stream)
    return jnp.sum(logL_values)

class CorrectorDifferentiableSimulatorGalaxPositions_batch_norm(nn.Module):
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
        code_length = 10.0 * u.kpc
        code_mass = 1e4 * u.Msun
        code_time = 3 * u.Gyr
        self.code_units = CodeUnits(code_length, code_mass, G=1, unit_time = code_time )  
        self.mean_X = jnp.load('/export/data/vgiusepp/galax_data/data_varying_position_uniform_prior/preprocess/mean_std_1e6_parameter.npz')['mean_theta']
        self.std_X = jnp.load('/export/data/vgiusepp/galax_data/data_varying_position_uniform_prior/preprocess/mean_std_1e6_parameter.npz')['std_theta']

    def __call__(self, t, theta, context, train=True):

        return self.forward(t, theta, context, train=train)[0]


    def NLL(self, theta_1, target):
        
        simulator_rng = self.make_rng('simulator')
        # print(f"In the corrector: theta_1 shape: {theta_1.shape}, target shape: {target.shape}")
        stream, _ = self.simulator_impl(theta_1, num_simulations=self.num_simulations,
                                        rng=simulator_rng, deterministic=True, )

        stream_pos, stream_vel = stream_to_array(stream=stream)
        pos_eq = halo_to_equatorial_batch(stream_pos)
        vel_eq = transform_velocity_batch(halo_to_equatorial, stream_pos, stream_vel)
        stream = jnp.concatenate([pos_eq, vel_eq], axis=1)

        # noise_std = jnp.array([10, 5, 0.1, 10.0, 5.0, 5.0])  # minimal observational noise
        noise_std = jnp.zeros(6)  # no observational noise

        # smooth_sigma = jnp.std(stream_target , axis=0) * 0.05  # 1% of target stream dispersion
        # return stream_likelihood_diag(model_stream=stream,
        #                      obs_stream=stream_target,
        #                      obs_errors=noise_std,
        #                      smooth_sigma=smooth_sigma)

        stream_cov = 0.05 *jnp.cov(target.T)
        return stream_likelihood_fullcov(model_stream=stream,
                                obs_stream=target,
                                obs_errors=noise_std,
                                smooth_sigma=stream_cov)
            
       

    def forward_flow(self, t, theta, context, train=False):

        # we need this because self.model was trained with stacked flow which has additional time dimensions
        return self.model(t, theta, context, train=train)

    def forward(self, t, theta, context, train=True):

        # predict flow
        flow_pred = self.model(t, theta, context, train=train)

        if self.freeze:
            flow_pred = stop_gradient(flow_pred)


        theta_1 = theta + jnp.einsum('ab,a->ab', flow_pred, 1 - t[:, 0])
        

        theta_1 = theta_1 * self.std_X + self.mean_X

        # print(f"In the corrector (should have a batch_dimension): theta_1 shape: {theta_1.shape}, context shape: {context.shape}")
        loss_fn = vmap(self.NLL, in_axes=0)
        grad_fn = vmap(jax.jacfwd(self.NLL), in_axes=0)
        
        loss, grad = loss_fn(theta_1, context), grad_fn(theta_1, context)

        loss = jnp.expand_dims(loss, axis=1)
        print('loss:', loss)
        print('grad:', grad)

        output = jnp.concatenate([loss, grad], axis=1)
        # output = jnp.nan_to_num(output).clip(-self.clip_output, self.clip_output)
        # output = jnp.nan_to_num(output)

        output = self.aggregration_impl(output, train=train)

        output = jnp.concatenate([flow_pred, t, output], axis=1)
        # drift = flow_pred + self.controlled_flow_impl(theta=flow_pred, context=output, t=t) # noqa
        drift = flow_pred + self.controlled_flow_impl(output, context=None) # noqa


        drift = (jnp.einsum('ab, a -> ab', drift, t[:, 0] > self.start_time) +
                 jnp.einsum('ab, a -> ab', flow_pred, t[:, 0] <= self.start_time))

        return drift, output