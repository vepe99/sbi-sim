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



class CorrectorDifferentiableSimulatorOdisseo_orbit_fitting(nn.Module):
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
        self.mean_X = jnp.load('/export/data/vgiusepp/odisseo_data/data_varying_position_uniform_prior_TSIT5/preprocess/mean_std_2e5_parameter.npz')['mean_theta']
        self.std_X = jnp.load('/export/data/vgiusepp/odisseo_data/data_varying_position_uniform_prior_TSIT5/preprocess/mean_std_2e5_parameter.npz')['std_theta']

    def __call__(self, t, theta, context, train=True):

        return self.forward(t, theta, context, train=train)[0]

    def NLL(self, theta_1, target):


        simulator_rng = self.make_rng('simulator')
        # print(f"In the corrector: theta_1 shape: {theta_1.shape}, target shape: {target.shape}")
        stream_coordinate_com, _ = self.simulator_impl(theta_1, num_simulations=self.num_simulations,
                                        rng=simulator_rng, deterministic=True, )  # noqa

        mask = target[:, 1]>stream_coordinate_com[0, :, 1]
        interp_stream_track = jnp.interp(
            target[:, 1], 
            stream_coordinate_com[:30, :, 1].ravel(), 
            stream_coordinate_com[:30, :, 2].ravel()
        )
        
        # Calculate residuals only for valid points
        residuals = jnp.where(mask, target[:, 2] - interp_stream_track, 0.0)
        n_valid = jnp.sum(mask)  # Number of valid data points
        sigma = 0.1  # Assumed observational uncertainty
        # Gaussian log-likelihood: ln L = -0.5 * [chi2 + N*ln(2π*σ²)]
        chi2 = jnp.sum(residuals**2) / sigma**2
        log_likelihood = -0.5 * (chi2 + n_valid * jnp.log(2 * jnp.pi * sigma**2))
        
        return -log_likelihood


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
        theta_1 = theta_1.at[:, 0].set(theta_1[:, 0] * u.Gyr.to(self.code_units.code_time))
        theta_1 = theta_1.at[:, 1].set((10**theta_1[:, 1]) * u.Msun.to(self.code_units.code_mass))
        theta_1 = theta_1.at[:, 2].set(theta_1[:, 2] * u.kpc.to(self.code_units.code_length))
        theta_1 = theta_1.at[:, 3].set((10**theta_1[:, 3]) * u.Msun.to(self.code_units.code_mass))
        theta_1 = theta_1.at[:, 4].set(theta_1[:, 4] * u.kpc.to(self.code_units.code_length))
        theta_1 = theta_1.at[:, 5].set((10**theta_1[:, 5]) * u.Msun.to(self.code_units.code_mass))
        theta_1 = theta_1.at[:, 6].set(theta_1[:, 6] * u.kpc.to(self.code_units.code_length))

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

        output = jnp.concatenate([flow_pred, t, output], axis=1)
        # drift = flow_pred + self.controlled_flow_impl(theta=flow_pred, context=output, t=t) # noqa
        drift = flow_pred + self.controlled_flow_impl(output, context=None) # noqa
        # drift = flow_pred + self.controlled_flow_impl(sample=flow_pred, timesteps=t, encoder_hidden_states=context, loss_grad=output, train=train)  #this is for the conv_2d_cross_attention
        
        drift = (jnp.einsum('ab, a -> ab', drift, t[:, 0] > self.start_time) +
                 jnp.einsum('ab, a -> ab', flow_pred, t[:, 0] <= self.start_time))

        return drift, output