from pathlib import Path
from typing import Dict, Union, Tuple

from jaxtyping import PyTree
from omegaconf import OmegaConf


import jax.random as jr
import flax.linen as nn

import jax.numpy as jnp

from .conditional_flow_matching_batchnorm import ConditionalFlowMatching_batchnorm, get_weighting_fun
from .distributions.base_distribution import BaseDistribution
from .improved_inference import BaseSampler
from .paths import sample_time
from ..utils import instantiate_from_config, restore_checkpoint

class SimulatorFeedback_batchnorm(ConditionalFlowMatching_batchnorm):

    def load_weights(self, weight_file: str):
        weight_file = Path(weight_file)
        checkpoint = restore_checkpoint(weight_file, type='best_val')

        return checkpoint['state'][0]
    
    @property
    def variables(self):
        return {'params': self.params, 'batch_stats': self.batch_stats}

    @variables.setter
    def variables(self, value):
        # optional: validate keys exist
        self.params = value['params']
        self.batch_stats = value.get('batch_stats', self.batch_stats)


    def __init__(self, dim_flow: int, dim_conditioning: Union[int, Tuple[int]], model: Dict,
                 schedule: Dict, sampler: Dict, prior: Dict,
                 model_pretrained: Dict, weight_file_pretrained: str,
                 weighting: str = "one", loss_type: str = "lipman",
                 start_time: float = 0.0,
                 time_alpha: float = 0.0,
                 scaling_type: str = 'none', bandwidth: float = 0.5, **kwargs):

        self.dim_flow = dim_flow
        self.dim_conditioning = dim_conditioning
        self.time_alpha = time_alpha

        # load pretrained model and weights
        self.model_pretrained = instantiate_from_config(model_pretrained)
        self.model_params = self.load_weights(weight_file_pretrained)

        model = OmegaConf.to_container(model, resolve=True)
        # hijack model config to load pretrained model
        model['params']['model'] = self.model_pretrained
        self.model: nn.Module = instantiate_from_config(model)

        self.start_time = start_time

        self.bandwidth = bandwidth
        self.loss_type = loss_type
        self.loss_fn = self.get_loss_fn(loss_type)
        self.path = instantiate_from_config(schedule)
        self.scaling_type = scaling_type
        self.scaled_model = self.get_scaled_model(scaling_type)
        self.prior: BaseDistribution = instantiate_from_config(prior)
        self.sampler: BaseSampler = instantiate_from_config(sampler)
        self.weighting_function = get_weighting_fun(weighting)

        self.opt = None
        self.initialized = False

    def setup(self, opt, example_data: PyTree, key: jr.PRNGKey, batch_size: int) -> Tuple[PyTree, jr.PRNGKey]:

        self.opt = opt
        self.batch_size = batch_size

        rng, init_rng, model_rng = jr.split(key, 3)

        t, _ = sample_time(rng, example_data["parameters"].shape[0],
                           t0=self.start_time, t1=self.end_time)

        t_in = jnp.expand_dims(t, axis=1)

        (params_rng, dropout_rng,
         drop_path_rng, dropout_rng, simulator_rng) = jr.split(init_rng, 5)

        init_dict = {'params': model_rng, 'drop_path': init_rng,
                     'dropout': dropout_rng, 'simulator': simulator_rng}

        # initialize model *with* train=True so BN running stats are created
        init_model = self.model.init(init_dict, t_in, example_data["parameters"],
                                    example_data["conditioning"], train=True)

        # debug check (optional)
        # print("init_model keys:", list(init_model.keys()))
        # extract and store running stats (if any)
        self.batch_stats = init_model.get("batch_stats", None)

        # inject pretrained weights for the nested 'model' param but DO NOT clobber batch_stats
        # (we keep init_model['batch_stats'] which was created above)
        init_model['params']['model'] = self.model_params

        # now initialize optimizer with full params (pretrained nested model included)
        self.opt.init(init_model['params'])


        self.initialized = True

        return init_model, rng