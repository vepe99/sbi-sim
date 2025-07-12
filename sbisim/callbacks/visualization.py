from typing import Tuple, List

import wandb
import os

from .callback import Callback

import jax.random as jr

from ..data.benchmarks.generator_dataloader import GeneratorDataloader
from ..strategy import Strategy

import jax
import jax.numpy as jnp

from matplotlib import pyplot as plt


import pandas as pd
from chainconsumer import Chain, ChainConsumer, Truth
from odisseo.option_classes import SimulationConfig, SimulationParams, MNParams, NFWParams, PlummerParams, PSPParams, MN_POTENTIAL, NFW_POTENTIAL, PSP_POTENTIAL
from astropy import units as u
from odisseo.units import CodeUnits

code_length = 10 * u.kpc
code_mass = 1e4 * u.Msun
G = 1
code_time = 3 * u.Gyr
code_units = CodeUnits(code_length, code_mass, G=1, unit_time = code_time )  

params = SimulationParams(t_end = (3 * u.Gyr).to(code_units.code_time).value,  
                          Plummer_params= PlummerParams(Mtot=(10**4.05 * u.Msun).to(code_units.code_mass).value,
                                                        a=(8 * u.pc).to(code_units.code_length).value),
                           MN_params= MNParams(M = (68_193_902_782.346756 * u.Msun).to(code_units.code_mass).value,
                                              a = (3.0 * u.kpc).to(code_units.code_length).value,
                                              b = (0.280 * u.kpc).to(code_units.code_length).value),
                          NFW_params= NFWParams(Mvir=(4.3683325e11 * u.Msun).to(code_units.code_mass).value,
                                               r_s= (16.0 * u.kpc).to(code_units.code_length).value,),      
                          PSP_params= PSPParams(M = 4501365375.06545 * u.Msun.to(code_units.code_mass),
                                                alpha = 1.8, 
                                                r_c = (1.9*u.kpc).to(code_units.code_length).value),                    
                          G=code_units.G, ) 

class BenchmarkScatterPlot(Callback):

    name: str = 'benchmark_scatter_plot'
    save_every: int = 10

    observation_idx: List[int] = [1,2]
    num_total_samples: int = 10000
    batch_size: int = 128

    def __init__(self, save_every: int = 10, savedir: str = None):
        super().__init__()
        self.save_every = save_every
        self.savedir = savedir

        if self.savedir is not None:
            os.makedirs(self.savedir + '/pictures', exist_ok=True)
        
    def __call__(self, logs: dict, rng: jr.PRNGKey, *args, **kwargs):
        return self._call(logs, rng, *args, **kwargs)

    def _call(self, logs: dict, rng: jr.PRNGKey, strategy: Strategy,
              train_loader: GeneratorDataloader, val_loader: GeneratorDataloader, *args, **kwargs) \
            -> Tuple[dict, jr.PRNGKey]:

        figure_list = []

        for id in self.observation_idx:

            observation, true_theta, reference_posterior = train_loader.get_observation(id)

            observation = jnp.array(observation).repeat(self.num_total_samples, axis=0)
            posterior_samples, _ = strategy.sample(self.num_total_samples, rng, conditioning=observation,
                                                   batch_size=self.batch_size)

            fig = plt.figure(figsize=(10, 10))
            plt.scatter(reference_posterior[:, 0], reference_posterior[:, 1], label="Reference Posterior", alpha=0.2)
            # plt.scatter(posterior_samples["samples"][:, 0], posterior_samples["samples"][:, 1],
            #             label="Posterior Samples", alpha=0.2)

            plt.legend()
            plt.xlabel("Parameter 1")
            plt.ylabel("Parameter 2")

            plt.title(f"Observation {id}")

            epoch = logs["epoch"]

            if self.savedir is not None:
                plt.savefig(f"{self.savedir}/pictures/posterior_observation_{id}_{epoch}.png")

            figure_list.append(fig)

        logs['posteriors'] = [wandb.Image(fig) for fig in figure_list]

        for fig in figure_list:
            plt.close(fig)

        return logs, rng

    def on_train_begin(self, *args, **kwargs):
        return self.__call__(*args, init=True, **kwargs)

    def on_epoch_end(self, logs: dict, rng: jr.PRNGKey, *args, **kwargs):

        if "epoch" in logs and logs["epoch"] % self.save_every == 0:
            return self.__call__(logs, rng, *args, **kwargs)
        else:
            return logs, rng

class corner_plot_posterior(Callback):

    name: str = 'corner_plot_posterior'
    save_every: int = 10

    observation_idx: List[int] = [1,2]
    num_total_samples: int = 10000
    batch_size: int = 128

    def __init__(self, save_every: int = 10, savedir: str = None, num_total_samples: int = 10_000,):
        super().__init__()
        self.save_every = save_every
        self.savedir = savedir
        self.num_total_samples = num_total_samples

        if self.savedir is not None:
            os.makedirs(self.savedir + '/pictures', exist_ok=True)
        
    def __call__(self, logs: dict, rng: jr.PRNGKey, *args, **kwargs):
        return self._call(logs, rng, *args, **kwargs)

    def _call(self, logs: dict, rng: jr.PRNGKey, strategy: Strategy,
              train_loader: GeneratorDataloader, val_loader: GeneratorDataloader, *args, **kwargs) \
            -> Tuple[dict, jr.PRNGKey]:

        figure_list = []

        for id in self.observation_idx:

            observation, true_theta, reference_posterior = train_loader.get_observation(id)

            observation = jnp.array(observation).repeat(self.num_total_samples, axis=0)
            posterior_samples, _ = strategy.sample(self.num_total_samples, rng, conditioning=observation,
                                                   batch_size=self.batch_size)
            print(f"Posterior samples shape: {posterior_samples['samples'].shape}")
            print(f'True theta: {true_theta}')
            df = pd.DataFrame(posterior_samples["samples"], columns=['t_end', 'M_plummer', 'M_NFW', 'M_MN'])
            # Create corner plot with ChainConsumer
            c = ChainConsumer()
            c.add_chain(Chain(samples=df, name="Experimental Results", ), )
            c.add_truth(Truth(location={
                't_end': true_theta[0, 0], 
                'M_plummer': true_theta[0, 1],
                'M_NFW': true_theta[0, 2], 
                'M_MN': true_theta[0, 3],
            }))
            fig = c.plotter.plot()

            # fig = plt.figure(figsize=(10, 10))
            # plt.scatter(reference_posterior[:, 0], reference_posterior[:, 1], label="Reference Posterior", alpha=0.2)
            # plt.scatter(posterior_samples["samples"][:, 0], posterior_samples["samples"][:, 1],
            #             label="Posterior Samples", alpha=0.2)

            # plt.legend()
            # plt.xlabel("Parameter 1")
            # plt.ylabel("Parameter 2")

            # plt.title(f"Observation {id}")


            epoch = logs["epoch"]

            if self.savedir is not None:
                plt.savefig(f"{self.savedir}/pictures/corner_plot_{id}_{epoch}.png")

            figure_list.append(fig)

        logs['posteriors'] = [wandb.Image(fig) for fig in figure_list]

        for fig in figure_list:
            plt.close(fig)

        return logs, rng

    # def on_train_begin(self, *args, **kwargs):
    #     return self.__call__(*args, init=True, **kwargs)
    
    def on_train_end(self, *args, **kwargs):
        return self.__call__(*args, init=True, **kwargs)

    def on_epoch_end(self, logs: dict, rng: jr.PRNGKey, *args, **kwargs):

        if "epoch" in logs and logs["epoch"] % self.save_every == 0:
            return self.__call__(logs, rng, *args, **kwargs)
        else:
            return logs, rng
