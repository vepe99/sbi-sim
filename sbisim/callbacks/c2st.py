from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from .callback import Callback
from ..data.benchmarks.generator_dataloader import GeneratorDataloader
from ..metrics.c2st import c2st

# from ..metrics.mmd import mmd

from ..strategy import Strategy

import jax.random as jr
import jax.numpy as jnp
from jax.scipy.stats import norm


from typing import Tuple, List, Optional

import wandb
import os

from .callback import Callback
import numpy as np

import jax.random as jr

from ..data.benchmarks.generator_dataloader import GeneratorDataloader
from ..strategy import Strategy
import tarp

import jax
import jax.numpy as jnp

from matplotlib import pyplot as plt
from math import log10


import pandas as pd
from chainconsumer import Chain, ChainConsumer, Truth
from odisseo.option_classes import SimulationConfig, SimulationParams, MNParams, NFWParams, PlummerParams, PSPParams, MN_POTENTIAL, NFW_POTENTIAL, PSP_POTENTIAL
from astropy import units as u
from odisseo.units import CodeUnits

class C2ST(Callback):

    name: str = 'rankss'
    save_every: int = 0

    # observation_idx: List[int] = [i for i in range(1, observation_idx)]
    num_total_samples: int = 1_000
    # batch_size: int = 1_000

    def __init__(self, save_every: int = 0, savedir: str = None, num_total_samples: int = 1_000, observation_idx: int = 10, batch_size: int = 1_000):
        super().__init__()
        self.save_every = save_every
        self.savedir = savedir
        self.num_total_samples = num_total_samples
        self.low=jnp.array([ 0.5,
                        10**3., 
                        10**log10(1/4 * 4.3683325e11), 
                        10**log10(1/4 * 68_193_902_782.346756), ])
        self.high=jnp.array([5, 
                        10**4.5, 
                        10**log10(2 * 4.3683325e11),
                        10**log10(2 * 68_193_902_782.346756),])
        self.observation_idx = [i for i in range(1, observation_idx)]
        self.batch_size = batch_size
        self.labels = ['$t_{end}$', '$M_{plummer}$', '$M_{NFW}$', '$M_{MN}$']
        if self.savedir is not None:
            os.makedirs(self.savedir + '/pictures', exist_ok=True)
        
    def __call__(self, logs: dict, rng: jr.PRNGKey, *args, **kwargs):
        return self._call(logs, rng, *args, **kwargs)

    def _call(self, logs: dict, rng: jr.PRNGKey, strategy: Strategy,
              train_loader: GeneratorDataloader, val_loader: GeneratorDataloader, *args, **kwargs) \
            -> Tuple[dict, jr.PRNGKey]:

        
        figure_list = []
        fig, axes = plt.subplots(1, 4, figsize=(25, 5))  # Create figure and 4 axes once

        true_theta_test_set = np.zeros((len(self.observation_idx), 4))
        predicted_theta_test_set = np.zeros((self.num_total_samples, len(self.observation_idx), 4))

        for id in self.observation_idx:

            observation, true_theta, reference_posterior = train_loader.get_observation(id)
            true_theta_test_set[id-1, :] = true_theta[0, :]

            observation = jnp.array(observation).repeat(self.num_total_samples, axis=0)
            posterior_samples, _ = strategy.sample(self.num_total_samples, rng, conditioning=observation,
                                                   batch_size=self.batch_size)
            posterior_samples['samples'] = 0.5 * (posterior_samples['samples'] + 1) * (self.high-self.low) + self.low
            predicted_theta_test_set[:, id-1, :] = posterior_samples['samples']

            np.savez(f"{self.savedir}/pictures/predicted_theta_test_set.npz",
                 predicted_theta=predicted_theta_test_set,
                 true_theta=true_theta_test_set)
            
        print(f"Predicted theta shape: {predicted_theta_test_set.shape}")
        print(f'true_theta_test_set: {true_theta_test_set.shape}')

        # print(logs)
        # epoch = logs["epoch"]

        np.savez(f"{self.savedir}/pictures/predicted_theta_test_set.npz",
                 predicted_theta=predicted_theta_test_set,
                 true_theta=true_theta_test_set)

        rank_plot = self._plot_ranks_histogram(
            samples=predicted_theta_test_set,
            trues=true_theta_test_set,
            nbins=10)
        
        coverage_plot = self._plot_coverage(
            samples=predicted_theta_test_set,
            trues=true_theta_test_set,
            plotscatter=True,
        )

        prediction_plot = self._plot_predictions(
            samples=predicted_theta_test_set,
            trues=true_theta_test_set,
        )

        tarp_plot = self._plot_TARP(
            posterior_samples=predicted_theta_test_set,
            theta=true_theta_test_set,
        )
        
        figure_list.append(rank_plot)
        figure_list.append(coverage_plot)
        figure_list.append(prediction_plot)

        if self.savedir is not None:
                rank_plot.savefig(f"{self.savedir}/pictures/ranks_histogram.pdf")
                coverage_plot.savefig(f"{self.savedir}/pictures/coverage_plot.pdf")
                prediction_plot.savefig(f"{self.savedir}/pictures/prediction_plot.pdf")
                tarp_plot.savefig(f"{self.savedir}/pictures/tarp_plot.pdf")

        figure_list.append(fig)


        logs['posteriors'] = [wandb.Image(fig) for fig in figure_list]

        for fig in figure_list:
            plt.close(fig)

        return logs, rng
    
    def on_test(self, logs: dict, rng: jr.PRNGKey, *args, **kwargs):
            return self.__call__(logs, rng, *args, **kwargs)
    
    def _get_ranks(
            self,
            samples: np.array,
            trues: np.array,
        ) -> np.array:
            """Get the marginal ranks of the true parameters in the posterior samples.

            Args:
                samples (np.array): posterior samples of shape (nsamples, ndata, npars)
                trues (np.array): true parameters of shape (ndata, npars)

            Returns:
                np.array: ranks of the true parameters in the posterior samples 
                    of shape (ndata, npars)
            """
            ranks = (samples < trues[None, ...]).sum(axis=0)
            return ranks


    def _plot_ranks_histogram(
            self, samples: np.ndarray, trues: np.ndarray, nbins: int = 10
        ) -> plt.Figure:
            """
            Plot a histogram of ranks for each parameter.

            Args:
                samples (numpy.ndarray): List of samples.
                trues (numpy.ndarray): Array of true values.
                signature (str): Signature for the histogram file name.
                nbins (int, optional): Number of bins for the histogram. Defaults to 10.

            Returns:
                matplotlib.figure.Figure: The generated figure.

            """
            ndata, npars = trues.shape
            navg = ndata / nbins
            ranks = self._get_ranks(samples, trues)

            fig, ax = plt.subplots(1, npars, figsize=(npars * 3, 4))
            if npars == 1:
                ax = [ax]

            for i in range(npars):
                ax[i].hist(np.array(ranks)[:, i], bins=nbins)
                ax[i].set_title(self.labels[i])
            ax[0].set_ylabel('counts')

            for axis in ax:
                axis.set_xlim(0, ranks.max())
                axis.set_xlabel('rank')
                axis.grid(visible=True)
                axis.axhline(navg, color='k')
                axis.axhline(navg - navg ** 0.5, color='k', ls="--")
                axis.axhline(navg + navg ** 0.5, color='k', ls="--")
            return fig

    def _plot_coverage(
            self, samples: np.ndarray, trues: np.ndarray, plotscatter: bool = True
        ) -> plt.Figure:
            """
            Plot the coverage of predicted percentiles against empirical percentiles.

            Args:
                samples (numpy.ndarray): Array of predicted samples.
                trues (numpy.ndarray): Array of true values.
                signature (str): Signature for the plot file name.
                plotscatter (bool, optional): Whether to plot the scatter plot. Defaults to True.

            Returns:
                matplotlib.figure.Figure: The generated figure.

            """
            ndata, npars = trues.shape
            ranks = self._get_ranks(samples, trues)

            unicov = [np.sort(np.random.uniform(0, 1, ndata)) for j in range(200)]
            unip = np.percentile(unicov, [5, 16, 84, 95], axis=0)

            fig, ax = plt.subplots(1, npars, figsize=(npars * 4, 4))
            if npars == 1:
                ax = [ax]
            cdf = np.linspace(0, 1, len(ranks))
            for i in range(npars):
                xr = np.sort(ranks[:, i])
                xr = xr / xr[-1]
                ax[i].plot(cdf, cdf, 'k--')
                if plotscatter:
                    ax[i].fill_between(cdf, unip[0], unip[-1],
                                    color='gray', alpha=0.2)
                    ax[i].fill_between(cdf, unip[1], unip[-2],
                                    color='gray', alpha=0.4)
                ax[i].plot(xr, cdf, lw=2, label='posterior')
                ax[i].set(adjustable='box', aspect='equal')
                ax[i].set_title(self.labels[i])
                ax[i].set_xlabel('Predicted Percentile')
                ax[i].set_xlim(0, 1)
                ax[i].set_ylim(0, 1)

            ax[0].set_ylabel('Empirical Percentile')
            for axis in ax:
                axis.grid(visible=True)
            return fig

    def _plot_predictions(
            self, samples: np.ndarray, trues: np.ndarray,
        ) -> plt.Figure:
            """
            Plot the mean and standard deviation of the predicted samples against
            the true values.

            Args:
                samples (np.ndarray): Array of predicted samples.
                trues (np.ndarray): Array of true values.
                signature (str): Signature for the plot.

            Returns:
                plt.Figure: The plotted figure.
            """
            npars = trues.shape[-1]
            mus, stds = samples.mean(axis=0), samples.std(axis=0)
            # print(f"mus shape: {mus.shape}, stds shape: {stds.shape}, trues shape: {trues.shape}")
            fig, axs = plt.subplots(1, npars, figsize=(npars * 4, 4))
            if npars == 1:
                axs = [axs]
            else:
                axs = axs.flatten()
            for j in range(npars):
                axs[j].errorbar(trues[:, j], mus[:, j], stds[:, j],
                                fmt="none", elinewidth=0.5, alpha=0.5)
                axs[j].plot(
                    *(2 * [np.linspace(min(trues[:, j]), max(trues[:, j]), 10)]),
                    'k--', ms=0.2, lw=0.5)
                axs[j].grid(which='both', lw=0.5)
                axs[j].set(adjustable='box', aspect='equal')
                axs[j].set_title(self.labels[j], fontsize=12)
                axs[j].set_xlabel('True')
            axs[0].set_ylabel('Predicted')

            return fig



    def _plot_TARP(
        self, posterior_samples: np.array, theta: np.array,
        references: str = "random", metric: str = "euclidean",
        bootstrap: Optional[bool] = True, norm: Optional[bool] = True,
        num_alpha_bins: Optional[int] = 10,
        num_bootstrap: Optional[int] = 100
    ) -> plt.Figure:
        """
        Plots the TARP credibility metric for the given posterior samples
        and theta values. See https://arxiv.org/abs/2302.03026 for details.

        Args:
            posterior_samples (np.array): Array of posterior samples.
            theta (np.array): Array of theta values.
            signature (str): Signature for the plot.
            references (str, optional): TARP reference type for TARP calculation. 
                Defaults to "random".
            metric (str, optional): TARP distance metric for TARP calculation. 
                Defaults to "euclidean".
            bootstrap (bool, optional): Whether to use bootstrapping for TARP error bars. 
                Defaults to False.
            norm (bool, optional): Whether to normalize the TARP metric. Defaults to True.
            num_alpha_bins (int, optional):number of bins to use for the TARP
                credibility values. Defaults to None.
            num_bootstrap (int, optional): Number of bootstrap iterations
                for TARP calculation. Defaults to 100.

        Returns:
            plt.Figure: The generated TARP plot.
        """
        ecp, alpha = tarp.get_tarp_coverage(
            posterior_samples, theta,
            references=references, metric=metric,
            norm=norm, bootstrap=bootstrap,
            num_alpha_bins=num_alpha_bins,
            num_bootstrap=num_bootstrap
        )

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.plot([0, 1], [0, 1], ls='--', color='k')
        if bootstrap:
            ecp_mean = np.mean(ecp, axis=0)
            ecp_std = np.std(ecp, axis=0)
            ax.plot(alpha, ecp_mean, label='TARP', color='b')
            ax.fill_between(alpha, ecp_mean - ecp_std, ecp_mean + ecp_std,
                            alpha=0.2, color='b')
            ax.fill_between(alpha, ecp_mean - 2 * ecp_std, ecp_mean + 2 * ecp_std,
                            alpha=0.2, color='b')
        else:
            ax.plot(alpha, ecp, label='TARP')
        ax.legend()
        ax.set_ylabel("Expected Coverage")
        ax.set_xlabel("Credibility Level")

        return fig
    
class C2ST100(Callback):

    name: str = 'rankss'
    save_every: int = 0

    # observation_idx: List[int] = [i for i in range(1, observation_idx)]
    num_total_samples: int = 1_000
    # batch_size: int = 1_000

    def __init__(self, save_every: int = 0, savedir: str = None, num_total_samples: int = 1_000, observation_idx: int = 10, batch_size: int = 1_000):
        super().__init__()
        self.save_every = save_every
        self.savedir = savedir
        self.num_total_samples = num_total_samples
        self.low=jnp.array([ 0.5,
                        10**3., 
                        10**log10(1/4 * 4.3683325e11), 
                        10**log10(1/4 * 68_193_902_782.346756), ])
        self.high=jnp.array([5, 
                        10**4.5, 
                        10**log10(2 * 4.3683325e11),
                        10**log10(2 * 68_193_902_782.346756),])
        self.observation_idx = [i+100+4 for i in range(1, observation_idx)]
        self.batch_size = batch_size
        self.labels = ['$t_{end}$', '$M_{plummer}$', '$M_{NFW}$', '$M_{MN}$']
        if self.savedir is not None:
            os.makedirs(self.savedir + '/pictures', exist_ok=True)
        
    def __call__(self, logs: dict, rng: jr.PRNGKey, *args, **kwargs):
        return self._call(logs, rng, *args, **kwargs)

    def _call(self, logs: dict, rng: jr.PRNGKey, strategy: Strategy,
              train_loader: GeneratorDataloader, val_loader: GeneratorDataloader, *args, **kwargs) \
            -> Tuple[dict, jr.PRNGKey]:

        
        # true_theta_test_set = np.zeros((len(self.observation_idx), 4))
        # predicted_theta_test_set = np.zeros((self.num_total_samples, len(self.observation_idx), 4))

        predicted_theta_test_set = np.load(f"{self.savedir}/pictures/predicted_theta_test_set_100.npz")['predicted_theta']
        true_theta_test_set = np.load(f"{self.savedir}/pictures/predicted_theta_test_set_100.npz")['true_theta']

        for id in self.observation_idx:

            observation, true_theta, reference_posterior = train_loader.get_observation(id)
            true_theta_test_set[id-1-100, :] = true_theta[0, :]

            observation = jnp.array(observation).repeat(self.num_total_samples, axis=0)
            posterior_samples, _ = strategy.sample(self.num_total_samples, rng, conditioning=observation,
                                                   batch_size=self.batch_size)
            posterior_samples['samples'] = 0.5 * (posterior_samples['samples'] + 1) * (self.high-self.low) + self.low
            predicted_theta_test_set[:, id-1-100, :] = posterior_samples['samples']

            np.savez(f"{self.savedir}/pictures/predicted_theta_test_set_100.npz",
                    predicted_theta=predicted_theta_test_set,
                    true_theta=true_theta_test_set)
        
        figure_list.append(rank_plot)
        figure_list.append(coverage_plot)
        figure_list.append(prediction_plot)
        figure_list.append(fig)
        logs['posteriors'] = [wandb.Image(fig) for fig in figure_list]

        for fig in figure_list:
            plt.close(fig)

        return logs, rng
    
    def on_test(self, logs: dict, rng: jr.PRNGKey, *args, **kwargs):
            return self.__call__(logs, rng, *args, **kwargs)

    

    def _get_ranks(
            self,
            samples: np.array,
            trues: np.array,
        ) -> np.array:
            """Get the marginal ranks of the true parameters in the posterior samples.

            Args:
                samples (np.array): posterior samples of shape (nsamples, ndata, npars)
                trues (np.array): true parameters of shape (ndata, npars)

            Returns:
                np.array: ranks of the true parameters in the posterior samples 
                    of shape (ndata, npars)
            """
            ranks = (samples < trues[None, ...]).sum(axis=0)
            return ranks


    def _plot_ranks_histogram(
            self, samples: np.ndarray, trues: np.ndarray, nbins: int = 10
        ) -> plt.Figure:
            """
            Plot a histogram of ranks for each parameter.

            Args:
                samples (numpy.ndarray): List of samples.
                trues (numpy.ndarray): Array of true values.
                signature (str): Signature for the histogram file name.
                nbins (int, optional): Number of bins for the histogram. Defaults to 10.

            Returns:
                matplotlib.figure.Figure: The generated figure.

            """
            ndata, npars = trues.shape
            navg = ndata / nbins
            ranks = self._get_ranks(samples, trues)

            fig, ax = plt.subplots(1, npars, figsize=(npars * 3, 4))
            if npars == 1:
                ax = [ax]

            for i in range(npars):
                ax[i].hist(np.array(ranks)[:, i], bins=nbins)
                ax[i].set_title(self.labels[i])
            ax[0].set_ylabel('counts')

            for axis in ax:
                axis.set_xlim(0, ranks.max())
                axis.set_xlabel('rank')
                axis.grid(visible=True)
                axis.axhline(navg, color='k')
                axis.axhline(navg - navg ** 0.5, color='k', ls="--")
                axis.axhline(navg + navg ** 0.5, color='k', ls="--")
            return fig

    def _plot_coverage(
            self, samples: np.ndarray, trues: np.ndarray, plotscatter: bool = True
        ) -> plt.Figure:
            """
            Plot the coverage of predicted percentiles against empirical percentiles.

            Args:
                samples (numpy.ndarray): Array of predicted samples.
                trues (numpy.ndarray): Array of true values.
                signature (str): Signature for the plot file name.
                plotscatter (bool, optional): Whether to plot the scatter plot. Defaults to True.

            Returns:
                matplotlib.figure.Figure: The generated figure.

            """
            ndata, npars = trues.shape
            ranks = self._get_ranks(samples, trues)

            unicov = [np.sort(np.random.uniform(0, 1, ndata)) for j in range(200)]
            unip = np.percentile(unicov, [5, 16, 84, 95], axis=0)

            fig, ax = plt.subplots(1, npars, figsize=(npars * 4, 4))
            if npars == 1:
                ax = [ax]
            cdf = np.linspace(0, 1, len(ranks))
            for i in range(npars):
                xr = np.sort(ranks[:, i])
                xr = xr / xr[-1]
                ax[i].plot(cdf, cdf, 'k--')
                if plotscatter:
                    ax[i].fill_between(cdf, unip[0], unip[-1],
                                    color='gray', alpha=0.2)
                    ax[i].fill_between(cdf, unip[1], unip[-2],
                                    color='gray', alpha=0.4)
                ax[i].plot(xr, cdf, lw=2, label='posterior')
                ax[i].set(adjustable='box', aspect='equal')
                ax[i].set_title(self.labels[i])
                ax[i].set_xlabel('Predicted Percentile')
                ax[i].set_xlim(0, 1)
                ax[i].set_ylim(0, 1)

            ax[0].set_ylabel('Empirical Percentile')
            for axis in ax:
                axis.grid(visible=True)
            return fig

    def _plot_predictions(
            self, samples: np.ndarray, trues: np.ndarray,
        ) -> plt.Figure:
            """
            Plot the mean and standard deviation of the predicted samples against
            the true values.

            Args:
                samples (np.ndarray): Array of predicted samples.
                trues (np.ndarray): Array of true values.
                signature (str): Signature for the plot.

            Returns:
                plt.Figure: The plotted figure.
            """
            npars = trues.shape[-1]
            mus, stds = samples.mean(axis=0), samples.std(axis=0)
            # print(f"mus shape: {mus.shape}, stds shape: {stds.shape}, trues shape: {trues.shape}")
            fig, axs = plt.subplots(1, npars, figsize=(npars * 4, 4))
            if npars == 1:
                axs = [axs]
            else:
                axs = axs.flatten()
            for j in range(npars):
                axs[j].errorbar(trues[:, j], mus[:, j], stds[:, j],
                                fmt="none", elinewidth=0.5, alpha=0.5)
                axs[j].plot(
                    *(2 * [np.linspace(min(trues[:, j]), max(trues[:, j]), 10)]),
                    'k--', ms=0.2, lw=0.5)
                axs[j].grid(which='both', lw=0.5)
                axs[j].set(adjustable='box', aspect='equal')
                axs[j].set_title(self.labels[j], fontsize=12)
                axs[j].set_xlabel('True')
            axs[0].set_ylabel('Predicted')

            return fig



    def _plot_TARP(
        self, posterior_samples: np.array, theta: np.array,
        references: str = "random", metric: str = "euclidean",
        bootstrap: Optional[bool] = True, norm: Optional[bool] = True,
        num_alpha_bins: Optional[int] = 10,
        num_bootstrap: Optional[int] = 100
    ) -> plt.Figure:
        """
        Plots the TARP credibility metric for the given posterior samples
        and theta values. See https://arxiv.org/abs/2302.03026 for details.

        Args:
            posterior_samples (np.array): Array of posterior samples.
            theta (np.array): Array of theta values.
            signature (str): Signature for the plot.
            references (str, optional): TARP reference type for TARP calculation. 
                Defaults to "random".
            metric (str, optional): TARP distance metric for TARP calculation. 
                Defaults to "euclidean".
            bootstrap (bool, optional): Whether to use bootstrapping for TARP error bars. 
                Defaults to False.
            norm (bool, optional): Whether to normalize the TARP metric. Defaults to True.
            num_alpha_bins (int, optional):number of bins to use for the TARP
                credibility values. Defaults to None.
            num_bootstrap (int, optional): Number of bootstrap iterations
                for TARP calculation. Defaults to 100.

        Returns:
            plt.Figure: The generated TARP plot.
        """
        ecp, alpha = tarp.get_tarp_coverage(
            posterior_samples, theta,
            references=references, metric=metric,
            norm=norm, bootstrap=bootstrap,
            num_alpha_bins=num_alpha_bins,
            num_bootstrap=num_bootstrap
        )

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.plot([0, 1], [0, 1], ls='--', color='k')
        if bootstrap:
            ecp_mean = np.mean(ecp, axis=0)
            ecp_std = np.std(ecp, axis=0)
            ax.plot(alpha, ecp_mean, label='TARP', color='b')
            ax.fill_between(alpha, ecp_mean - ecp_std, ecp_mean + ecp_std,
                            alpha=0.2, color='b')
            ax.fill_between(alpha, ecp_mean - 2 * ecp_std, ecp_mean + 2 * ecp_std,
                            alpha=0.2, color='b')
        else:
            ax.plot(alpha, ecp, label='TARP')
        ax.legend()
        ax.set_ylabel("Expected Coverage")
        ax.set_xlabel("Credibility Level")

        return fig
    

class C2STAllParameters(Callback):

    name: str = 'rankss'
    save_every: int = 0

    # observation_idx: List[int] = [i for i in range(1, observation_idx)]
    num_total_samples: int = 1_000
    # batch_size: int = 1_000

    def __init__(self, save_every: int = 0, savedir: str = None, num_total_samples: int = 1_000, observation_idx: int = 10, batch_size: int = 1_000):
        super().__init__()
        self.save_every = save_every
        self.savedir = savedir
        self.num_total_samples = num_total_samples
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
        self.observation_idx = [i for i in range(1, observation_idx)]
        self.batch_size = batch_size
        self.labels = ['$t_{end}$', '$M_{plummer}$', '$a_{plummer}$', '$M_{NFW}$', '$r_{NFW}$', '$M_{MN}$', '$a_{MN}$']
        if self.savedir is not None:
            os.makedirs(self.savedir + '/pictures', exist_ok=True)
        
    def __call__(self, logs: dict, rng: jr.PRNGKey, *args, **kwargs):
        return self._call(logs, rng, *args, **kwargs)

    def _call(self, logs: dict, rng: jr.PRNGKey, strategy: Strategy,
              train_loader: GeneratorDataloader, val_loader: GeneratorDataloader, *args, **kwargs) \
            -> Tuple[dict, jr.PRNGKey]:

        
        figure_list = []
        fig, axes = plt.subplots(1, 4, figsize=(30, 5))  # Create figure and 4 axes once

        true_theta_test_set = np.zeros((len(self.observation_idx), 7))
        predicted_theta_test_set = np.zeros((self.num_total_samples, len(self.observation_idx), 7))

        for id in self.observation_idx:

            observation, true_theta, reference_posterior = train_loader.get_observation(id)
            true_theta_test_set[id-1, :] = true_theta[0, :]

            observation = jnp.array(observation).repeat(self.num_total_samples, axis=0)
            posterior_samples, _ = strategy.sample(self.num_total_samples, rng, conditioning=observation,
                                                   batch_size=self.batch_size)
            posterior_samples['samples'] = 0.5 * (posterior_samples['samples'] + 1) * (self.high-self.low) + self.low
            predicted_theta_test_set[:, id-1, :] = posterior_samples['samples']
        print(f"Predicted theta shape: {predicted_theta_test_set.shape}")
        print(f'true_theta_test_set: {true_theta_test_set.shape}')

        # print(logs)
        # epoch = logs["epoch"]
        

        rank_plot = self._plot_ranks_histogram(
            samples=predicted_theta_test_set,
            trues=true_theta_test_set,
            nbins=10)
        
        coverage_plot = self._plot_coverage(
            samples=predicted_theta_test_set,
            trues=true_theta_test_set,
            plotscatter=True,
        )

        prediction_plot = self._plot_predictions(
            samples=predicted_theta_test_set,
            trues=true_theta_test_set,
        )

        tarp_plot = self._plot_TARP(
            posterior_samples=predicted_theta_test_set,
            theta=true_theta_test_set,
        )
        
        figure_list.append(rank_plot)
        figure_list.append(coverage_plot)
        figure_list.append(prediction_plot)

        if self.savedir is not None:
                rank_plot.savefig(f"{self.savedir}/pictures/ranks_histogram.pdf")
                coverage_plot.savefig(f"{self.savedir}/pictures/coverage_plot.pdf")
                prediction_plot.savefig(f"{self.savedir}/pictures/prediction_plot.pdf")
                tarp_plot.savefig(f"{self.savedir}/pictures/tarp_plot.pdf")

        figure_list.append(fig)

        np.savez(f"{self.savedir}/pictures/predicted_theta_test_set.npz",
                 predicted_theta=predicted_theta_test_set,
                 true_theta=true_theta_test_set)

        logs['posteriors'] = [wandb.Image(fig) for fig in figure_list]

        for fig in figure_list:
            plt.close(fig)

        return logs, rng
    
    def on_test(self, logs: dict, rng: jr.PRNGKey, *args, **kwargs):
            return self.__call__(logs, rng, *args, **kwargs)
    
    def _get_ranks(
            self,
            samples: np.array,
            trues: np.array,
        ) -> np.array:
            """Get the marginal ranks of the true parameters in the posterior samples.

            Args:
                samples (np.array): posterior samples of shape (nsamples, ndata, npars)
                trues (np.array): true parameters of shape (ndata, npars)

            Returns:
                np.array: ranks of the true parameters in the posterior samples 
                    of shape (ndata, npars)
            """
            ranks = (samples < trues[None, ...]).sum(axis=0)
            return ranks


    def _plot_ranks_histogram(
            self, samples: np.ndarray, trues: np.ndarray, nbins: int = 10
        ) -> plt.Figure:
            """
            Plot a histogram of ranks for each parameter.

            Args:
                samples (numpy.ndarray): List of samples.
                trues (numpy.ndarray): Array of true values.
                signature (str): Signature for the histogram file name.
                nbins (int, optional): Number of bins for the histogram. Defaults to 10.

            Returns:
                matplotlib.figure.Figure: The generated figure.

            """
            ndata, npars = trues.shape
            navg = ndata / nbins
            ranks = self._get_ranks(samples, trues)

            fig, ax = plt.subplots(1, npars, figsize=(npars * 3, 4))
            if npars == 1:
                ax = [ax]

            for i in range(npars):
                ax[i].hist(np.array(ranks)[:, i], bins=nbins)
                ax[i].set_title(self.labels[i])
            ax[0].set_ylabel('counts')

            for axis in ax:
                axis.set_xlim(0, ranks.max())
                axis.set_xlabel('rank')
                axis.grid(visible=True)
                axis.axhline(navg, color='k')
                axis.axhline(navg - navg ** 0.5, color='k', ls="--")
                axis.axhline(navg + navg ** 0.5, color='k', ls="--")
            return fig

    def _plot_coverage(
            self, samples: np.ndarray, trues: np.ndarray, plotscatter: bool = True
        ) -> plt.Figure:
            """
            Plot the coverage of predicted percentiles against empirical percentiles.

            Args:
                samples (numpy.ndarray): Array of predicted samples.
                trues (numpy.ndarray): Array of true values.
                signature (str): Signature for the plot file name.
                plotscatter (bool, optional): Whether to plot the scatter plot. Defaults to True.

            Returns:
                matplotlib.figure.Figure: The generated figure.

            """
            ndata, npars = trues.shape
            ranks = self._get_ranks(samples, trues)

            unicov = [np.sort(np.random.uniform(0, 1, ndata)) for j in range(200)]
            unip = np.percentile(unicov, [5, 16, 84, 95], axis=0)

            fig, ax = plt.subplots(1, npars, figsize=(npars * 4, 4))
            if npars == 1:
                ax = [ax]
            cdf = np.linspace(0, 1, len(ranks))
            for i in range(npars):
                xr = np.sort(ranks[:, i])
                xr = xr / xr[-1]
                ax[i].plot(cdf, cdf, 'k--')
                if plotscatter:
                    ax[i].fill_between(cdf, unip[0], unip[-1],
                                    color='gray', alpha=0.2)
                    ax[i].fill_between(cdf, unip[1], unip[-2],
                                    color='gray', alpha=0.4)
                ax[i].plot(xr, cdf, lw=2, label='posterior')
                ax[i].set(adjustable='box', aspect='equal')
                ax[i].set_title(self.labels[i])
                ax[i].set_xlabel('Predicted Percentile')
                ax[i].set_xlim(0, 1)
                ax[i].set_ylim(0, 1)

            ax[0].set_ylabel('Empirical Percentile')
            for axis in ax:
                axis.grid(visible=True)
            return fig

    def _plot_predictions(
            self, samples: np.ndarray, trues: np.ndarray,
        ) -> plt.Figure:
            """
            Plot the mean and standard deviation of the predicted samples against
            the true values.

            Args:
                samples (np.ndarray): Array of predicted samples.
                trues (np.ndarray): Array of true values.
                signature (str): Signature for the plot.

            Returns:
                plt.Figure: The plotted figure.
            """
            npars = trues.shape[-1]
            mus, stds = samples.mean(axis=0), samples.std(axis=0)
            # print(f"mus shape: {mus.shape}, stds shape: {stds.shape}, trues shape: {trues.shape}")
            fig, axs = plt.subplots(1, npars, figsize=(npars * 4, 4))
            if npars == 1:
                axs = [axs]
            else:
                axs = axs.flatten()
            for j in range(npars):
                axs[j].errorbar(trues[:, j], mus[:, j], stds[:, j],
                                fmt="none", elinewidth=0.5, alpha=0.5)
                axs[j].plot(
                    *(2 * [np.linspace(min(trues[:, j]), max(trues[:, j]), 10)]),
                    'k--', ms=0.2, lw=0.5)
                axs[j].grid(which='both', lw=0.5)
                axs[j].set(adjustable='box', aspect='equal')
                axs[j].set_title(self.labels[j], fontsize=12)
                axs[j].set_xlabel('True')
            axs[0].set_ylabel('Predicted')

            return fig



    def _plot_TARP(
        self, posterior_samples: np.array, theta: np.array,
        references: str = "random", metric: str = "euclidean",
        bootstrap: Optional[bool] = True, norm: Optional[bool] = True,
        num_alpha_bins: Optional[int] = 10,
        num_bootstrap: Optional[int] = 100
    ) -> plt.Figure:
        """
        Plots the TARP credibility metric for the given posterior samples
        and theta values. See https://arxiv.org/abs/2302.03026 for details.

        Args:
            posterior_samples (np.array): Array of posterior samples.
            theta (np.array): Array of theta values.
            signature (str): Signature for the plot.
            references (str, optional): TARP reference type for TARP calculation. 
                Defaults to "random".
            metric (str, optional): TARP distance metric for TARP calculation. 
                Defaults to "euclidean".
            bootstrap (bool, optional): Whether to use bootstrapping for TARP error bars. 
                Defaults to False.
            norm (bool, optional): Whether to normalize the TARP metric. Defaults to True.
            num_alpha_bins (int, optional):number of bins to use for the TARP
                credibility values. Defaults to None.
            num_bootstrap (int, optional): Number of bootstrap iterations
                for TARP calculation. Defaults to 100.

        Returns:
            plt.Figure: The generated TARP plot.
        """
        ecp, alpha = tarp.get_tarp_coverage(
            posterior_samples, theta,
            references=references, metric=metric,
            norm=norm, bootstrap=bootstrap,
            num_alpha_bins=num_alpha_bins,
            num_bootstrap=num_bootstrap
        )

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.plot([0, 1], [0, 1], ls='--', color='k')
        if bootstrap:
            ecp_mean = np.mean(ecp, axis=0)
            ecp_std = np.std(ecp, axis=0)
            ax.plot(alpha, ecp_mean, label='TARP', color='b')
            ax.fill_between(alpha, ecp_mean - ecp_std, ecp_mean + ecp_std,
                            alpha=0.2, color='b')
            ax.fill_between(alpha, ecp_mean - 2 * ecp_std, ecp_mean + 2 * ecp_std,
                            alpha=0.2, color='b')
        else:
            ax.plot(alpha, ecp, label='TARP')
        ax.legend()
        ax.set_ylabel("Expected Coverage")
        ax.set_xlabel("Credibility Level")

        return fig


class C2ST_OT_AllParameters(Callback):

    name: str = 'rankss'
    save_every: int = 0

    # observation_idx: List[int] = [i for i in range(1, observation_idx)]
    num_total_samples: int = 1_000
    # batch_size: int = 1_000

    def __init__(self, save_every: int = 0, savedir: str = None, num_total_samples: int = 1_000, observation_idx: int = 10, batch_size: int = 1_000):
        super().__init__()
        self.save_every = save_every
        self.savedir = savedir
        self.num_total_samples = num_total_samples
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
        self.observation_idx = [i for i in range(1, observation_idx)]
        self.batch_size = batch_size
        self.labels = ['$t_{end}$', '$M_{plummer}$', '$a_{plummer}$', '$M_{NFW}$', '$r_{NFW}$', '$M_{MN}$', '$a_{MN}$']
        if self.savedir is not None:
            os.makedirs(self.savedir + '/pictures', exist_ok=True)
        
    def __call__(self, logs: dict, rng: jr.PRNGKey, *args, **kwargs):
        return self._call(logs, rng, *args, **kwargs)

    def _call(self, logs: dict, rng: jr.PRNGKey, strategy: Strategy,
              train_loader: GeneratorDataloader, val_loader: GeneratorDataloader, *args, **kwargs) \
            -> Tuple[dict, jr.PRNGKey]:

        
        figure_list = []
        fig, axes = plt.subplots(1, 4, figsize=(30, 5))  # Create figure and 4 axes once

        true_theta_test_set = np.zeros((len(self.observation_idx), 7))
        predicted_theta_test_set = np.zeros((self.num_total_samples, len(self.observation_idx), 7))

        for id in self.observation_idx:

            observation, true_theta, reference_posterior = train_loader.get_observation(id)
            true_theta_test_set[id-1, :] = true_theta[0, :]

            observation = jnp.array(observation).repeat(self.num_total_samples, axis=0)
            posterior_samples, _ = strategy.sample(self.num_total_samples, rng, conditioning=observation,
                                                   batch_size=self.batch_size)
            posterior_samples['samples'] = norm.cdf(posterior_samples['samples']) * (self.high - self.low) + self.low
            predicted_theta_test_set[:, id-1, :] = posterior_samples['samples']
        print(f"Predicted theta shape: {predicted_theta_test_set.shape}")
        print(f'true_theta_test_set: {true_theta_test_set.shape}')

        # print(logs)
        # epoch = logs["epoch"]
        

        rank_plot = self._plot_ranks_histogram(
            samples=predicted_theta_test_set,
            trues=true_theta_test_set,
            nbins=10)
        
        coverage_plot = self._plot_coverage(
            samples=predicted_theta_test_set,
            trues=true_theta_test_set,
            plotscatter=True,
        )

        prediction_plot = self._plot_predictions(
            samples=predicted_theta_test_set,
            trues=true_theta_test_set,
        )

        tarp_plot = self._plot_TARP(
            posterior_samples=predicted_theta_test_set,
            theta=true_theta_test_set,
        )
        
        figure_list.append(rank_plot)
        figure_list.append(coverage_plot)
        figure_list.append(prediction_plot)

        if self.savedir is not None:
                rank_plot.savefig(f"{self.savedir}/pictures/ranks_histogram.pdf")
                coverage_plot.savefig(f"{self.savedir}/pictures/coverage_plot.pdf")
                prediction_plot.savefig(f"{self.savedir}/pictures/prediction_plot.pdf")
                tarp_plot.savefig(f"{self.savedir}/pictures/tarp_plot.pdf")

        figure_list.append(fig)

        np.savez(f"{self.savedir}/pictures/predicted_theta_test_set.npz",
                 predicted_theta=predicted_theta_test_set,
                 true_theta=true_theta_test_set)

        logs['posteriors'] = [wandb.Image(fig) for fig in figure_list]

        for fig in figure_list:
            plt.close(fig)

        return logs, rng
    
    def on_test(self, logs: dict, rng: jr.PRNGKey, *args, **kwargs):
            return self.__call__(logs, rng, *args, **kwargs)
    
    def _get_ranks(
            self,
            samples: np.array,
            trues: np.array,
        ) -> np.array:
            """Get the marginal ranks of the true parameters in the posterior samples.

            Args:
                samples (np.array): posterior samples of shape (nsamples, ndata, npars)
                trues (np.array): true parameters of shape (ndata, npars)

            Returns:
                np.array: ranks of the true parameters in the posterior samples 
                    of shape (ndata, npars)
            """
            ranks = (samples < trues[None, ...]).sum(axis=0)
            return ranks


    def _plot_ranks_histogram(
            self, samples: np.ndarray, trues: np.ndarray, nbins: int = 10
        ) -> plt.Figure:
            """
            Plot a histogram of ranks for each parameter.

            Args:
                samples (numpy.ndarray): List of samples.
                trues (numpy.ndarray): Array of true values.
                signature (str): Signature for the histogram file name.
                nbins (int, optional): Number of bins for the histogram. Defaults to 10.

            Returns:
                matplotlib.figure.Figure: The generated figure.

            """
            ndata, npars = trues.shape
            navg = ndata / nbins
            ranks = self._get_ranks(samples, trues)

            fig, ax = plt.subplots(1, npars, figsize=(npars * 3, 4))
            if npars == 1:
                ax = [ax]

            for i in range(npars):
                ax[i].hist(np.array(ranks)[:, i], bins=nbins)
                ax[i].set_title(self.labels[i])
            ax[0].set_ylabel('counts')

            for axis in ax:
                axis.set_xlim(0, ranks.max())
                axis.set_xlabel('rank')
                axis.grid(visible=True)
                axis.axhline(navg, color='k')
                axis.axhline(navg - navg ** 0.5, color='k', ls="--")
                axis.axhline(navg + navg ** 0.5, color='k', ls="--")
            return fig

    def _plot_coverage(
            self, samples: np.ndarray, trues: np.ndarray, plotscatter: bool = True
        ) -> plt.Figure:
            """
            Plot the coverage of predicted percentiles against empirical percentiles.

            Args:
                samples (numpy.ndarray): Array of predicted samples.
                trues (numpy.ndarray): Array of true values.
                signature (str): Signature for the plot file name.
                plotscatter (bool, optional): Whether to plot the scatter plot. Defaults to True.

            Returns:
                matplotlib.figure.Figure: The generated figure.

            """
            ndata, npars = trues.shape
            ranks = self._get_ranks(samples, trues)

            unicov = [np.sort(np.random.uniform(0, 1, ndata)) for j in range(200)]
            unip = np.percentile(unicov, [5, 16, 84, 95], axis=0)

            fig, ax = plt.subplots(1, npars, figsize=(npars * 4, 4))
            if npars == 1:
                ax = [ax]
            cdf = np.linspace(0, 1, len(ranks))
            for i in range(npars):
                xr = np.sort(ranks[:, i])
                xr = xr / xr[-1]
                ax[i].plot(cdf, cdf, 'k--')
                if plotscatter:
                    ax[i].fill_between(cdf, unip[0], unip[-1],
                                    color='gray', alpha=0.2)
                    ax[i].fill_between(cdf, unip[1], unip[-2],
                                    color='gray', alpha=0.4)
                ax[i].plot(xr, cdf, lw=2, label='posterior')
                ax[i].set(adjustable='box', aspect='equal')
                ax[i].set_title(self.labels[i])
                ax[i].set_xlabel('Predicted Percentile')
                ax[i].set_xlim(0, 1)
                ax[i].set_ylim(0, 1)

            ax[0].set_ylabel('Empirical Percentile')
            for axis in ax:
                axis.grid(visible=True)
            return fig

    def _plot_predictions(
            self, samples: np.ndarray, trues: np.ndarray,
        ) -> plt.Figure:
            """
            Plot the mean and standard deviation of the predicted samples against
            the true values.

            Args:
                samples (np.ndarray): Array of predicted samples.
                trues (np.ndarray): Array of true values.
                signature (str): Signature for the plot.

            Returns:
                plt.Figure: The plotted figure.
            """
            npars = trues.shape[-1]
            mus, stds = samples.mean(axis=0), samples.std(axis=0)
            # print(f"mus shape: {mus.shape}, stds shape: {stds.shape}, trues shape: {trues.shape}")
            fig, axs = plt.subplots(1, npars, figsize=(npars * 4, 4))
            if npars == 1:
                axs = [axs]
            else:
                axs = axs.flatten()
            for j in range(npars):
                axs[j].errorbar(trues[:, j], mus[:, j], stds[:, j],
                                fmt="none", elinewidth=0.5, alpha=0.5)
                axs[j].plot(
                    *(2 * [np.linspace(min(trues[:, j]), max(trues[:, j]), 10)]),
                    'k--', ms=0.2, lw=0.5)
                axs[j].grid(which='both', lw=0.5)
                axs[j].set(adjustable='box', aspect='equal')
                axs[j].set_title(self.labels[j], fontsize=12)
                axs[j].set_xlabel('True')
            axs[0].set_ylabel('Predicted')

            return fig



    def _plot_TARP(
        self, posterior_samples: np.array, theta: np.array,
        references: str = "random", metric: str = "euclidean",
        bootstrap: Optional[bool] = True, norm: Optional[bool] = True,
        num_alpha_bins: Optional[int] = 10,
        num_bootstrap: Optional[int] = 100
    ) -> plt.Figure:
        """
        Plots the TARP credibility metric for the given posterior samples
        and theta values. See https://arxiv.org/abs/2302.03026 for details.

        Args:
            posterior_samples (np.array): Array of posterior samples.
            theta (np.array): Array of theta values.
            signature (str): Signature for the plot.
            references (str, optional): TARP reference type for TARP calculation. 
                Defaults to "random".
            metric (str, optional): TARP distance metric for TARP calculation. 
                Defaults to "euclidean".
            bootstrap (bool, optional): Whether to use bootstrapping for TARP error bars. 
                Defaults to False.
            norm (bool, optional): Whether to normalize the TARP metric. Defaults to True.
            num_alpha_bins (int, optional):number of bins to use for the TARP
                credibility values. Defaults to None.
            num_bootstrap (int, optional): Number of bootstrap iterations
                for TARP calculation. Defaults to 100.

        Returns:
            plt.Figure: The generated TARP plot.
        """
        ecp, alpha = tarp.get_tarp_coverage(
            posterior_samples, theta,
            references=references, metric=metric,
            norm=norm, bootstrap=bootstrap,
            num_alpha_bins=num_alpha_bins,
            num_bootstrap=num_bootstrap
        )

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.plot([0, 1], [0, 1], ls='--', color='k')
        if bootstrap:
            ecp_mean = np.mean(ecp, axis=0)
            ecp_std = np.std(ecp, axis=0)
            ax.plot(alpha, ecp_mean, label='TARP', color='b')
            ax.fill_between(alpha, ecp_mean - ecp_std, ecp_mean + ecp_std,
                            alpha=0.2, color='b')
            ax.fill_between(alpha, ecp_mean - 2 * ecp_std, ecp_mean + 2 * ecp_std,
                            alpha=0.2, color='b')
        else:
            ax.plot(alpha, ecp, label='TARP')
        ax.legend()
        ax.set_ylabel("Expected Coverage")
        ax.set_xlabel("Credibility Level")

        return fig
    
# class C2ST(Callback):

#     name: str = 'c2st'
#     num_observations: int = 10
#     num_posterior_samples: int = 10000
#     batch_size: int = 128
#     metrics: List[str] = ['c2st']

#     def __init__(self, batch_size: int = 512, metrics: List[str] = None):
#         super().__init__()
#         self.batch_size = batch_size
#         if metrics is not None:
#             self.metrics = metrics

#     def __call__(self, logs: dict, rng: jr.PRNGKey, *args, **kwargs):
#         return self._call(logs, rng, *args, **kwargs)

#     def _call(self, logs: dict, rng: jr.PRNGKey, strategy: Strategy,
#               train_loader: GeneratorDataloader, val_loader: GeneratorDataloader, *args, **kwargs) \
#                 -> Tuple[dict, jr.PRNGKey]:

#         scores = {metric: 0.0 for metric in self.metrics}

#         p = tqdm(range(1, self.num_observations+1))

#         for observation_idx in p:

#             observation, true_theta, reference_posterior = train_loader.get_observation(observation_idx)

#             observation = jnp.array(observation).repeat(self.num_posterior_samples, axis=0)
#             posterior_samples, _ = strategy.sample(self.num_posterior_samples, rng,
#                                                    conditioning=observation, batch_size=self.batch_size)

#             if 'c2st' in self.metrics:
#                 score_idx = c2st(reference_posterior, posterior_samples["samples"])
#                 scores['c2st'] += score_idx
#                 p.set_description(f'Observation {observation_idx} C2ST: {score_idx:.3f}')

#             if 'mmd' in self.metrics:
#                 score_mmd = mmd(np.array(reference_posterior), np.array(posterior_samples["samples"]))
#                 scores['mmd'] += score_mmd
#                 print(f'Observation {observation_idx} MMD: {score_mmd:.3f}')

#         for metric in self.metrics:
#             scores[metric] = scores[metric] / self.num_observations
#             logs[metric] = scores[metric]

#         return logs, rng

#     def on_test(self, logs: dict, rng: jr.PRNGKey, *args, **kwargs):
#         return self.__call__(logs, rng, *args, **kwargs)