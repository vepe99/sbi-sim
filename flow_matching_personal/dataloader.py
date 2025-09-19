
from abc import ABC, abstractmethod
from math import ceil
from pathlib import Path
from typing import Iterable, Tuple

import jax
import jax.random as jr
import jax.numpy as jnp
from jax.scipy.stats import norm
import numpy as np

import pandas as pd
from math import log10

class GeneratorDataloaderAllParameters_position_OT_newprior_numpy(Iterable):
    num_samples: int
    rng: jr.PRNGKey
    num_steps: int
    batch_size: int
    X: jnp.ndarray
    y: jnp.ndarray
    normalize: bool
    

    def __iter__(self):
        self.current_step = 1

        if not self.replacement:
            perm = jr.permutation(self.rng, self.X.shape[0])
            self.X = self.X[perm]
            self.y = self.y[perm]
            self.rng = jr.split(self.rng)[0]

        return self

    def load_file(self, file):

        if self.use_jax:
            return jnp.load(file, allow_pickle=True)
        else:
            print('using numpy to load')
            return np.load(file, allow_pickle=True)

    def get_observation(self, idx):

        base_dir = Path(self.DATA_ROOT).joinpath(f'num_observation_{idx}')

        if not base_dir.exists():
            raise FileNotFoundError(f"Directory {base_dir} does not exist")

        observation = self.load_file(base_dir.joinpath(f"observation.npy"))

        true_theta = self.load_file(base_dir.joinpath(f"true_parameters.npy"))
        if not self.use_log:
            if self.use_jax:
                true_theta = true_theta.at[:, [1, 3, 5]].set(10**true_theta[:, [1, 3, 5]])  # convert to original scale
            else:
                true_theta[:, [1, 3, 5]] = 10**true_theta[:,[1, 3, 5]]  # convert to original scale
        else: 
            pass

        reference_posterior = self.load_file(base_dir.joinpath(f"reference_posterior_samples.npy"))

        # if self.normalize:
        #     # true_theta = (true_theta - self.mean_X) / self.std_X
        #     # reference_posterior = (reference_posterior - self.mean_X) / self.std_X
        #     # observation = (observation - self.mean_y) / self.std_y
        #     # Normalize each histogram to sum to 1 (convert to probability distribution)
        #     total = jnp.sum(observation, axis=(1, 2), keepdims=True)
        #     observation = observation / (total + 1e-8)  # Add epsilon to avoid division by zero

        return observation, true_theta, reference_posterior
    

    def __init__(self, dataset: str, DATA_ROOT: str, num_samples: int, split: Tuple[int], seed: jr.PRNGKey,
                 num_steps: int, batch_size: int, normalize: bool = True, replacement: bool = False, return_histogram: bool = False,
                 use_jax: bool = False, use_log: bool = False):
        """
        :param dataset:
        :param DATA_ROOT:
        :param num_samples:
        :param split:
        :param seed:
        :param num_steps:
        :param batch_size:
        :param normalize:
        :param replacement:
        """
        print(dataset)

        DATA_ROOT = f'/export/data/vgiusepp/odisseo_data/data_varying_position_newprior/sbi_sim/data/sbi-benchmarks/{dataset}/'
        print(f"DATA_ROOT: {DATA_ROOT}")
        path = Path(DATA_ROOT)

        if num_samples not in [100, 1000, 10000, 100_000, 250_000, 1_000_000, 10000000]:
            raise ValueError(f"num_samples ({num_samples}) must be in [100, 1000, 10000, 100000, 250_000, 1000000]")

        self.replacement = replacement

        self.normalize = normalize
        self.DATA_ROOT = DATA_ROOT
        self.dataset = dataset
        self.batch_size = batch_size
        self.rng = jr.PRNGKey(seed)

        self.current_step = 1
        self.dim_theta = 10

        split_start = split[0]
        split_end = split[1]

        self.load_data_from_hf(dataset, Path(DATA_ROOT))
        self.use_jax = use_jax
        self.use_log = use_log

        self.y = self.load_file(path.joinpath(f"x_{num_samples}.npy"))
        self.X = self.load_file(path.joinpath(f"theta_{num_samples}.npy"))
        if not self.use_log:
            if self.use_jax:
                print('using jax to load')
                self.X = self.X.at[:, [1, 3, 5]].set(10**self.X[:, [1, 3, 5]])  # convert to original scale
            else:
                self.X[:, [1, 3, 5]] = 10**self.X[:, [1, 3, 5]]  # convert to original scale
        else:
            pass

        # samples seem to be sorted when loading from file; make sure to shuffle them!
        permutation_seed = jr.PRNGKey(0)
        perm = jr.permutation(permutation_seed, num_samples)
        self.X = self.X[perm]
        self.y = self.y[perm]

        self.mean_X = self.X.mean(axis=0)
        self.std_X = self.X.std(axis=0)
        

        # self.mean_y = self.y.mean(axis=0)
        # self.std_y = self.y.std(axis=0)

        if self.normalize:
            # Convert uniform to [0,1] first
            self.X = (self.X - self.mean_X) / self.std_X
            
        self.X = self.X[split_start:split_end]
        self.y = self.y[split_start:split_end]

        self.num_samples = self.X.shape[0]
        if not self.replacement:
            self.num_steps = int(ceil(self.num_samples / self.batch_size))
            print(f"Dataloader sampling without replacements (num_steps: {self.num_steps})")
        else:
            self.num_steps = num_steps

    def __next__(self):
        if self.current_step <= self.num_steps:

            if not self.replacement:
                X = jnp.array(self.X[self.batch_size * (self.current_step-1): self.batch_size * self.current_step])
                y = jnp.array(self.y[self.batch_size * (self.current_step-1): self.batch_size * self.current_step])
            else:
                X = jnp.array(self.X[jr.choice(self.rng, self.num_samples, (self.batch_size,))])
                y = jnp.array(self.y[jr.choice(self.rng, self.num_samples, (self.batch_size,))])
                self.rng = jr.split(self.rng)[0]

            self.current_step += 1
            return {'parameters': X, 'weighting': jnp.ones((self.batch_size, 1)), 'conditioning': y}

        else:

            raise StopIteration

    def __len__(self):
        return self.num_steps
    
    def load_data_from_hf(self, dataset, path):

        path_dir = Path(path)
        if not path_dir.exists():

            # download folder from huggingface
            from huggingface_hub import hf_hub_download, snapshot_download
            print(f"Downloading dataset {dataset} from huggingface...")

            snapshot_download(repo_id=f"thuerey-group/sbi-toy-datasets", 
                          repo_type="dataset",
                          local_dir=path,
                          allow_patterns=[f"{dataset}/*"])
            
        else:

            print(f"Loading {dataset} locally from existing directory {path_dir}")


class OdisseoOTAllParametersPosition_newprior(GeneratorDataloaderAllParameters_position_OT_newprior_numpy):

    def __init__(self, *args, **kwargs,):
        super().__init__("odisseo_AllParametersPosition_newprior", *args, **kwargs, )
