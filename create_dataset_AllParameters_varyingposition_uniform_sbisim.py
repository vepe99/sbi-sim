import os

# import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Set to the GPU you want to use, or '0' for the first GPU
# os.environ['JAX_PLATFORM_NAME'] = 'cpu'

# from autocvd import autocvd
# autocvd(num_gpus = 1)

import numpy as np
import pandas as pd 
from tqdm import tqdm 
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from astropy import units as u
from odisseo.units import CodeUnits


def to_inference_parameters(theta, ):
        """
        convert from simulation parameters to inference parameters
        """
        # code_length = 10.0 * u.kpc
        # code_mass = 1e4 * u.Msun
        # code_time = 3 * u.Gyr
        # code_units = CodeUnits(code_length, code_mass, G=1, unit_time = code_time )  

        theta[0] = theta[0] # t_end is already in Gyr
        theta[1] = np.log10(theta[1]).item() # Plummer mass is already in Msun
        theta[2] = theta[2] # Plummer a
        theta[3] = np.log10(theta[3]).item()  # NFW Mvir
        theta[4] = theta[4]  # NFW r_s
        theta[5] = np.log10(theta[5]).item() # MN M
        theta[6] = theta[6] # MN a
        theta[7] = theta[7] #x is already in kpc
        theta[8] = theta[8] #y is already in kpc
        theta[9] = theta[9] #z is already in kpc
        theta[10] = theta[10] #vx is already in km/s
        theta[11] = theta[11] #vy is already in km/s
        theta[12] = theta[12] #vz is already in km/s
        return theta

def convert_to_single_npy(path_stored, path_to_save, num_simulations=1000):
    print('start converting all the npz files to a single npy file')
    all_data_path = [os.path.join(path_stored, f) for f in sorted(os.listdir(path_stored))[:num_simulations] if f.endswith('.npz') and 'file' in f]
    x = []
    theta = []
    observation = []
    true_theta = []
    reference_posterior = []
    
    for file_path in tqdm(all_data_path):
        data = np.load(file_path)
        x.append(data['x']) #histogram
        theta.append(to_inference_parameters(data['theta'])) # we keep only the total integration time, the mass of Plummer, NFW and MN
    observation.append(data['x'])
    true_theta.append(to_inference_parameters(data['theta']))
    reference_posterior.append([])
    print('x shape:', np.array(x).shape)
    print('theta shape:', np.array(theta).shape)
    print('observation shape:', np.array(observation).shape)
    print('true_theta shape:', np.array(true_theta).shape)
    print('reference_posterior shape:', np.array(reference_posterior).shape)


    np.save(os.path.join(path_to_save, f'x_{num_simulations}.npy'), x)
    np.save(os.path.join(path_to_save, f'theta_{num_simulations}.npy'), theta)
    np.save(os.path.join(path_to_save, f'observation.npy'), observation)
    np.save(os.path.join(path_to_save, f'true_parameters.npy'), true_theta)
    np.save(os.path.join(path_to_save, f'reference_posterior_samples.npy'), reference_posterior)
    
    print(f'done converting all the npz files to a single npz file in the folder {path_to_save}')

def num_observation_npy(path_stored, path_to_save, simulation_observation_index=1_000):
    print('start converting num_observation files to npy')
    all_data_path = [os.path.join(path_stored, f) for f in sorted(os.listdir(path_stored))[simulation_observation_index:simulation_observation_index+10] if f.endswith('.npz') and 'file' in f]
    x = []
    theta = []
    i=1
    for file_path in tqdm(all_data_path):
        path_to_save_num_observation = os.path.join(path_to_save, f'num_observation_{i}')
        if not os.path.exists(path_to_save_num_observation):
            os.makedirs(path_to_save_num_observation)
        data = np.load(file_path)
        x.append(data['x'][:1000]) #histogram
        theta.append(to_inference_parameters(data['theta'])) 

        np.save(os.path.join(path_to_save_num_observation, f'observation.npy'), x)
        np.save(os.path.join(path_to_save_num_observation, f'true_parameters.npy'), theta)
        i += 1
    print('x shape:', np.array(x).shape)
    print('theta shape:', np.array(theta).shape)
    print(f'done converting all the num_observation files in the folder {path_to_save}')

def create_dataset(path_stored, path_to_save, num_simulations=[10_000, 100_000], num_observations=10, seed=42):
    """
    Create multiple datasets from the stored npz files with non-overlapping random indices.
    Also creates observation files using unused indices.
    """
    print('Creating datasets with non-overlapping random indices')
    
    # Get all available files
    all_files = [f for f in sorted(os.listdir(path_stored)) if f.endswith('.npz') and 'file' in f]
    total_files = len(all_files)
    
    print(f"Found {total_files} total files")
    
    # Check if we have enough files (training + single observation + multiple observations)
    total_needed = sum(num_simulations) + 1 + num_observations  # +1 for single observation.npy
    if total_needed > total_files:
        raise ValueError(f"Not enough files! Need {total_needed}, but only have {total_files}")
    
    # Create random permutation of all file indices
    rng = jax.random.PRNGKey(seed)
    all_indices = jax.random.permutation(rng, jnp.arange(total_files))
    
    # Split indices for each dataset size (non-overlapping)
    current_start = 0
    dataset_indices = {}
    
    # 1. Create training datasets
    for num_sims in num_simulations:
        # Get non-overlapping slice of indices
        indices = all_indices[current_start:current_start + num_sims]
        dataset_indices[num_sims] = indices
        current_start += num_sims
        
        print(f"Dataset {num_sims}: using indices {current_start - num_sims} to {current_start - 1}")
    
    # Process each training dataset
    for num_sims, indices in dataset_indices.items():
        print(f"\nProcessing dataset with {num_sims} simulations...")
        
        # Get file paths for this dataset
        selected_files = [all_files[i] for i in indices]
        all_data_paths = [os.path.join(path_stored, f) for f in selected_files]
        
        x = []
        theta = []
        
        for file_path in tqdm(all_data_paths, desc=f"Processing {num_sims} files"):
            data = np.load(file_path)
            x.append(data['x'][:1000])  # histogram
            theta.append(to_inference_parameters(data['theta']))  # selected parameters
        
        print(f'x shape: {np.array(x).shape}')
        print(f'theta shape: {np.array(theta).shape}')
        
        # Save dataset
        np.save(os.path.join(path_to_save, f'x_{num_sims}.npy'), x)
        np.save(os.path.join(path_to_save, f'theta_{num_sims}.npy'), theta)
        
        print(f'Saved dataset with {num_sims} simulations')
    
    # 2. Create single observation files using next available file
    if current_start < total_files:
        print(f"\nCreating single observation from file index {current_start}")
        observation_file = all_files[current_start]
        data = np.load(os.path.join(path_stored, observation_file))
        
        observation = [data['x'][:1000]]
        true_theta = [to_inference_parameters(data['theta'])]
        reference_posterior = [[]]
        
        np.save(os.path.join(path_to_save, f'observation.npy'), observation)
        np.save(os.path.join(path_to_save, f'true_parameters.npy'), true_theta)
        np.save(os.path.join(path_to_save, f'reference_posterior_samples.npy'), reference_posterior)
        
        print('Saved single observation files')
        current_start += 1
    
    # 3. Create multiple observation files (num_observation_npy functionality)
    print(f'\nCreating {num_observations} observation files from indices {current_start} to {current_start + num_observations - 1}')
    observation_indices = all_indices[current_start:current_start + num_observations]
    
    x_all = []
    theta_all = []
    
    for i, idx in enumerate(tqdm(observation_indices, desc="Creating observation files"), 1):
        file_path = os.path.join(path_stored, all_files[idx])
        
        # Create directory for this observation
        path_to_save_num_observation = os.path.join(path_to_save, f'num_observation_3e5_{i}')
        if not os.path.exists(path_to_save_num_observation):
            os.makedirs(path_to_save_num_observation)
        
        # Load and process data
        data = np.load(file_path)
        x_single = [data['x']]  # histogram - wrap in list for single observation
        theta_single = [to_inference_parameters(data['theta'])]  # selected parameters
        reference_posterior = [[]]
        
        # Save individual observation files
        np.save(os.path.join(path_to_save_num_observation, f'observation.npy'), x_single)
        np.save(os.path.join(path_to_save_num_observation, f'true_parameters.npy'), theta_single)
        np.save(os.path.join(path_to_save_num_observation, f'reference_posterior_samples.npy'), reference_posterior)
        
        # Accumulate for logging
        x_all.append(data['x'])
        theta_all.append(to_inference_parameters(data['theta']))

    print(f'Multiple observations - x shape: {np.array(x_all).shape}')
    print(f'Multiple observations - theta shape: {np.array(theta_all).shape}')
    print(f'Created {num_observations} observation directories')
    
    print(f'\nDone creating all datasets and observations in {path_to_save}')
    
    # Return summary of what was created
    return {
        'dataset_indices': dataset_indices,
        'single_observation_index': current_start - 1,
        'multiple_observation_indices': observation_indices.tolist(),
        'total_files_used': current_start + num_observations
    }

if __name__ == "__main__":
    path_stored = '/export/data/vgiusepp/odisseo_data/data_varying_position_uniform_prior/'
    path_to_save = '/export/data/vgiusepp/odisseo_data/data_varying_position_uniform_prior/sbi_sim/data/sbi-benchmarks/odisseo_AllParametersPosition_uniformprior/'
 
    # Create everything in one call
    summary = create_dataset(
        path_stored, 
        path_to_save, 
        num_simulations=[300_000], 
        num_observations=1000, 
        seed=42
    )    
    
    print("\nSummary:")
    print(f"Training datasets created: {list(summary['dataset_indices'].keys())}")
    print(f"Single observation index: {summary['single_observation_index']}")
    print(f"Multiple observation indices: {summary['multiple_observation_indices']}")
    print(f"Total files used: {summary['total_files_used']}")


    # true_observation_GD1 = np.load(os.path.join(path_stored, 'true.npz'))
    # x = true_observation_GD1['x'][:1000]  
    # theta = true_observation_GD1['theta']
    # reference_posterior = [[]]

    # # Create directory for the true observation
    # path_to_save_true_observation = os.path.join(path_to_save, f'true_observation')
    # if not os.path.exists(path_to_save_true_observation):
    #     os.makedirs(path_to_save_true_observation)

    # # Save the true observation data
    # np.save(os.path.join(path_to_save_true_observation, f'observation.npy'), x)
    # np.save(os.path.join(path_to_save_true_observation, f'true_parameters.npy'), theta)
    # np.save(os.path.join(path_to_save_true_observation, f'reference_posterior_samples.npy'), reference_posterior)
