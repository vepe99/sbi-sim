import os
import jax.numpy as np
import pandas as pd 
from tqdm import tqdm 
import jax
import jax.numpy as jnp 

from astropy import units as u
from odisseo.units import CodeUnits

# def convert_to_csv(path_stored, path_to_save, num_simulations=1000):
#     print('start converting all the npz files to csv')
#     all_data_path = [os.path.join(path_stored, f) for f in sorted(os.listdir(path_stored))[:num_simulations] if f.endswith('.npz')]
#     refence_path = [os.path.join(path_stored, f) for f in sorted(os.listdir(path_stored))[-1] if f.endswith('.npz')]
#     x = []
#     theta = []
#     observation = []
#     true_theta = []
#     reference_posterior = []
#     for file_path in tqdm(all_data_path):
#         data = np.load(file_path)
#         x.append(data['x'].flatten())
#         theta.append(data['theta'][[0, 1, 3, 5]]) # we keep only the total integration time, the mass of Plummer, NFW and MN
#         # Save as CSV with a single row (common for ML observations)
#     df_x = pd.DataFrame(x)
#     df_x.to_csv(os.path.join(path_to_save, f'x_{num_simulations}.csv'), index=False, header=False)
#     df_theta = pd.DataFrame(theta)
#     df_theta.to_csv(os.path.join(path_to_save, f'theta_{num_simulations}.csv'), index=False, header=False)
#     for file_path in refence_path:
#         observation.append(data['x'].flatten())
#         true_theta.append(data['theta'])
#         reference_posterior.append([])
#     df_observation = pd.DataFrame(observation)
#     df_observation.to_csv(os.path.join(path_to_save, f'observation.csv'), index=False, header=False)
#     df_true_theta = pd.DataFrame(true_theta)
#     df_true_theta.to_csv(os.path.join(path_to_save, f'true_parameters.csv'), index=False, header=False)
#     df_reference_posterior = pd.DataFrame(reference_posterior)
#     df_reference_posterior.to_csv(os.path.join(path_to_save, f'reference_posterior_samples.csv'), index=False, header=False)
#     print(f'done converting all the npz files to csv in the folder {path_to_save}')


def to_inference_parameters(theta, ):
        """
        convert from simulation parameters to inference parameters
        """
        code_length = 10.0 * u.kpc
        code_mass = 1e4 * u.Msun
        code_time = 3 * u.Gyr
        code_units = CodeUnits(code_length, code_mass, G=1, unit_time = code_time )  

        theta[0] = theta[0] # t_end is already in Gyr
        theta[1] = np.log10(theta[1]).item() # Plummer mass is already in Msun
        theta[2] = theta[2] * code_units.code_length.to(u.kpc) # Plummer a
        theta[3] = np.log10(theta[3] * code_units.code_mass.to(u.Msun)).item()  # NFW Mvir
        theta[4] = theta[4] * code_units.code_length.to(u.kpc)  # NFW r_s
        theta[5] = np.log10(theta[5] * code_units.code_mass.to(u.Msun)).item() # MN M
        theta[6] = theta[6] * code_units.code_length.to(u.kpc) # MN a
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
        x.append(data['x'][:1000]) #histogram
        theta.append(to_inference_parameters(data['theta'])[[0, 1, 3, 5]]) # we keep only the total integration time, the mass of Plummer, NFW and MN
        observation.append(data['x'][:1000])
        true_theta.append(to_inference_parameters(data['theta'])[[0, 1, 3, 5]])
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
        theta.append(to_inference_parameters(data['theta'])[[0, 1, 3, 5]]) # we keep only the total integration time, the mass of Plummer, NFW and MN

        np.save(os.path.join(path_to_save_num_observation, f'observation.npy'), x)
        np.save(os.path.join(path_to_save_num_observation, f'true_parameters.npy'), theta)
        i += 1
    print('x shape:', np.array(x).shape)
    print('theta shape:', np.array(theta).shape)
    print(f'done converting all the num_observation files in the folder {path_to_save}')


if __name__ == "__main__":
    path_stored = '/export/data/vgiusepp/odisseo_data/data_fix_position/'
    path_to_save = './data/sbi-benchmarks/odisseo/'
    # convert_to_csv(path_stored, path_to_save, num_simulations=1000)
    convert_to_single_npy(path_stored, path_to_save, num_simulations=10_000)
    num_observation_npy(path_stored, path_to_save, simulation_observation_index=10_000)



