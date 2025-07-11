import os
import jax.numpy as np
import pandas as pd 
from tqdm import tqdm 

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
        theta.append(data['theta'][[0, 1, 3, 5]]) # we keep only the total integration time, the mass of Plummer, NFW and MN
        observation.append(data['x'])
        true_theta.append(data['theta'])
        reference_posterior.append([])

    np.save(os.path.join(path_to_save, f'x_{num_simulations}.npy'), x)
    np.save(os.path.join(path_to_save, f'theta_{num_simulations}.npy'), theta)
    np.save(os.path.join(path_to_save, f'observation.npy'), observation)
    np.save(os.path.join(path_to_save, f'true_parameters.npy'), true_theta)
    np.save(os.path.join(path_to_save, f'reference_posterior_samples.npy'), reference_posterior)
    
    print(f'done converting all the npz files to a single npz file in the folder {path_to_save}')

def num_observation_npy(path_stored, path_to_save, simulation_observation_index=[1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009]):
    print('start converting num_observation files to npy')
    all_data_path = [os.path.join(path_stored, f) for f in sorted(os.listdir(path_stored))[1000:1010] if f.endswith('.npz') and 'file' in f]
    x = []
    theta = []

    i=1
    for file_path in tqdm(all_data_path):
        path_to_save_num_observation = os.path.join(path_to_save, f'num_observation_{i}')
        if not os.path.exists(path_to_save_num_observation):
            os.makedirs(path_to_save_num_observation)
        data = np.load(file_path)
        x.append(data['x']) #histogram
        theta.append(data['theta'][[0, 1, 3, 5]]) # we keep only the total integration time, the mass of Plummer, NFW and MN

        np.save(os.path.join(path_to_save_num_observation, f'observation.npy'), x)
        np.save(os.path.join(path_to_save_num_observation, f'true_parameters.npy'), theta)
        i += 1
    print(f'done converting all the num_observation files in the folder {path_to_save}')


if __name__ == "__main__":
    path_stored = '/export/data/vgiusepp/odisseo_data/data_fix_position/preprocess/'
    path_to_save = './data/sbi-benchmarks/odisseo/'
    # convert_to_csv(path_stored, path_to_save, num_simulations=1000)
    # convert_to_single_npy(path_stored, path_to_save, num_simulations=1_000)
    num_observation_npy(path_stored, path_to_save, simulation_observation_index=range(1000, 1010))


