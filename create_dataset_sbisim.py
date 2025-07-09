import os
import jax.numpy as np
import pandas as pd 
from tqdm import tqdm 

def convert_to_csv(path_stored, path_to_save, num_simulations=1000):
    print('start converting all the npz files to csv')
    all_data_path = [os.path.join(path_stored, f) for f in sorted(os.listdir(path_stored))[:num_simulations] if f.endswith('.npz')]
    refence_path = [os.path.join(path_stored, f) for f in sorted(os.listdir(path_stored))[-1] if f.endswith('.npz')]
    x = []
    theta = []
    observation = []
    true_theta = []
    reference_posterior = []
    for file_path in tqdm(all_data_path):
        data = np.load(file_path)
        x.append(data['x'].flatten())
        theta.append(data['theta'][[0, 1, 3, 5]]) # we keep only the total integration time, the mass of Plummer, NFW and MN
        # Save as CSV with a single row (common for ML observations)
        df_x = pd.DataFrame([x])
        df_x.to_csv(os.path.join(path_to_save, f'x_{num_simulations}.csv'), index=False, header=False)
        df_theta = pd.DataFrame([theta])
        df_theta.to_csv(os.path.join(path_to_save, f'theta_{num_simulations}.csv'), index=False, header=False)
    for file_path in refence_path:
        observation.append(data['x'].flatten())
        true_theta.append(data['theta'])
        reference_posterior.append([])
        df_observation = pd.DataFrame([observation])
        df_observation.to_csv(os.path.join(path_to_save, f'observation.csv'), index=False, header=False)
        df_true_theta = pd.DataFrame([true_theta])
        df_true_theta.to_csv(os.path.join(path_to_save, f'true_parameters.csv'), index=False, header=False)
        df_reference_posterior = pd.DataFrame([reference_posterior])
        df_reference_posterior.to_csv(os.path.join(path_to_save, f'reference_posterior_samples.csv'), index=False, header=False)
    print(f'done converting all the npz files to csv in the folder {path_to_save}')

if __name__ == "__main__":
    path_stored = '/export/data/vgiusepp/odisseo_data/data_fix_position/'
    path_to_save = '/export/data/vgiusepp/odisseo_data/data_fix_position/sbi_sim'
    convert_to_csv(path_stored, path_to_save, num_simulations=1000)

