Repo for the A&A paper on Odisseo+FlowMatching (work in progress)
In order to reproduce the paper figure the Odisseo package (https://github.com/vepe99/Odisseo) and the sbi-sim package (odisseo_branch) need to be installed.

Figure file:

- Run `figure_1_2.ipynb` for Fig. 1-2.
- Run `figure_3.ipynb` Fig. 3.
- Fig. 6-7-8-9 requires to either download the dataset from Zenodo (`https://zenodo.org/uploads/17711491`) or to generate the training set by following steps from 1-3. Afterwards run 4-5:
  1. to generate the training set using `generate_trainingset.py`
  2. to generate the dataset using `create_dataset_AllParameters_varyingposition_uniform_TSIT5_sbisim.py`
  3. to get the normalization for parameters \theta and the observation x
  4. to get the posterior samples (as .npz file) you need to run, in the folder sbi-sim, the following comand to train and test
     `python3 --config=configs/experiments/sbi_variants/fm-ot/odisseo_CrossAttentionSetTransfomer_onehead_3layers_128_OT_AllParameters_Positions_uniformprior_2e5_TSIT5_error.yaml --name=OdisseoTraining_and_Testing --dryrun`
     this generate a folder named `OdisseoTraining_and_Testing`, if you need just to test after training add to the previous command "training.active=False" at the end.
     If you need to modify the batch_size to adapt to your harware, modify the file `odisseo_CrossAttentionSetTransfomer_onehead_3layers_128_OT_AllParameters_Positions_uniformprior_2e5_TSIT5_error.yaml`
     On a H200 the generation of the training set took ~ 7.5 GPU hours. On a A100 the training and testing took ~ 8 GPU hours.
  5. run the `./PPC/test_results.ipynb`
- Fig. 10-11 in `./PPC/PPC.ipynb` (it requires you to run and store the forward modelled posterior samples).
