Repo for the A&A paper on Odisseo+FlowMatching (work in progress)
In order to reproduce the paper figure the Odisseo package (https://github.com/vepe99/Odisseo) and the sbi-sim package (odisseo_branch) need to be installed.

Figure file:

- Fig. 2-3 Are in `figure_2_3.ipynb`.
- Fig. 4 is in `figure_4.ipynb`.
- Fig. 7-8-9 requires :
  - to generate the training set using `generate_trainingset.py`
  - to generate the dataset using `create_dataset_AllParameters_varyingposition_uniform_TSIT5_sbisim.py`
  - to get the normalization for parameters \theta and the observation x
  - to get the posterior samples (as .npz file) you need to run, in the folder sbi-sim, the following comand to train and test
    `python3 --config=configs/experiments/sbi_variants/fm-ot/odisseo_CrossAttentionSetTransfomer_onehead_3layers_128_OT_AllParameters_Positions_uniformprior_2e5_TSIT5_error.yaml --name=OdisseoTraining_and_Testing --dryrun`
    this generate a folder named `OdisseoTraining_and_Testing`, if you need just to test after training add to the previous command "training.active=False" at the end.
    If you need to modify the batch_size to adapt to your harware, modify the file `odisseo_CrossAttentionSetTransfomer_onehead_3layers_128_OT_AllParameters_Positions_uniformprior_2e5_TSIT5_error.yaml`
    On a H200 the generation of the training set took ~ 7.5 GPU hours. On a A100 the training and testing took ~ 8 GPU hours.
  - run the `./PPC/test_results.ipynb`
- Fig. 10-11 in `./PPC/PPC.ipynb` (it requires you to run and store the forward modelled posterior samples).
