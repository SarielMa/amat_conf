## Adaptive Adversarian Training to Improve Adversarian Robustness of DNNs for Medical Image Segmentation and Detection

To install the basic version of nnUnet and download the datasets, Heart, Hippocampus and Prostate MRI, please refer to https://github.com/MIC-DKFZ/nnUNet 

Remember to modify the path in the nnUnet/nnunet/paths.py

the training and evaluation modules are in amat_conf/nnUNet/nnunet/run/:

run_training.py to train the original nnunet model;

run_AMAT_training.py to train the nnunet model with AMAT defense method;

run_PGD_training.py to train the nnunet model with vanilla adversarial training method;




