
# Adaptive Adversarian Training to Improve Adversarian Robustness of DNNs for Medical Image Segmentation and Detection

To install the basic version of multi-task unet, and download the Cephalometric dataset, please refer to https://github.com/qsyao/attack_landmark_detection.

train.py trains the multi-task unet with original settings in the paper, Miss the Point: Targeted Adversarial Attack on Multiple Landmark Detection;

train_with_dice_loss.py trains the model with dice loss;

train_PGD_dice.py trains the model with vanilla adversarial training;

train_AMAT_2zscore.py trains the model with AMAT method in 