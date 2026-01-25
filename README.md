# INTRODUCTION

Thank you so much for your interest in the prostate DWI quality inference! This model takes T2, ADC map, and high-b value DW MRI images and assesses them according to the principles of PI-QUALv2.1.

In order to keep inference lean in a low computational power/CPU-based setting, the inference is presented as a probabilitity that the DWI image belongs to the diagnostic or non-diagnostic class. In the lean version intended for public use, confidence intervals are NOT presented. Source code is provided for those interested in more robust testing of the model enclosed.

## A note on the environment

Note: The model presented here has been trained and tested using Python 3.9.13. We find that this version of Python best facilitates MONAI functionality. We recommend using this code in an environment with Python 3.9.13 installed. In GETTING STARTED, the authors walk the end user through setting up the ideal environment in which to run the model.

## A note on pre-processing of images

A small note on image pre-processing: The necessary pipeline to prepare T2, HiB, and ADC images for inference has been incorporated into the code presented here. 3D registration was not necessary due to the presence of rough 3D registration performed by our MRI machines themselves. Thus, we assume the images are in roughly similar 3D space prior to inference. Sub-optimal inference may be observed if this pre-condition is not met. The authors are aware that this software/3D registration is now a very common feature of most modern MRI machines. For those without this feature, 3D registration may be required prior to use of the enclosed model for optimal results.

## Files that are contained within this repository:

In this repository, you will find the following files:

### GUI Files for Use - "out of the box" files to use the model as currently trained

-- quality_inference_gui.py - This file contains the Python code for a basic GUI to import images for inference according to the most up-to-date version of the model. As of 					5/2025, the weights for the ADC arm of the model are contained within adc_model_epoch_99_2class.pth while the weights for the high-b value arm of the model are contained within hib_model_epoch_100_2class.py. 

-- hib_model_epoch_100_2class.pth - Trained weights for the hib version of the model

-- adc_model_epoch_99_2class.pth - Trained weights for the adc version of the model

### Training Pipeline  - Code used for training the current version of the model
-- quality_model_adc_may1_selfattention_2classes_evaluate.py - pipeline used for training and evaluation of the ADD map arm of the model

-- quality_model_hib_may1_selfattention_2classes_evaluate.py - pipeline used for training and evaluation of the high-b DW MRI arm of the model

For maximal transparency, NOTHING has been changed in the training pipeline from the code used to train or evaluate the model. The entire training pipeline is included. 

# GETTING STARTED











