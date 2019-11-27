# Melanoma Detection Through Transfer Learning

## Introduction

  The aim of this project is to take pretrained state-of-the-art models, and utilize them for the task of 
	detecting Melanoma through Dermoscopic images. This is a process known as transfer learning, 
	and it has been used for a wide range of applications. Melanoma is a type of skin cancer that affects a large
	portion of the world's population as compared to other forms of the cancer. 
	Dermoscopic images are high resolution images taken through a Dermoscope, a specialized medical tool.

	The purpose of this project is to provide a baseline and a starting point for those who might want to experiment
	with their own ideas or try different models or techniques. The included presentation goes over some of the best
	results that were obtained by others on this task, and links to their various approaches to the problem.

## Dataset

  The **International Skin Imaging Collaboration** launched a competition on March 2019 where one of the tasks
	was to classify a set of dermoscopic images as belonging to one of several classes of skin cancer.
	The dataset contains **24,000** dermoscopic images divided meant for several tasks.
	This project utilizes that dataset, but it only goes as far as classifying the images as either
	belonging to the benign class (meaning that the lesion has no indications of Melanoma) or the malignant class
	(meaning that there is a high chance that the person suffers from a Melanoma). 

	The dataset can be downloaded through the following link in accordance with the terms & conditions specified by the collaboration:
	https://isic-archive.com/api/v1/

## Models

  The models used in this experiment are the the **ResNet-50** and the **ResNext-50**. In order
	to try and improve the performance of the models, Self Attention was also utilized.
	Each of the models was first tried without the use of Self Attention, and then with several
	attetion layers placed at certain locations in the pipeline of each one.

## Project Files & Directories

  Below is a description of each directory and its role in the project:

	1. **documents**: contains a powerpoint presentation that briefly goes over the project and the results obtained.
										Also contains a video narration that goes over the presentation slides and the original proposal.
	2. **nets**: contains all the models along with the hyperparameters used in training.
	3. **outputs**: contains the outputs obtained from training the models over a 100 epochs.
	4. **slice_n_dice**: contains all the python and bash scripts used to obtain and format the images, segmentation masks and metadata.
	5. **samples**: contains the downloaded images along with the metadata, split into a train/validation/test hierarchy.

	Files:

	1. **main.py**: the main script where the training and evaluation take place. The type of network is chosen and is trained
									for the specified number of epochs.
	2. **project_code_link**: a link to the repository where this project is hosted.
