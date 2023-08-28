# Brain-Region-Segmentation


This repository contains code for segmenting brain regions using a U-Net AI model, a popular architecture for image segmentation tasks.

## Table of Contents
- [Introduction](#introduction)
- [Data Preparation](#data-preparation)
- [Preprocessing](#preprocessing)
- [U-Net Model](#u-net-model)
- [Training](#training)
- [Evaluation](#evaluation)
- [Known Issue](#known-issue)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Accurate segmentation of brain regions is essential for various medical applications. This project focuses on using a U-Net AI model to automate this segmentation process, improving efficiency and reliability.

## Data Preparation
To begin, we organize our dataset into two folders: 'brain masks' containing masks of brain regions and 'CTA images' containing the corresponding CTA (Computed Tomography Angiography) images.

## Preprocessing
Each brain mask and CTA image undergoes preprocessing before being used for training. The process involves generating Maximum Intensity Projections (MIPs) along the z-axis to capture important features, followed by normalization. The results are saved as PNG files in dedicated folders.

## U-Net Model
The U-Net architecture is employed due to its effectiveness in image segmentation tasks. It consists of an encoder path to capture context and a decoder path for precise localization.

## Training
We generate a custom dataset and convert it into a tensor dataset suitable for training. Hyperparameters such as learning rate, loss function (e.g., dice coefficient loss), and optimizer (e.g., Adam) are defined. The U-Net model is then trained using the prepared dataset. 

Evaluation
The model's MSE value becomes constant after a certain number of iterations and the model doesnot generate the desired output
After training, the model's performance is evaluated using various metrics, including the Dice score. However, am currently experiencing an issue where the Dice score evaluation is consistently zero. I am actively investigating this issue and working on a solution to improve the model's performance.

this was the model's training result 
Epoch [1/30] Batch [1/1] Loss: 1.2363
Epoch [2/30] Batch [1/1] Loss: 1.2279
Epoch [3/30] Batch [1/1] Loss: 1.2038
Epoch [4/30] Batch [1/1] Loss: 1.1205
Epoch [5/30] Batch [1/1] Loss: 0.9204
Epoch [6/30] Batch [1/1] Loss: 0.7273
Epoch [7/30] Batch [1/1] Loss: 0.6941
Epoch [8/30] Batch [1/1] Loss: 0.6932
Epoch [9/30] Batch [1/1] Loss: 0.6931
Epoch [10/30] Batch [1/1] Loss: 0.6931
Epoch [11/30] Batch [1/1] Loss: 0.6931
Epoch [12/30] Batch [1/1] Loss: 0.6931
Epoch [13/30] Batch [1/1] Loss: 0.6931
Epoch [14/30] Batch [1/1] Loss: 0.6931
Epoch [15/30] Batch [1/1] Loss: 0.6931
Epoch [16/30] Batch [1/1] Loss: 0.6931
Epoch [17/30] Batch [1/1] Loss: 0.6931
Epoch [18/30] Batch [1/1] Loss: 0.6931
Epoch [19/30] Batch [1/1] Loss: 0.6931
Epoch [20/30] Batch [1/1] Loss: 0.6931
Epoch [21/30] Batch [1/1] Loss: 0.6931
Epoch [22/30] Batch [1/1] Loss: 0.6931
Epoch [23/30] Batch [1/1] Loss: 0.6931
Epoch [24/30] Batch [1/1] Loss: 0.6931
Epoch [25/30] Batch [1/1] Loss: 0.6931
Epoch [26/30] Batch [1/1] Loss: 0.6931
Epoch [27/30] Batch [1/1] Loss: 0.6931
Epoch [28/30] Batch [1/1] Loss: 0.6931
Epoch [29/30] Batch [1/1] Loss: 0.6931
Epoch [30/30] Batch [1/1] Loss: 0.6931
