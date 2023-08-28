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
