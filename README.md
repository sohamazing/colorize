# Image Colorization CNN
*Gray-scale to Color Conversion*  
*Author: Soham Mukherjee*  
*Date: December 2023*

## Overview
This project aims to colorize grayscale images using Convolutional Neural Networks (CNNs) in the YUV color space. It includes functions for training and inference on multi-output-channel CNNs, as well as utilities for YUV to RGB conversion and display, and neural model complexity analysis based on FLOPs and MACCs. 

The colab file included in this project includes all the details.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Model](#model)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

## Installation
Ensure you have the required libraries installed. You can install them using the following command:
pip install numpy tensorflow matplotlib requests scipy


## Usage
To use this project, follow these steps:

### Data Generation:
1. Download images (DIV2K dataset). Images 001-800 are used as the training set and the images 801-900 are used as the testing set.
2. Convert the images in each set to YUV420p using FFmpeg.
3. Get random p x p patches from images in each set. Each patch comprises a p x p Y channel and two corresponding U, V channels at half resolution p/2 x p/2. The Y patches are written out to a Y file, while the U, V channels are output interleaved as 2 channels to a separate file.
4. Randomly sort the patches from the Y and UV patch files.
5. Create numpy NPZ array files to store the sorted Y and UV patches for each set for convenience. Note that each NPZ file contains one multi-dimensional Y array of size (#patches x p x p), and one corresponding multi-dimensional UV array of size (#patches x p/2 x p/2 x 2). There will be two NPZ files produced one for the training set and another for the testing set.

### Training and testing (see colab code):

6. Load YUV data NPZ files for training and testing using Google Drive paths.
7. Split the training set into training and validation sets.
8. Normalize and reshape the data using the `preprocess` function.
9. Generate a CNN model with the `generate_cnn` function.
10. Train the CNN model with the training and validation data.
11. Save the trained model to Google Drive.
12. Test the trained model on the test dataset.

## Data
The project utilizes YUV data stored in NPZ files. The training and testing data paths are defined as follows:
- Training Data: `/content/drive/MyDrive/train64.npz`
- Testing Data: `/content/drive/MyDrive/valid64.npz`

For each forward pass of the CNN:
- Model inputs are the Y channels.
- Model outputs (labels) are the U and V channels.
- Together we have the predicted color YUV image.

## Model
The CNN model is designed for image colorization. It includes convolutional layers with varying filter sizes and strides. The architecture is defined in the `generate_cnn` function.

## Training
To train the model, execute the following steps directly in the Colab notebook:
1. Load training and testing data.
2. Split the training set into training and validation sets.
3. Normalize and reshape the data.
4. Generate a CNN model using `generate_cnn`.
5. Display the model summary and estimate FLOPS/MACCs.
6. Train the model using the `fit` method.
7. Save the trained model to Google Drive.

Example training code:
# Train the network
my_cnn.fit(
    x=y_train_norm,
    y=uv_train_norm,
    epochs=25,
    batch_size=1024,
    shuffle=True,
    validation_data=(y_valid_norm, uv_valid_norm),
)


## Evaluation
The evaluation of the trained model can be performed using a custom testing function on the test set. The function `test_my_cnn` preprocesses the test data, generates UV colors using the trained model, and displays the results using the `display3` function directly in the Colab notebook.

Example evaluation code:

# Load the trained model
model_out = '/content/drive/MyDrive/Colab Notebooks/colorizeA.keras'
my_cnn = tf.keras.models.load_model(model_out)

# Test the model on the test set
test_my_cnn(my_cnn, y_test, uv_test)


## Results
Provide details and visualizations of the results obtained from training and evaluation.

## Acknowledgments
- The project makes use of TensorFlow, NumPy, FFmpeg and other open-source libraries.
- The project uses the public DIV2K Dataset for images.
