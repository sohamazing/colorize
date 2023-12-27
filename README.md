# Image Colorization CNN
*Gray-scale to Color Conversion*  
*Author: Soham Mukherjee*  
*Date: December 2023*

## Overview
This project aims to colorize grayscale images using Convolutional Neural Networks (CNNs) and the YUV color space. It includes functions for YUV to RGB conversion, image display, data preprocessing, and model evaluation. The CNN architecture is designed for colorization, and the FLOPS and MACCs are estimated for model analysis.

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
```bash
pip install numpy tensorflow matplotlib requests scipy


## Usage
To use this project, follow these steps:
1. Download images (DIV2K dataset).
2. Convert images to YUV420p using FFmpeg.
3. Create patches from images.
4. Create NPZ file to store YUV channels for each image as numpy arrays.
5. Mount Google Drive in your notebook to access NPZ data.
6. Load YUV data from NPZ files using Google Drive paths.
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
```python
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
```python
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
