# Galaxy-Morphologie-Classification-Deep-Learning
A deep learning solution for the Galaxy Zoo Challenge, using transfer learning with EfficientNetB0 to classify galaxy morphologies. This repository contains a complete TensorFlow/Keras implementation for multi-output classification of galaxy images into 37 morphological categories.

## How the Galaxy Zoo Classification Code Works

This code implements a deep learning solution for classifying galaxy images from the Galaxy Zoo challenge. The system uses transfer learning with a pre-trained EfficientNetB0 model to predict probabilities for 37 different morphological features of galaxies.

### Workflow Overview:

Data Preparation: Loads galaxy images and corresponding labels from CSV files
Data Augmentation: Applies random transformations to increase dataset diversity
Model Building: Uses EfficientNetB0 as feature extractor with custom classification head
Transfer Learning: First freezes base model weights, then unfreezes for fine-tuning
Training: Uses Adam optimizer with binary crossentropy loss for multi-label classification
Prediction: Generates probability predictions for all 37 output classes
Submission: Creates properly formatted CSV file for Kaggle submission

## Library Explanations:
os - Used for file path operations and directory navigation to access image files and manage file paths across different operating systems.

numpy as np - Essential for numerical computations, array operations, and handling image data as numerical arrays. Used for converting images to arrays and batch processing.

pandas as pd - Handles CSV files containing galaxy labels and IDs. Manages the structured data for training and creates the final submission file in the required format.

tensorflow as tf - Core deep learning framework that provides all neural network components, layers, optimizers, and training utilities. The foundation for building and training the model.

from tensorflow.keras import applications, optimizers -

applications: Provides pre-trained models like EfficientNetB0 for transfer learning

optimizers: Contains optimization algorithms (Adam) for model training

from tensorflow.keras.preprocessing.image import ImageDataGenerator - Creates data pipelines with real-time data augmentation. Generates batches of tensor image data with on-the-fly transformations to prevent overfitting.

from tensorflow.keras.models import Model - Allows creating custom model architectures by connecting layers. Essential for adding custom classification head to the pre-trained base.

from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D -

Dense: Fully connected layers for classification

Dropout: Regularization technique to prevent overfitting

GlobalAveragePooling2D: Reduces spatial dimensions before final classification

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau - Training monitoring utilities:

ModelCheckpoint: Saves best model during training

EarlyStopping: Prevents overtraining by stopping when validation loss stops improving

ReduceLROnPlateau: Dynamically adjusts learning rate for better convergence

## Why These Specific Libraries? :

TensorFlow/Keras was chosen for its comprehensive deep learning capabilities and excellent transfer learning support

EfficientNetB0 provides state-of-the-art performance with relatively low computational requirements

Pandas efficiently handles the large CSV files with 37 columns of probability labels

ImageDataGenerator solves memory issues by loading images in batches and provides crucial data augmentation

The callback system ensures optimal training and prevents common issues like overfitting

### The code implements a two-phase training approach: first training the classification head while keeping the base model frozen, then fine-tuning the entire network for optimal performance on the specific galaxy classification task.

