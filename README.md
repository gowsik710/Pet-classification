# Pet-classification

# CNN Image Classification with TensorFlow and Keras

## Project Overview
This project implements a Convolutional Neural Network (CNN) model to classify images into two classes using TensorFlow and Keras. The model is trained on an augmented dataset to improve generalization and reduce overfitting. The aim is to achieve a balance between accuracy and computational efficiency.

## Detailed Project Description

### 1. Importing Required Packages
- TensorFlow and Keras libraries for building and training the CNN model.
- ImageDataGenerator for real-time image augmentation to increase dataset diversity.

### 2. Data Augmentation and Preparation
- Used `ImageDataGenerator` to apply transformations such as:
  - Rescaling pixel values to [0,1].
  - Random shear transformations.
  - Random zoom.
  - Horizontal flipping.
- Loaded training and testing datasets from directories using `flow_from_directory`.
- Images are resized to 64x64 pixels.
- Batch size is set to 32.

### 3. Model Architecture
- **Convolutional Layers**: Extract spatial features with filters of size 5x5.
  - First Conv2D layer: 32 filters, ReLU activation, input shape (64,64,3).
  - Second Conv2D layer: 64 filters, ReLU activation.
- **MaxPooling Layers**: Reduce spatial dimensions to prevent overfitting and reduce computation.
- **Flatten Layer**: Converts 2D feature maps to 1D feature vector.
- **Dense Layers**: Fully connected layers for classification.
  - First Dense layer: 32 units, ReLU activation.
  - Dropout layer with 40% dropout rate to reduce overfitting.
  - Output Dense layer: 1 unit with sigmoid activation for binary classification.

### 4. Model Compilation and Training
- Loss function: Binary Crossentropy.
- Optimizer: Adam.
- Metrics: Accuracy.
- Model trained for 300 epochs with validation on the test set.
- Training progress is verbose for monitoring.

### 5. Model Evaluation
- After training, the model is evaluated on the test dataset.
- Observed results:
  - At 100 epochs: Validation loss ~1.0773, accuracy ~70%.
  - At 200 epochs: Validation loss ~1.6069, accuracy ~65%.
  - At 300 epochs: Validation loss ~4.1340, accuracy ~60%.
- Indicates overfitting after too many epochs, where accuracy decreases despite longer training.

## Key Takeaways
- **Data Augmentation** helps increase dataset variability, improving model generalization.
- **Dropout Layer** reduces overfitting by randomly dropping neurons during training.
- **Overfitting Warning**: After a certain number of epochs, the model starts overfitting, leading to decreased accuracy.
- Optimal training duration should be chosen based on validation metrics to avoid overfitting.

## Technologies and Tools Used
- Python
- TensorFlow 2.4.1
- Keras API
- Libraries: numpy, matplotlib (if needed for visualization)
