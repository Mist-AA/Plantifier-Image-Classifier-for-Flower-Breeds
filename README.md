# Plantifier - The Image Classifier to identify Flower Breeds

## Overview:
This project aims to create a robust image classifier for flowers using deep learning with PyTorch. The implementation is divided into two parts: the development notebook and the command line application.

## Part 1 - Development Notebook:
### Package Imports:
In the first cell of the notebook, all the necessary packages and modules are imported. This includes PyTorch for building and training the neural network, torchvision for handling image data, and matplotlib for visualizations.

### Training Data Augmentation:
The training data is augmented using torchvision transforms, which include random scaling, rotations, mirroring, and cropping. This helps enhance the model's ability to generalize to different variations in the input data.

### Data Normalization and Loading:
The training, validation, and testing data are appropriately cropped and normalized to ensure consistent input to the model. Data for each set (train, validation, test) is loaded using torchvision's ImageFolder, and DataLoader is employed for efficient batch loading during training.

### Pretrained Network and Feedforward Classifier:
A pretrained network, such as VGG16, is loaded from torchvision.models, and its parameters are frozen. A new feedforward network is defined as a classifier, using the features extracted by the pretrained network as input.

### Training the Network:
The parameters of the feedforward classifier are trained while keeping the parameters of the feature network static. During training, the validation loss and accuracy are displayed to monitor the model's performance.

### Testing Accuracy and Model Saving:
The network's accuracy is measured on the test data. Once trained, the model is saved as a checkpoint, including associated hyperparameters and the class_to_idx dictionary.

### Loading Checkpoints, Image Processing, and Class Prediction:
Functions are implemented to load a checkpoint and rebuild the model. The process_image function converts a PIL image into an object suitable for the trained model. The predict function takes the path to an image and a checkpoint, returning the top K most probable classes for that image.

### Sanity Checking with Matplotlib:
A matplotlib figure is created to display an image along with its top 5 most probable classes, showing both class indices and actual flower names.

## Part 2 - Command Line Application:
### Training a Network with train.py:
The train.py script allows users to train a new network on a dataset of images. Basic usage is python train.py data_directory, and options include setting the directory to save checkpoints, choosing the architecture, setting hyperparameters, and using GPU for training.

### Training Validation Log:
As the network trains, the script prints out training loss, validation loss, and validation accuracy, providing insights into the model's progress.

### Model Architecture and Hyperparameters:
Users can choose from at least two different architectures available from torchvision.models. The script allows setting hyperparameters for learning rate, the number of hidden units, and training epochs, providing flexibility in model customization.

### Training with GPU:
The training script accommodates users who want to train the model on a GPU for faster processing.

### Predicting Classes with predict.py:
The predict.py script reads in an image and a checkpoint, then prints the most likely image class and its associated probability.

### Top K Classes and Displaying Class Names:
Users can customize the output by specifying the top K most likely classes along with associated probabilities. Additionally, the script allows loading a JSON file that maps the class values to other category names, providing more human-readable output.

### Predicting with GPU:
For efficient inference, the predict.py script allows users to use the GPU to calculate predictions.

## How to Use:
1. Clone the repository.
2. Navigate to the project directory.
3. For training: python train.py data_directory --options
4. For prediction: python predict.py /path/to/image checkpoint --options

### Source
This was the final project for the AI Programming with Python Nanodegree on Udacity as part of the initial phase of AWS AI & ML Scholarship Program in 2022.
