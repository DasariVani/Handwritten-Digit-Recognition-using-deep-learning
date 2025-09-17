# Handwritten-Digit-Recognition-using-deep-learning

Handwritten Digit Recognition Using CNN (MNIST Dataset)

This project implements a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset. It demonstrates the use of deep learning for computer vision tasks and provides a clean, beginner-friendly example of image classification.
Features
Uses TensorFlow and Keras for building a CNN model.

Automatically loads and preprocesses the MNIST dataset.

Achieves high accuracy (~99%) on handwritten digit classification.

Visualizes training and validation accuracy and loss.

Saves the trained model for future use.
Technologies Used
Python 3.x
TensorFlow 2.x
Matplotlib (for visualization)

Dataset
The MNIST dataset contains 70,000 grayscale images of handwritten digits from 0 to 9.
60,000 images for training
10,000 images for testing
The dataset is automatically downloaded using TensorFlow’s datasets API.


How to Use
Clone the repository or download the code files.
Install dependencies:

bash
pip install tensorflow matplotlib
Run the script:

bash
python digit_recognition.py
The training process will begin and output progress on the console.

Training and validation accuracy/loss graphs will appear once training completes.

The trained model will be saved as mnist_cnn_model.h5 in the project folder.

Code Explanation
Data Loading & Normalization: MNIST data loaded and pixel values normalized to 0-1.
Model Architecture: Two convolutional layers followed by max pooling, flattening, dense layers, and a softmax output.
Training: Model trained over 5 epochs with validation after each.
Evaluation: Prints test accuracy and detailed classification report.
Visualization: Plots for accuracy and loss over epochs using matplotlib.

Future Improvements
Experiment with more epochs and hyperparameter tuning.
Try different model architectures like ResNet or EfficientNet.
Deploy the trained model using Flask/Django for web apps.
Add functionality to predict digits from user-uploaded images.
