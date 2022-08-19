"""1. Load our Keras Model and the MNIST Dataset
Download our previous model and Load it with load_model"""

#!gdown --id 1jW5aHd7_fAi3UrbT9MRTDbKyxjwfQ3WC

# We need to import our load_model function
from tensorflow.keras.models import load_model

model = load_model('mnist_simple_cnn_10_Epochs.h5')

# We can load the built in datasets from this function
from tensorflow.keras.datasets import mnist

# loads the MNIST training and test dataset
(x_train, y_train), (x_test, y_test)  = mnist.load_data()


"""2. Viewing Our Misclassifications
First let's get our Test Predictions"""

import numpy as np

# We reshape our test data
print(x_test.shape)
x_test = x_test.reshape(10000,28,28,1)
print(x_test.shape)

# Get the predictions for all 10K samples in our test data
print("Predicting classes for all 10,000 test images...")
pred = np.argmax(model.predict(x_test), axis=-1)
print("Completed.\n")


import cv2
import numpy as np

# Use numpy to create an array that stores a value of 1 when a misclassification occurs
result = np.absolute(y_test - pred)
misclassified_indices = np.nonzero(result > 0)

#  Display the indices of mislassifications
print(f"Indices of misclassifed data are: \n{misclassified_indices}")
print(len(misclassified_indices[0]))