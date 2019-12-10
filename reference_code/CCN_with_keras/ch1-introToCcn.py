# Images as data: visualizations
# Import matplotlib
import matplotlib.pyplot as plt
# Load the image
data = plt.imread('bricks.png')
# Display the image
plt.imshow(data)
plt.show()

#Images as data: changing images
# Set the red channel in this part of the image to 1
data[:10,:10,0] = 1
# Set the green channel in this part of the image to 0
data[:10,:10,1] = 0
# Set the blue channel in this part of the image to 0
data[:10,:10,2] = 0
# Visualize the result
plt.imshow(data)
plt.show()

# Using one-hot encoding to represent images
import numpy as np
# The number of image categories
n_categories = 3
# The unique values of categories in the data
categories = np.array(["shirt", "dress", "shoe"])
# Initialize ohe_labels as all zeros
ohe_labels = np.zeros((len(labels), n_categories))
# Loop over the labels
for ii in range(len(labels)):
    # Find the location of this label in the categories variable
    jj = np.where(categories == labels[ii])
    # Set the corresponding zero to one
    ohe_labels[ii, jj] = 1

# Evaluating a classifier
# Calculate the number of correct predictions
number_correct = (test_labels * predictions).sum() #one hot encoded arrays
print(number_correct)
# Calculate the proportion of correct predictions
proportion_correct = number_correct/len(predictions)
print(proportion_correct)

# Build a neural network
# Imports components from Keras
from keras.models import Sequential
from keras.layers import Dense
# Initializes a sequential model
model = Sequential()
# First layer
model.add(Dense(10, activation="relu", input_shape=(784,)))
# Second layer
model.add(Dense(10, activation="relu", input_shape=(784,)))
# Output layer
model.add(Dense(3, activation="softmax", input_shape=(784,)))

# Compile a neural network
# Compile the model
model.compile(optimizer="adam", 
           loss='categorical_crossentropy', 
           metrics=['accuracy'])

# Fitting a neural network model to clothing data
# Reshape the data to two-dimensional array
train_data = train_data.reshape(50, 784)
# Fit the model
model.fit(train_data, train_labels, validation_split=0.2, epochs=3)

# Cross-validation for neural network evaluationa
# Reshape test data
test_data = test_data.reshape(10, 784)
# Evaluate the model
model.evaluate(test_data, test_labels)