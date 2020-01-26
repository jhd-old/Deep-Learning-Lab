# One dimensional convolutions
array = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
kernel = np.array([1, -1, 0])
conv = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# Output array
for ii in range(8):
    conv[ii] = (kernel * array[ii:ii+3]).sum()
# Print conv
print(conv)

# Image convolutions
kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
result = np.zeros(im.shape)
# Output array
for ii in range(im.shape[0] - 3):
    for jj in range(im.shape[1] - 3):
        result[ii, jj] = (im[ii:ii+3, jj:jj+3] * kernel).sum()
# Print result
print(result)

# Defining image convolution kernels
# kernel for vertical lines
kernel = np.array([[-1, 1, -1], 
                   [-1, 1, -1],
                   [-1, 1, -1]])
# kernel for horizontal lines
kernel = np.array([[-1,- 1, -1], 
                   [1, 1, 1],
                   [-1, -1, -1]])
# light spot surrounded by dark pixels
kernel = np.array([[-1, -1, -1], 
                   [-1, 1, -1],
                   [-1, -1, -1]])
# dark spot surrounded by bright pixels
kernel = np.array([[1, 1, 1], 
                   [1, -1, 1],
                   [1, 1, 1]])

# Implementing image convolutions in Keras
# Convolutional network for image classification
# Import the necessary components from Keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
# Initialize the model object
model = Sequential()
# Add a convolutional layer
model.add(Conv2D(10, kernel_size=3, activation='relu', 
               input_shape=[img_rows, img_cols, 1]))
# Flatten the output of the convolutional layer
model.add(Flatten())
# Add an output layer for the 3 categories
model.add(Dense(3, activation='softmax'))
# Compile the model 
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
# Fit the model on a training set
model.fit(train_data, train_labels, 
          validation_split=0.2, 
          epochs=3, batch_size=10)
                

# Evaluating a CNN with test data
# Evaluate the model on separate test data
model.evaluate(test_data, test_labels, batch_size=10)

# Tweaking your convolutions
# Add padding to a CNN
# Initialize the model
model = Sequential()
# Add the convolutional layer
model.add(Conv2D(10, kernel_size=3, activation='relu', 
                 input_shape=(img_rows, img_cols, 1), 
                 padding='same'))
# Feed into output layer
model.add(Flatten())
model.add(Dense(3, activation='softmax'))

# Add strides to a convolutional network
# Initialize the model
model = Sequential()
# Add a convolutional layer
model.add(Conv2D(10, kernel_size=3, activation='relu', 
                 input_shape=(img_rows, img_cols, 1), 
                 strides=2))
# Feed into output layer
model.add(Flatten())
model.add(Dense(3, activation='softmax'))

# size of output O=(I−K+2P)/S+1
