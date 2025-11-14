
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Conv2D, Flatten, AveragePooling2D
from keras import Sequential
from keras.datasets import mnist
import numpy as np
from tensorflow.keras.preprocessing import image
import cv2

# ---------------------- Step 1: Load MNIST ----------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize and reshape for CNN
x_train = x_train / 255.0
x_test = x_test / 255.0

# Pad images from 28x28 to 32x32
x_train = np.pad(x_train, ((0,0),(2,2),(2,2)), 'constant')
x_test = np.pad(x_test, ((0,0),(2,2),(2,2)), 'constant')

x_train = x_train.reshape(-1,32,32,1)
x_test = x_test.reshape(-1,32,32,1)

# ---------------------- Step 2: Build LeNet-5 ----------------------
model = Sequential()
model.add(Conv2D(6, kernel_size=(5,5), padding='valid', activation='tanh', input_shape=(32,32,1)))
model.add(AveragePooling2D(pool_size=(2,2), strides=2, padding='valid'))

model.add(Conv2D(16, kernel_size=(5,5), padding='valid', activation='tanh'))
model.add(AveragePooling2D(pool_size=(2,2), strides=2, padding='valid'))

model.add(Flatten())
model.add(Dense(120, activation='tanh'))
model.add(Dense(84, activation='tanh'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ---------------------- Step 3: Train the model ----------------------
model.fit(x_train, y_train, epochs=5, batch_size=128, validation_data=(x_test, y_test))

# ---------------------- Step 4: User input for prediction ----------------------
# print("\nYou can provide your own digit image for prediction.")
print("User Input is given\n")
user_img_path = "D:\\Artificial_Intellgence\\Deep_Learning\\CNN\\imageOf4.png"

# Load image
img = image.load_img(user_img_path, color_mode='grayscale', target_size=(28,28))
img_array = image.img_to_array(img)
img_array = img_array / 255.0

# Pad to 32x32
img_array = np.pad(img_array, ((2,2),(2,2),(0,0)), 'constant')
img_array = img_array.reshape(1,32,32,1)

# Predict
pred = model.predict(img_array)
predicted_digit = np.argmax(pred)

print(f"\nPredicted digit: {predicted_digit}")
