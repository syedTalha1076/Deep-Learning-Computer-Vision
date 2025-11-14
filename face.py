import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dropout,Flatten,Conv2D,MaxPooling2D,Dense,BatchNormalization

train_dataset = keras.utils.image_dataset_from_directory(
    directory ="D:\\Artificial_Intellgence\\Deep_Learning\\Projects_Deep_Learning\\CNN_Projects\\FacialExpression\\train",
    labels = 'inferred',
    label_mode='int',
    batch_size=32,
    image_size=(96,96)
)

validation_dataset = keras.utils.image_dataset_from_directory(
    directory = "D:\\Artificial_Intellgence\\Deep_Learning\\Projects_Deep_Learning\\CNN_Projects\\FacialExpression\\test",
    labels = "inferred",
    label_mode="int",
    batch_size = 32,
    image_size = (96,96)
)

def process(image,label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.1)
    image = tf.cast(image / 255., tf.float32)
    return image,label

train_dataset = train_dataset.map(process)
validation_dataset = validation_dataset.map(process)


model = Sequential()

model.add(Conv2D(32,kernel_size=(3,3),padding='valid',activation='relu',input_shape=(96,96,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Conv2D(64,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Conv2D(128,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(64,activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(7,activation='softmax'))

print(model.summary())

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

history = model.fit(train_dataset,epochs = 1,validation_data=validation_dataset)
print(history)

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'],color='red',label='train')
plt.plot(history.history['val_accuracy'],color='blue',label='validation')
plt.legend()
plt.show()

# similarly for loss

plt.plot(history.history['loss'],color='red',label='train')
plt.plot(history.history['val_loss'],color='blue',label='validation')
plt.legend()
plt.show()

import cv2

test_img = cv2.imread('FacialExpression\\test\\test\\angry\\PublicTest_5964952.jpg')
plt.imshow(test_img)
plt.show()

print("Test image: ",test_img.shape) 

# Large image so resize
test_img = cv2.resize(test_img,(96,96))

test_input = test_img.reshape(1,96,96,3) # 1 --> Represent Batch we have 1 image so 1 batch

print(model.predict(test_input))
