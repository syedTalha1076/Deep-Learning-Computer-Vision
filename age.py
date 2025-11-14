
# ! Finding total number of classes
# import os

# # path to your training folder
# train_dir = r"D:\\Artificial_Intellgence\\Deep_Learning\\Projects_Deep_Learning\\CNN_Projects\\Images\\train"

# # get all subfolders (each subfolder = one class)
# class_names = [folder for folder in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, folder))]
# num_classes = len(class_names)

# print("Class names:", class_names)
# print("Number of classes:", num_classes)
# ==================================================
#* Project

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten,MaxPooling2D,BatchNormalization,Dropout,Conv2D

train_dataset = keras.utils.image_dataset_from_directory(
    directory="D:\\Artificial_Intellgence\\Deep_Learning\\Projects_Deep_Learning\\CNN_Projects\\Images\\train",
    labels = "inferred",
    label_mode="int",
    batch_size = 32,
    image_size=(256,256)
)

test_dataset=keras.utils.image_dataset_from_directory(
    directory = "D:\\Artificial_Intellgence\\Deep_Learning\\Projects_Deep_Learning\\CNN_Projects\\Images\\test",
    labels = "inferred",
    label_mode='int',
    batch_size=32,
    image_size=(256,256)
)

# print("working correctly")

# def process(image,label):
#     image = tf.cast(image/255.,tf.float32)
#     return image,label

def process(image,label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.1)
    image = tf.cast(image / 255., tf.float32)
    return image,label


train_dataset = train_dataset.map(process)
test_dataset = test_dataset.map(process)

# making layers

model = Sequential()

model.add(Conv2D(32,kernel_size=(3,3),padding='valid',activation='relu',input_shape=(256,256,3)))
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

model.add(Dense(5,activation='softmax'))

print(model.summary())

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

history = model.fit(train_dataset,epochs = 10,validation_data=test_dataset)
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

