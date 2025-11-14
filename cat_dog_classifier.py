import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,BatchNormalization,Dropout

# Generators ==> Divide data into batches because Here we have multiple images
#                so it is difficult to do work at the same time with this lot of data
#                for which RAM becomes less so thats why we will use Generators concept

# we will use 2 Generators one for training and another for testing

train_dataset = keras.utils.image_dataset_from_directory(
    #! Notice in link pass 2 back slashes than it will work
    directory = "D:\\Artificial_Intellgence\\Deep_Learning\\CNN\\train",
    labels = "inferred",
    label_mode = "int",
    batch_size = 32,
    image_size = (256, 256)
)


validation_dataset =  keras.utils.image_dataset_from_directory(
    directory = 'D:\Artificial_Intellgence\\Deep_Learning\\CNN\\test', # provide train path
    labels = 'inferred',
    label_mode = 'int',  # this will assign cat 1 and Dog 0
    batch_size = 32,
    image_size = (256,256) # in our data all images has different size so we make it same

) 


# The images is store in numpy array having values between 0 to 255 
# Now we have to Normalize means that values becomes between 0 and 1

def process(image,label):
    image = tf.cast(image/255.,tf.float32)
    return image,label

train_dataset = train_dataset.map(process) # map function will extract each image and send to the above function
validation_dataset = validation_dataset.map(process)

# CNN Model
model = Sequential()
# 1st convalutional layer
model.add(Conv2D(32,kernel_size=(3,3),padding = 'valid',activation = 'relu',input_shape=(256,256,3)))
model.add(BatchNormalization()) # Reduce Overfitting 
# add pooling layer
model.add(MaxPooling2D(pool_size=(2,2),strides = 2,padding = 'valid'))

# 2nd Convalution layer
model.add(Conv2D(64,kernel_size=(3,3),padding = 'valid',activation = 'relu'))
model.add(BatchNormalization()) # Reduce Overfitting

model.add(MaxPooling2D(pool_size=(2,2),strides = 2,padding = 'valid'))

# 3rd Convalutional layer
model.add(Conv2D(128,kernel_size=(3,3),padding = 'valid',activation = 'relu'))
model.add(BatchNormalization()) # Reduce Overfitting
model.add(MaxPooling2D(pool_size=(2,2),strides = 2,padding = 'valid'))

# Flatten 
model.add(Flatten())

# Add 3 Fully Connected layer
model.add(Dense(128,activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(64,activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(1,activation = 'sigmoid'))

print(model.summary())

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# training will take too much time because we are working on CPU
 
# Check new image

import cv2

test_img = cv2.imread('D:\\Artificial_Intellgence\\Deep_Learning\\CNN\\testImg.jpg')
plt.imshow(test_img)
plt.show()

print("Test image: ",test_img.shape) 

# Large image so resize
test_img = cv2.resize(test_img,(256,256))

test_input = test_img.reshape(1,256,256,3) # 1 --> Represent Batch we have 1 image so 1 batch

print(model.predict(test_input))

# 0 mean cat and 1 mean dog