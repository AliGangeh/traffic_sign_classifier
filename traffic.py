# this was all done with the help of this course: https://www.udemy.com/applied-deep-learningtm-the-complete-self-driving-car-course/
# big thanks to Rayan, Amer, Jad, and Sarmad for making such an awesome course.

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense
from keras.layers import Flatten, Dropout
from keras.utils.np_utils import to_categorical
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import random
import pickle
import pandas as pd
import cv2
import os
import requests
from PIL import Image

# clones training data from bitbucket repo.
myCmd = 'git clone https://bitbucket.org/jadslim/german-traffic-signs \n ls german-traffic-sign'
os.system(myCmd)

np.random.seed(0)

# saves the training, validation, and test data as f and pickles them
with open('german-traffic-signs/train.p', 'rb') as f:
    train_data = pickle.load(f)
with open('german-traffic-signs/valid.p', 'rb') as f:
    val_data = pickle.load(f)
with open('german-traffic-signs/test.p', 'rb') as f:
    test_data = pickle.load(f)

# stores feautres (the image) and labels
X_train, y_train = train_data['features'], train_data['labels']
X_val, y_val = val_data['features'], val_data['labels']
X_test, y_test = test_data['features'], test_data['labels']

# prints how many images there are, the size of the images, and how many color channels it has.
print(X_train.shape)
print(X_test.shape)
print(X_val.shape)

# checks if all the numberes are correct and correspond.
assert(X_train.shape[0] == y_train.shape[0]), "The number of images is not equal to the number of labels."
assert(X_train.shape[1:] == (32,32,3)), "The dimensions of the images are not 32 x 32 x 3."
assert(X_val.shape[0] == y_val.shape[0]), "The number of images is not equal to the number of labels."
assert(X_val.shape[1:] == (32,32,3)), "The dimensions of the images are not 32 x 32 x 3."
assert(X_test.shape[0] == y_test.shape[0]), "The number of images is not equal to the number of labels."
assert(X_test.shape[1:] == (32,32,3)), "The dimensions of the images are not 32 x 32 x 3."

# imports the names of the different signs
data = pd.read_csv('german-traffic-signs/signnames.csv')

# displays a sample set of what the images of the dataset look from each class
num_of_samples=[]
cols = 5
num_classes = 43
fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5,50))
fig.tight_layout()
for i in range(cols):
    for j, row in data.iterrows():
      x_selected = X_train[y_train == j]
      axs[j][i].imshow(x_selected[random.randint(0,(len(x_selected) - 1)), :, :], cmap=plt.get_cmap('hsv'))
      axs[j][i].axis("off")
      if i == 2:
        axs[j][i].set_title(str(j) + " - " + row["SignName"])
        num_of_samples.append(len(x_selected))
plt.show()

# displays distribution of the dataset and the amount of images in each class
plt.figure(figsize=(12, 4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title("Distribution of the train dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.show()

# shows a random image from the daataset
plt.imshow(X_train[random.randint(0, len(X_train) - 1)])
plt.axis("off")
plt.title("example")
plt.show()

# prints the dimensions of the image and the label of the image
print(X_train[random.randint(0, len(X_train) - 1)].shape)
print(y_train[random.randint(0, len(X_train) - 1)])

# converts image to grayscale image
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

# converts random image to grayscale and shows it
img = grayscale(X_train[random.randint(0, len(X_train) - 1)])
plt.imshow(img)
plt.axis("off")
plt.title("grayscale example")
plt.show()

# shows that because it's grayscale it no longer has 3 color chanels but only one (32,32,3) vs (32,32)
print(img.shape)

# defines "equalizing" image normalizes the brightness of it.
def equalize(img):
    img = cv2.equalizeHist(img)
    return img

# displays equlized image
img = equalize(img)
plt.imshow(img)
plt.axis("off")
plt.title("equalized example")

plt.show()

# combines the two above functions and also converts the grayscale value  which is from 0-255 to a value between 0-1 this allows for it to be used in our model more effectively
def preprocess(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img

# preprocess' the data sets
X_train = np.array(list(map(preprocess, X_train)))
X_test = np.array(list(map(preprocess, X_test)))
X_val = np.array(list(map(preprocess, X_val)))

#displays preprocessed example image
plt.imshow(X_train[random.randint(0, len(X_train) - 1)])
plt.axis('off')
plt.title("preprocessed image")
plt.show()
print(X_train.shape)

# reshapes the data set from (34799, 32, 32) to (34799, 32, 32, 1) for the 1 grayscale colorchannel
X_train = X_train.reshape(34799, 32, 32, 1)
X_test = X_test.reshape(12630, 32, 32, 1)
X_val = X_val.reshape(4410, 32, 32, 1)

# generates new image  which are slightly varied from the original datast through the different parameters like width, height, zoom, shear, and rotation
# it then stores them in X and y batch
datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1, rotation_range=10.)
datagen.fit(X_train)
batches = datagen.flow(X_train, y_train, batch_size = 15)
X_batch, y_batch = next(batches)

# displays generated images
fig, axs = plt.subplots(1, 15, figsize=(20, 5))
plt.title("generated images")
fig.tight_layout()
for i in range(15):
    axs[i].imshow(X_batch[i].reshape(32, 32))
    axs[i].axis("off")
plt.show()

print(X_batch.shape)

y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)
y_val = to_categorical(y_val, 43)

# create model
def modified_model():
  model = Sequential()
  model.add(Conv2D(60, (5, 5), input_shape=(32, 32, 1), activation='relu'))
  model.add(Conv2D(60, (5, 5), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(30, (3, 3), activation='relu'))
  model.add(Conv2D(30, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Flatten())
  model.add(Dense(500, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(43, activation='softmax'))

  model.compile(Adam(lr = 0.001), loss='categorical_crossentropy', metrics=['accuracy'])
  return model

model = modified_model()
print(model.summary())

history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=50),
                            steps_per_epoch=2000,
                            epochs=1,
                            validation_data=(X_val, y_val), shuffle = 1)

#plots loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.xlabel('epoch')
plt.plot()
plt.show()

#plots accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['training','test'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.plot()
plt.show()

# evaluates model on test data and returns how it did
score = model.evaluate(X_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# intakes an image from the internet and displays it
url = 'https://c8.alamy.com/comp/A0RX23/cars-and-automobiles-must-turn-left-ahead-sign-A0RX23.jpg'
r = requests.get(url, stream=True)
img = Image.open(r.raw)
plt.imshow(img)
plt.title("test image")
plt.show()

#displays preprocessed image
img = np.asarray(img)
img = cv2.resize(img, (32, 32))
img = preprocess(img)
plt.imshow(img)
img = img.reshape(1, 32, 32, 1)
plt.title("predicted sign: "+ str(model.predict_classes(img)))
plt.show()
