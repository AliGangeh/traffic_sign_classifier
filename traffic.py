!git clone https://bitbucket.org/jadslim/german-traffic-signs/

!ls german-traffic-signs

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import pickle
import pandas as pd
import random
import cv2

np.random.seed(0)

with open('german-traffic-signs/train.p', "rb") as f:
  train_data = pickle.load(f)
with open('german-traffic-signs/valid.p', "rb") as f:
  val_data = pickle.load(f)
with open('german-traffic-signs/test.p', "rb") as f:
  test_data = pickle.load(f)

print(type(train_data))

X_train, y_train = train_data["features"], train_data["labels"]
X_val, y_val = val_data["features"], val_data["labels"]
X_test, y_test = test_data["features"], test_data["labels"]

assert(X_train.shape[0] == y_train.shape[0]), "the number of images != number of labels"
assert(X_val.shape[0] == y_val.shape[0]), "the number of images != number of labels"
assert(X_test.shape[0] == y_test.shape[0]), "the number of images != number of labels"
assert(X_train.shape[1:] == (32,32,3)), "dimensions fo the images arent 32 x 32 x 3"
assert(X_val.shape[1:] == (32,32,3)), "dimensions fo the images arent 32 x 32 x 3"
assert(X_test.shape[1:] == (32,32,3)), "dimensions fo the images arent 32 x 32 x 3"

data = pd.read_csv("german-traffic-signs/signnames.csv")

num_of_samples = []

cols = 5
num_classes = 43

fig, axs = plt.subplots(nrows=num_classes, ncols = cols, figsize=(5, 50))
fig.tight_layout()
for i in range(cols):
    for j, row in data.iterrows():
        x_selected = X_train[y_train == j]
        axs[j][i].imshow(x_selected[random.randint(0, (len(x_selected) - 1)), :, :], cmap=plt.get_cmap("gray"))
        axs[j][i].axis("off")
        if i == 2:
            axs[j][i].set_title(str(j) + "-" + row["SignName"])
            num_of_samples.append(len(x_selected))

print(num_of_samples)
plt.figure(figsize=(12, 4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title("Distribution of the training dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.show

plt.imshow(X_train[1000])
plt.axis("off")
print(X_train[1000].shape)
print(y_train[1000])

def grayscale(img):
  img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  return img

img = grayscale(X_train[1000])
plt.imshow(img)
plt.axis("off")
print(img.shape)

def equalize(img):
  img = cv2.equalizeHist(img)
  return img

img = equalize(img)
plt.imshow(img)
plt.axis("off")
print(img.shape)

def preprocessing(img):
  img = grayscale(img)
  img = equalize(img)
  img = img/255
  return img

X_train = np.array(list(map(preprocessing, X_train)))
X_val = np.array(list(map(preprocessing, X_val)))
X_test = np.array(list(map(preprocessing, X_test)))

plt.imshow(X_train[random.randint(0, len(X_train) - 1 )])
plt.axis("off")
print(X_train.shape)

X_train = X_train.reshape(34799 ,32 ,32, 1)
X_test = X_test.reshape(12630, 32, 32, 1)
X_val = X_val.reshape(4410, 32, 32, 1)

y_train = to_categorical(y_train, 43)
y_test = to_catagorical(y_test, 43)
y_val = to_catagorical(y_val, 43)
