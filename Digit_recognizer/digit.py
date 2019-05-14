import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from PIL import Image

random_seed = 42

def read_data(path, detail = 0, label_exist = 1):
    train_df = pd.read_csv(path)
    train_df = train_df.apply(pd.to_numeric, downcast='unsigned')
    if label_exist == 1:
        train_labels = train_df.iloc[:, :1]
        train_data = train_df.iloc[:, 1:]
    else:
        train_data = train_df
    if detail == 1:
        print(train_data.info())
        print(train_data.head())
        if label_exist == 1:
            print(train_labels.info())
            print(train_labels.head())
    if label_exist == 1:
        return (train_df, train_data, train_labels)
    else:
        return (train_df, train_data)

def visualize_sample(data, nb = 20):
    for i in range (nb):
        img = np.reshape(data[i], (int(sqrt(data[i].size)), -1))
        img = Image.fromarray(img)
        plt.subplot(nb / 10 + 1 ,10, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.subplots_adjust(wspace=0)
    plt.show()

def count_classes(labels):
    train_examples = labels.size
    number_of_labels = len(set(labels))
    labels_classes = [0] * number_of_labels
    for label in labels:
        if label >= 0 & label < number_of_labels:
            labels_classes[label] += 1
    for idx, label in enumerate(labels_classes):
        print("Label %d has %4d occurences %5.2f%%" % (idx, label, label * 100 / train_examples))
    plt.bar(x = [*range(10)], height=labels_classes)
    plt.show()

train_df, train_data, train_labels = read_data("data/train.csv", 0, label_exist = 1)
test_df, test_data = read_data("data/test.csv", 0, label_exist = 0)

train_data = np.array(train_data, dtype=np.float64)
test_data = np.array(test_data, dtype=np.float64)

train_labels = np.array(train_labels['label'])
# visualize_sample(train_data, 50)
# count_classes(train_labels)

#Normalizing
train_data = train_data / 255.0;
test_data = test_data / 255.0;

train_data = train_data.reshape(-1, 28, 28, 1);
test_data = test_data.reshape(-1, 28, 28, 1);

print(train_data.shape)

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

train_data, X_val, train_labels, Y_val = train_test_split(train_data, train_labels, test_size = 0.2, random_state=random_seed)

import keras

one_hot_labels = keras.utils.to_categorical(train_labels, num_classes=10)
Y_val = keras.utils.to_categorical(Y_val, num_classes=10)

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, AveragePooling2D, Dropout

# #Model 1
# model = Sequential()
# model.add(Conv2D(32, kernel_size = 3, activation = 'relu'))
# model.add(Conv2D(16, kernel_size = 3, activation = 'relu'))
# model.add(Conv2D(8, kernel_size = 3, padding ="same", activation = 'relu'))
# model.add(Flatten())
# model.add(Dense(64, input_dim = 576, activation = 'relu'))
# model.add(Dense(64, input_dim = 64, activation = 'relu'))
# model.add(Dense(32, input_dim = 64, activation = 'relu'))
# model.add(Dense(32, input_dim = 32, activation = 'relu'))
# model.add(Dense(16, input_dim = 32, activation = 'relu'))
# model.add(Dense(10, input_dim = 16, activation = 'softmax'))
#

#Model 2 type: Lenet 5
model = Sequential()
model.add(Conv2D(12, kernel_size = 5, padding = "same", activation = 'relu'))
model.add(MaxPooling2D((3,3), strides = 2))
# model.add(Dropout(rate = 0.1))
model.add(Conv2D(32, kernel_size = 5, activation = 'relu'))
model.add(MaxPooling2D((3,3), strides = 2))
# model.add(Dropout(rate = 0.1))
model.add(Flatten())
model.add(Dropout(rate = 0.3))
model.add(Dense(120, activation = 'relu'))
model.add(Dropout(rate = 0.5))
model.add(Dense(84, input_dim = 120, activation = 'relu'))
model.add(Dropout(rate = 0.3))
model.add(Dense(10, input_dim = 84, activation = 'softmax'))


model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics = ['accuracy'])

model.fit(train_data, one_hot_labels, epochs = 60, batch_size = 64, validation_data = (X_val, Y_val), verbose = 1)

import os

#Save submission
test_labels = model.predict(test_data)
results = np.argmax(test_labels, axis = 1)
if (os.path.exists('submission.txt')):
    os.remove('submission.txt')
with open('submission.txt', 'w') as file:
    file.write("ImageId,Label\n")
    for i, result in enumerate(results):
        line = str(i + 1) + ',' + str(result) + '\n'
        file.write(line)
    print("Write successfull")

confusion_mtx = confusion_matrix(np.argmax(Y_val, axis = 1), np.argmax(model.predict(X_val), axis = 1))

import seaborn as sns
sns.heatmap(confusion_mtx, annot=True,  fmt="d", vmin=0, vmax=20)
plt.show()
