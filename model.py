import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
import cv2 as cv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# get files name
import os



IMG_SIZE = 32 # pixxilating the image 


# add paths
DATADIR = "isolated_alphabets_per_alphabet"
CATEGORIES = []
CATEGORIES_TEST = []


DATADIR_TEST = "test"

kind_data = os.listdir(DATADIR)
# put names in CATEGORIES
for i in range(len(kind_data)):
     CATEGORIES.append(kind_data[i])

# get test data


def create_test_data():
    test_images = []
    test_labels = []
    test_ids = []
    for category in CATEGORIES:
        path = os.path.join(DATADIR_TEST, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv.imread(os.path.join(path, img))
                new_array = cv.resize(img_array, (IMG_SIZE, IMG_SIZE))
                test_images.append(new_array)
                test_labels.append(class_num)
                test_ids.append(img)
            except Exception as e:
                pass
    return np.array(test_images),  np.array(test_labels), np.array(test_ids)


# get training data

def create_training_data():
    train_images = []
    train_labels = []
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv.imread(os.path.join(path, img))
                new_array = cv.resize(img_array, (IMG_SIZE, IMG_SIZE))
                train_images.append(new_array)
                train_labels.append(class_num)
            except Exception as e:
                pass
    return np.array(train_images),  np.array(train_labels)



# apply function and storage their values in variables
train_images, train_labels = create_training_data()
test_images, test_labels, test_ids = create_test_data()
# (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()


# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images/255.0, test_images/255.0




#Create the convolutional base
model = models.Sequential()
model.add(layers.Conv2D(128, (3, 3), activation='tanh',
                        input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(256, (3, 3), activation='tanh'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(256, (3, 3), activation='tanh'))
model.add(layers.Flatten())

model.add(layers.Dense(256, activation='tanh'))
model.add(layers.Dense(65))

#opt = tf.keras.optimizers.Adam(learning_rate=0.0001)


model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=65, batch_size=3000,
                    validation_data=(test_images, test_labels))


# make arrays and add id, 
output_csv_array = []
output_csv_array.append(['ID', 'Actual Categry','Predicted Categry','Percent Confidence'])
for index, img in enumerate(test_images):
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    prediction_array = []
    prediction_array.append(test_ids[index])
    prediction_array.append(CATEGORIES[test_labels[index]])
    prediction_array.append(CATEGORIES[np.argmax(score)])
    prediction_array.append(str(round(100 * np.max(score), 2)))
    output_csv_array.append(prediction_array)
    # print(
    #     "This image most likely belongs to {} with a {:.2f} percent confidence. actual type is {}, with id {}"
    #     .format(CATEGORIES[np.argmax(score)], 100 * np.max(score), CATEGORIES[test_labels[index]], test_ids[index])
    # )



import csv
with open('output.csv', 'w') as result_file:
    wr = csv.writer(result_file, dialect='excel')
    wr.writerows(output_csv_array)


# to grapf outcomes

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']


epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'bo', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()


