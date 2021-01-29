import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow.keras as keras

dataset = keras.datasets.fashion_mnist
(training_images,training_labels),(test_images,test_labels)= dataset.load_data()

training_images.size

training_images=training_images/255.0
training_images = training_images.reshape(60000,28,28,1)

test_images=test_images/255.0
test_images = test_images.reshape(10000,28,28,1)

from keras.models import Sequential

from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
#ConClassifier = Sequential()

ConClassifier = Sequential([Conv2D(4,(3,3),activation='relu',input_shape=(28,28,1)),
                            MaxPooling2D(2,2),
                            Conv2D(4,(3,3),activation='relu'),
                            MaxPooling2D(2,2)
                            ,Flatten(),
                            Dense(units=8,activation='relu'),
                            Dense(10,activation='softmax')])


ConClassifier.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
ConClassifier.summary()
ConClassifier.fit(training_images,training_labels,batch_size=1000,epochs=10)
val = ConClassifier.evaluate(test_images,test_labels)

"""
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
training_labels = sc.fit_transform(training_labels)
from sklearn.model_selection import train_test_split
"""



