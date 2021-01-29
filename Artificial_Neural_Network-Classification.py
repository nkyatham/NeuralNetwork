import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("/Users/nithinkyatham/Downloads/Churn_Modelling.csv")


X = dataset.iloc[:,3:13].values

Y=dataset.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder = LabelEncoder()
X[:,2] = labelencoder.fit_transform(X[:,2])

from sklearn.compose import ColumnTransformer
columntransformer = ColumnTransformer([("Geography", OneHotEncoder(), [1])], remainder = 'passthrough')
X = columntransformer.fit_transform(X)
X = X.astype('float64')

X=X[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# To Initialize ANN we use Sequential
from keras.models import Sequential

# To Add layers to ANN we use Dense
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(input_dim = 11, units = 6, activation="relu", kernel_initializer="uniform"))
classifier.add(Dense(units = 6, activation="relu", kernel_initializer="uniform"))
classifier.add(Dense(units = 1, activation="sigmoid", kernel_initializer="uniform"))

# Gradient Descent or Compiling ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

#Training ANN
classifier.fit(X_train,Y_train,batch_size=10,nb_epoch=200)

# Predicting ANN
y_pred = classifier.predict(X_test)

# Checking the Accuracy of prediction using Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)



