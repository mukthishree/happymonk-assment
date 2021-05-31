import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('iris.csv')

labelencoder = LabelEncoder()
dataset["variety"] = labelencoder.fit_transform(dataset["variety"])

X = dataset.iloc[:, :4]
y = dataset.iloc[:,4]
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = Sequential()

classifier.add(Dense(6, kernel_initializer = 'he_uniform',activation='relu',input_dim = 4))

classifier.add(Dense(6, kernel_initializer = 'he_uniform',activation='relu'))

classifier.add(Dense(3, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'Adamax', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model_history=classifier.fit(X_train, y_train,validation_split=0.33, batch_size = 10, epochs = 100)


plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

y_pred = classifier.predict_classes(X_test)

score=accuracy_score(y_pred,y_test)
print("score:", score)


y_prediction = labelencoder.inverse_transform(y_pred)
for i in y_prediction:
    print(i)

