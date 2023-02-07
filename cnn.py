import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Activation, Dense, Dropout, Conv1D, Flatten, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D, AveragePooling1D, Input, Add
from keras.optimizers import SGD
import pickle

with open('Data/train_test_0.1_0.2s.pickle', 'rb') as handle:
    data = pickle.load(handle)


x_train = data['X_train']
y_train = data['y_train']
x_test = data['X_test']
y_test = data['y_test']

print(x_test.shape)

model = Sequential()

model.add(Conv1D(32, 3, activation='relu', input_shape=(71,1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='softmax'))

# Compilation du modèle
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entraînement du modèle
model.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_test, y_test))

# Évaluation du modèle sur les données de test
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)
