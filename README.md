# pruebagithub
Esta es una prueba que realizo para verificar que estoy aprendiendo a utilizar GitHub
# A continuación comparto un código para entrenar una red neuronal:

#Cargamos las librerías necesarias
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam
import random
#Establecemos la semilla para TensorFlow
seed_value = 42
tf.random.set_seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)
#Cargamos la librería necesaria para buscar los mejores hiperparámetros
!pip install -U keras-tuner
from keras_tuner.tuners import RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters
#Cargamos las librerías necesarias para realizar transfer learning
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers

#Cargamos el set de datos con el que vamos a trabajar
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#Exploramos los datos
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)

#Exploración de los datos
%matplotlib inline

plt.imshow(x_train[5],cmap='binary')
plt.show()

#Revisamos cómo son las características
x_train[88].shape

#Mostramos una imagen y la exploramos
x_train[88][1]  

#Definimos el modelo 1
model_1 = Sequential()
model_1.add(Conv2D(32, kernel_size=(3,3), activation = 'relu', input_shape=(32,32,3)))
model_1.add(MaxPooling2D(pool_size=(2,2)))
model_1.add(Dropout(0.2))
model_1.add(Conv2D(64, kernel_size=(3,3), activation = 'relu'))
model_1.add(MaxPooling2D(pool_size=(2,2)))
model_1.add(Dropout(0.2))
model_1.add(Conv2D(128, kernel_size=(3,3), activation = 'relu'))
model_1.add(MaxPooling2D(pool_size=(2,2)))
model_1.add(Dropout(0.2))
model_1.add(Flatten())
model_1.add(Dense(128, activation = 'relu'))
model_1.add(Dense(10, activation = 'softmax')) #ultima capa
model_1.summary()
# El dropout no puede ir al final de todo, o sea no se puede poner en la capa de salida!!
# Puedo poner un dropout en la capa de entrada, de hecho se puede poner el dropout como capa de entrada
# La función de activación se pone en las capas de proceso, no en la capa dropout

# Compilamos el modelo 1
model_1.compile(loss="categorical_crossentropy",optimizer='adam',metrics=["accuracy","mse"])
model_1.summary()

#Para el modelo 1 colocaremos 20 epochs y una paciencia de 3
batch_size_1 = 50
n_epochs_1= 20
callbacks_1= [EarlyStopping(monitor="val_accuracy",patience=3)]

# Para el modelo 1
histo_1=model_1.fit(x=X_train, y=Y_train,batch_size=batch_size_1, epochs=n_epochs_1,callbacks=callbacks_1, verbose = 1,
                  validation_data=(X_test, Y_test), shuffle=True)
