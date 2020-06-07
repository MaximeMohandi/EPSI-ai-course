import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

neural_network = keras.Sequential()
neural_network.add(keras.layers.Dense(
    units=50,
    input_shape=(784,),
    activation="relu"
))

neural_network.add(keras.layers.Dense(
    units=10,
    activation='softmax'
))

neural_network.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

hist = neural_network.fit(
    x=x_train,
    y=y_train,
    epochs=10,
    validation_data=(x_test, y_test)
)

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(hist.history['accuracy'], 'o-')
plt.plot(hist.history['val_accuracy'], 'x-')
plt.title("Taux d'exactitude des prévisions", fontsize=15)
plt.ylabel("Taux d'exactitude", fontsize=12)
plt.xlabel("Itération d'apprentissage", fontsize=15)
plt.legend(['apprentissage', 'validation'], loc='lower right', fontsize=12)

plt.subplot(2, 1, 2) 
plt.plot(hist.history['loss'], 'o-') 
plt.plot(hist.history['val_loss'], 'x-') 
plt.title('Erreur résiduelle moyenne', fontsize=15) 
plt.ylabel('Erreur', fontsize=12) 
plt.xlabel("Itérations d'apprentissage", fontsize=15) 
plt.legend(['apprentissage', 'validation'], loc='upper right', fontsize=12)

plt.tight_layout(h_pad=2.5) 
plt.show()


neural_network.save('neuralNetwork.keras')
keras.models.save_model(neural_network, "neuralNetwork.keras")