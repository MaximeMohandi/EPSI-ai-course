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

neural_network = keras.models.load_model('neuralNetwork.keras')

print('network properties')
neural_network.summary()


test_nb = 25

print("Calcul des sorties associées a une entrée :")
print("Test N°{} => attendu : {}".format(test_nb, y_test[test_nb].argmax()))

print("Résultat attendu : ")
print("Sorties brutes:",  neural_network.predict(x_test[test_nb:test_nb+1])[0])
print("Classe de sortie:", neural_network.predict_classes(x_test[test_nb:test_nb+1])[0], '\n')

print('FIABILITE DU RESEAU:') 
print('====================')
perf=neural_network.evaluate(
    x=x_test, 
    y=y_test
)

print("Taux d'exactitude sur le jeu de test: {:.2f}%".format(perf[1]*100)) 
NbErreurs=int(10000*(1-perf[1])) 
print("==>",NbErreurs," erreurs de classification !") 
print("==>",10000-NbErreurs," bonnes classifications !") 
Predictions=neural_network.predict_classes(x_test) 

i=-1 
Couleur='Green' 

plt.figure(figsize=(12,8), dpi=200) 
for NoImage in range(12*8): 
    i=i+1 
    while y_test[i].argmax() != Predictions[i]: 
        i=i+1 

    plt.subplot(8,12,NoImage+1) 
    plt.imshow(x_test[i].reshape(28,28), cmap='gray', interpolation='none') 
    plt.title("Prédit:{} - Correct:{}".format(neural_network.predict_classes(x_test[i:i+1])[0],y_test[i].argmax()), pad=2,size=5, color=Couleur)
    plt.xticks(ticks=[]) 
    plt.yticks(ticks=[]) 

plt.show()

