import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# C1: description de la couche de convolution
MonReseau = tf.keras.Sequential()
MonReseau.add(tf.keras.layers.Conv2D(
    filters=6,             # 6 noyaux de convolutions (6 feature maps)
    kernel_size=(5,5),     # noyau de convolution 5x5
    strides=(1,1),         # décalages horizontal=1 / vertical=1
    activation='tanh',     # fct d'activation=Tanh
    input_shape=(28,28,1), # taille des entrées (car c'est la 1ère couche)
    padding='same'         #ajout d'un bord à l'image pour éviter la # réduction de taille (nb de pixels calculé# à partir de la taille du noyau)
))

# S2: description de la couche de pooling (average)

MonReseau.add(tf.keras.layers.AveragePooling2D(
    pool_size=(2,2),       # noyau de pooling 2x2
    strides=(2,2),         # décalages horizontal=2 / vertical=2
    padding='valid'        #pas d'ajout de bord
))

#C3: description de la couche de convolution
MonReseau.add(tf.keras.layers.Conv2D(
    filters=16,            # 16 noyaux de convolutions (16 feature maps)
    kernel_size=(5,5),     # noyau de convolution 5x5
    strides=(1,1),         # décalages horizontal=1 / vertical=1
    activation='tanh',     # fct d'activation=Tanh
    padding='valid'        # pas d'ajout de bord à l'image
)) 

# S4: description de la couche de pooling (average)
MonReseau.add(tf.keras.layers.AveragePooling2D(
    pool_size=(2,2),       # noyau de pooling 2x2
    strides=(2,2),         # décalages horizontal=2 / vertical=2
    padding='valid'        # pas d'ajout de bord à l'image 
))      

# C5: connexion totale entre les pixels et la 1ère couche de 120 neurones
# Mise à plat des 16x(5x5)=400 pixels des images de convolution
MonReseau.add(tf.keras.layers.Flatten())

# Création d'un couche de 120 neurones avec fonction d'activation Tanh
MonReseau.add(tf.keras.layers.Dense(120, activation='tanh'))

# FC6: connexion totale avec couche de 84 neurones avec fct d'activation Tanh
MonReseau.add(tf.keras.layers.Dense(84, activation='tanh'))

# Sortie: 10 neurones avec fct d'activation Softmax
MonReseau.add(tf.keras.layers.Dense(10, activation='softmax'))

#Affichage
MonReseau.summary()

#----------------------------------------------------------------------------# 
# COMPILATION du réseau #  => configuration de la procédure pour l'apprentissage#
# ----------------------------------------------------------------------------
MonReseau.compile(
    optimizer='adam',                # algo d'apprentissage
    loss='categorical_crossentropy', # mesure de l'erreur
    metrics=['accuracy']             # mesure du taux de succès
)            

#----------------------------------------------------------------------------# 
# APPRENTISSAGE du réseau#  => calcul des paramètres du réseau à partir des exemples
# #----------------------------------------------------------------------------
hist=MonReseau.fit(
    x=x_train, # données d'entrée pour l'apprentissage
    y=y_train, # sorties désirées associées aux données d'entrée
    epochs=11, # nombre de cycles d'apprentissage 
    batch_size=128, # taille des lots pour l'apprentissage
    validation_data=(x_test,y_test) # données de test
) 
#------------------------------------------------------------# 
#Affichage des graphiques d'évolutions de l'apprentissage
# ------------------------------------------------------------# 
#création de la figure ('figsize' pour indiquer la taille)
plt.figure(figsize=(8,8))

# evolution du pourcentage des bonnes classifications
plt.subplot(2,1,1)
plt.plot(hist.history['accuracy'],'o-')
plt.plot(hist.history['val_accuracy'],'x-')
plt.title("Taux d'exactitude des prévisions",fontsize=15)
plt.ylabel('Taux exactitude',fontsize=12)
plt.xlabel("Itérations d'apprentissage",fontsize=15)
plt.legend(['apprentissage', 'validation'], loc='lower right',fontsize=12)

# Evolution des valeurs de l'erreur résiduelle moyenne
plt.subplot(2,1,2)
plt.plot(hist.history['loss'],'o-')
plt.plot(hist.history['val_loss'],'x-')
plt.title('Ereur résiduelle moyenne',fontsize=15)
plt.ylabel('Erreur',fontsize=12)
plt.xlabel("Itérations d'apprentissage",fontsize=15)
plt.legend(['apprentissage', 'validation'], loc='upper right',fontsize=12)

plt.tight_layout(h_pad=2.5) # espacement entre les 2 figures
plt.show()

# performances du réseau sur les données de tests
perf=MonReseau.evaluate(x=x_test, y=y_test)
print("Taux d'exactitude sur le jeu de test: {:.2f}%".format(perf[1]*100))