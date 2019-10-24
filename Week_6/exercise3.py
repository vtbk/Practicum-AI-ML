import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from random import randint
import sys

from uitwerkingen import *

# ==============================================
# HELPER FUNCTIES
def plotMatrix(data):
    plt.figure()
    plt.matshow(data)
    plt.show()


# ==== Laden van de data en zetten van belangrijke variabelen ====
print ("Laden van de data...")
data = keras.datasets.fashion_mnist 
(train_images, train_labels), (test_images, test_labels) = data.load_data()
labels = ['T-shirt/topje', 'Broek', 'Pullover', 'Jurk', 'Jas', 'Sandalen', 'Shirt', 'Sneaker', 'Tas', 'Lage laars'] 
print ("Done.")

print ("Formaat van de train_images: {}".format(train_images.shape))
print ("Formaat van de train_labels: {}".format(train_labels.shape))
print ("Formaat van de test_images: {}".format(test_images.shape))
print ("Formaat van de test_labels: {}".format(test_labels.shape))
print ("Grootte van de labels: {}".format(len(labels)))

# ===============  OPGAVE 1 ======================
# ===============  OPGAVE 1a ======================
# print ("Plotten van een willekeurig plaatje uit de trainings-dataset")
# if (len(sys.argv)>1 and sys.argv[1]=='skip') :
#     print ("Slaan we over")
# else:
#     rnd = randint(0, train_images.shape[0])
#     hyp = labels[train_labels[rnd]]
#     plotImage(train_images[rnd], hyp)

# ===============  OPGAVE 1b ======================
# X = np.array( ([1,2,3,4],[2,2,4,4],[4,3,2,1]) )
# r = X/4
# print ("Aanroepen van de methode scaleData met de matrix:")
# print (X)
# print (scaleData(X))
# print ("Het resultaat zou gelijk moeten zijn aan:")
# print (r)

# train_images = scaleData(train_images)
# test_images = scaleData(test_images)


# ===============  OPGAVE 1c ======================
print ("")
print ("Aanmaken van het model.")
if (len(sys.argv)>1 and sys.argv[1]=='cached') :
    model = tf.keras.models.load_model('cached_model.h5')
else: 
    model = buildModel()
    print(model.summary())
    print(train_images.shape)
    print ("Trainen van het model...") 
    model.fit(train_images, train_labels, epochs=6)
    print ("Training afgerond.")
    model.save('cached_model.h5')
exit() 
# ===============  OPGAVE 2 ======================
print ("")
print ("Bepalen van de confusion matrix van het getrainde netwerk.")
pred = np.argmax(model.predict(test_images), axis=1)
cm = confMatrix(test_labels, pred)

sess = tf.Session()
with sess.as_default():
    data = cm.eval() 

print ("De confusion matrix:") 
if (len(sys.argv)>1 and sys.argv[1]=='skip') :
    print ("Tekenen slaan we over")
else:
    plotMatrix(data)
  
print (data)
print (data.shape)

print ("Bepalen van de tp, tn, fp, fn")
metrics = confEls(data,labels)
print (metrics)
print ("Bepalen van de scores:")
scores = confData(metrics)
print (scores)

print ("Klaar. Ga nu verder met de laatste opgave.")






