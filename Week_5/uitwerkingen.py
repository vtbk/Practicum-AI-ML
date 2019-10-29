import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

# ==== OPGAVE 1 ====
def plotNumber(nrVector):
    nrVector = nrVector.reshape((20, 20), order = 'F')
    plt.matshow(nrVector)
    plt.show()
    

# ==== OPGAVE 2a ====
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
    

# ==== OPGAVE 2b ====
def get_y_matrix(y, m):
    # Gegeven een vector met waarden y_i van 1...x, retourneer een (ijle) matrix
    # van m×x met een 1 op positie y_i en een 0 op de overige posities.
    # Let op: de gegeven vector y is 1-based en de gevraagde matrix is 0-based,
    # dus als y_i=1, dan moet regel i in de matrix [1,0,0, ... 0] zijn, als
    # y_i=10, dan is regel i in de matrix [0,0,...1] (in dit geval is de breedte
    # van de matrix 10 (0-9), maar de methode moet werken voor elke waarde van 
    # y en m
    #https://cmdlinetips.com/2018/03/sparse-matrices-in-python-with-scipy/
    
    cols = y.T[0] 
    cols = cols - 1 #The values decreased by one correspond to the index of the 1 in the to-be-created matrix
    rows = [i for i in range(m)] 
    data = [1 for _ in range(m)] #The actual non-zero data consists of m ones 
    width = max(cols) + 1
    return csr_matrix((data, (rows, cols)), shape=(m, width)).toarray()

# ==== OPGAVE 2c ==== 
# ===== deel 1: =====
def predictNumber(Theta1, Theta2, X):
    # Deze methode moet een matrix teruggeven met de output van het netwerk
    # gegeven de waarden van Theta1 en Theta2. Elke regel in deze matrix 
    # is de waarschijnlijkheid dat het sample op die positie (i) het getal
    # is dat met de kolom correspondeert.

    # De matrices Theta1 en Theta2 corresponderen met het gewicht tussen de
    # input-laag en de verborgen laag, en tussen de verborgen laag en de
    # output-laag, respectievelijk. 

    # Een mogelijk stappenplan kan zijn:

    #    1. voeg enen toe aan de gegeven matrix X; dit is de input-matrix a1
    #    2. roep de sigmoid-functie van hierboven aan met a1 als actuele
    #       parameter: dit is de variabele a2
    #    3. voeg enen toe aan de matrix a2, dit is de input voor de laatste
    #       laag in het netwerk
    #    4. roep de sigmoid-functie aan op deze a2; dit is het uiteindelijke
    #       resultaat: de output van het netwerk aan de buitenste laag.

    # Voeg enen toe aan het begin van elke stap en reshape de uiteindelijke
    # vector zodat deze dezelfde dimensionaliteit heeft als y in de exercise.
    a1 = np.insert(X, 0, 1, axis=1)
    a2 = sigmoid(np.dot(a1, Theta1.T))
    a2 = np.insert(a2, 0, 1, axis=1)
    return sigmoid(np.dot(a2, Theta2.T))
    

# ===== deel 2: =====
def computeCost(Theta1, Theta2, X, y):
    y = get_y_matrix(y, y.shape[0])
    predictions = predictNumber(Theta1, Theta2, X)
    #Calculate error margin of predicted chance 
    error = lambda chance, pred_chance: -1 * chance * np.log(pred_chance) - (1 - chance) * np.log(1 - pred_chance)
    #For every single prediction: check error margin of predicted chance for all properties (i.e. num 0-9)
    total_error = np.sum([error(ch, pr) for c, p in zip(y, predictions) for ch, pr in zip(c, p)])
    cost = total_error / X.shape[0]
    return cost
    
# ==== OPGAVE 3a ====
def sigmoidGradient(z): 
    return sigmoid(z) * (1 - sigmoid(z))
    

# ==== OPGAVE 3b ====
def nnCheckGradients(Theta1, Theta2, X, y): 
    Delta2 = np.zeros(Theta1.shape)
    Delta3 = np.zeros(Theta2.shape)
    m = X.shape[0] 
    y = get_y_matrix(y, m)

    for i in range(m): 
        a1 = X[i]
        a1 = np.insert(a1, 0, 1, axis=0)
        a2 = sigmoid(np.dot(a1, Theta1.T))
        a2 = np.insert(a2, 0, 1, axis=0)
        a3 = sigmoid(np.dot(a2, Theta2.T))

        #Backpropagation
        S3 = a3 - y[i]
        S2 = np.dot(Theta2.T, S3)[:1] * sigmoidGradient(np.dot(a1, Theta1.T))
        Delta2 = Delta2 + np.dot(S2.reshape((S2.shape[0], 1)), a1.reshape((1, a1.shape[0])))
        Delta3 = Delta3 + np.dot(S3.reshape((S3.shape[0], 1)), a2.reshape((1, a2.shape[0])))

        # print(Delta3[3])
        # print("Shape of {} is {}".format('s3', S3.shape))
        # print("Shape of {} is {}".format('s2', S2.shape))
        # print("Shape of {} is {}".format('a1', a1.shape))
        # print("Shape of {} is {}".format('a2', a2.shape))
        # print("Shape of {} is {}".format('a3', a3.shape))
        # print("Shape of {} is {}".format('Theta2', Theta1.shape))
        # print("Shape of {} is {}".format('Theta2', Theta2.shape))
        # print("Shape of {} is {}".format('Delta2', Delta2.shape))
        # print("Shape of {} is {}".format('Delta3', Delta3.shape))
        # exit()
        
    Delta2_grad = Delta2 / m
    Delta3_grad = Delta3 / m
    
    return Delta2_grad, Delta3_grad

