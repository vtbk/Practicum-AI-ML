import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

# ==== OPGAVE 1 ====
def plotNumber(nrVector):
    # Let op: de manier waarop de data is opgesteld vereist dat je gebruik maakt
    # van de Fortran index-volgorde – de eerste index verandert het snelst, de 
    # laatste index het langzaamst; als je dat niet doet, wordt het plaatje 
    # gespiegeld en geroteerd. Zie de documentatie op 
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html
    
    nrVector = nrVector.reshape((20, 20), order = 'F')
    plt.matshow(nrVector)
    plt.show()
    

# ==== OPGAVE 2a ====
def sigmoid(z):
    # Maak de code die de sigmoid van de input z teruggeeft. Zorg er hierbij
    # voor dat de code zowel werkt wanneer z een getal is als wanneer z een
    # vector is.
    # Maak gebruik van de methode exp() in NumPy.
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
    
    ####TODO FIX THIS FIX THIS FIX THIS ####
    cols = y.T[0] #Get the data
    cols = cols-1 #Remove 1 from all values (because 1 has to go on index 0, 2 on 1 etc...)
    rows = [i for i in range(len(cols))] #Get the all the values
    data = [1 for _ in range(len(cols))] #Get a similair matrix filled with 1's
    w = max(cols)+1 # arrays zijn zero-based
    y_vec = csr_matrix((data, (rows, cols)), shape=(len(rows), w)).toarray()
    return y_vec


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

    #https://blog.quantinsti.com/forward-propagation-neural-networks/
    


    X = np.insert(X, 0, 1, axis=1)
    a2 = sigmoid(np.dot(X, Theta1.T))
    a2 = np.insert(a2, 0, 1, axis=1)
    r = sigmoid(np.dot(a2, Theta2.T))
    #print(sigmoid(r))
    return r


# ===== deel 2: =====
def computeCost(Theta1, Theta2, X, y):
    # Deze methode maakt gebruik van de methode predictNumber() die je hierboven hebt
    # geïmplementeerd. Hier wordt het voorspelde getal vergeleken met de werkelijk 
    # waarde (die in de parameter y is meegegeven) en wordt de totale kost van deze
    # voorspelling (dus met de huidige waarden van Theta1 en Theta2) berekend en
    # geretourneerd.
    # Let op: de y die hier binnenkomt is de m×1-vector met waarden van 1...10. 
    # Maak gebruik van de methode get_y_matrix() die je in opgave 2a hebt gemaakt
    # om deze om te zetten naar een matrix. 
    y = get_y_matrix(y, y.shape[0])
    predictions = predictNumber(Theta1, Theta2, X)

    #Calculate error margin of predicted chance 
    error = lambda chance, pred_chance: -1 * chance * np.log(pred_chance) - (1 - chance) * np.log(1 - pred_chance)
    #For every single prediction: check error margin of predicted chance for all properties (i.e. num 0-9)
    total_error = np.sum([error(ch, pr) for c, p in zip(y, predictions) for ch, pr in zip(c, p)])
    cost = total_error / X.shape[0]
    return cost
    
'''
    y = get_y_matrix(y, y.shape[0])
    predictions = predictNumber(Theta1, Theta2, X)

    #calculate error margin of predicted chance vs actual chance
    error = lambda chance, pred_chance: -1 * chance * np.log(pred_chance) - (1 - chance) * np.log(1 - pred_chance)
    #For every single prediction: check error margin of predicted chance for all properties (i.e. num 0-9)
    total_error = np.array([error(ch, pr) for c, p in zip(y, predictions) for ch, pr in zip(c, p)]).sum()
    cost = total_error / X.shape[0]
    return cost
    
    total_cost = 0
    for correct, prediction in zip(y_matrix, predictions):
        prediction_cost = 0
        for chance, predicted_chance in zip(correct, prediction):
            prediction_cost +=  -1 * chance * np.log(predicted_chance) - (1 - chance) * np.log(1 - predicted_chance)
        total_cost += prediction_cost

    return total_cost/X.shape[0]

    cost = 0
    for index, pred in enumerate(predictions):
        for i, pixel in enumerate(pred):
            correct_pixel = y_matrix[index][i]

            tcost = -1 * correct_pixel * np.log(pixel) - (1 - correct_pixel) * np.log(1 - pixel)
            print(tcost)
            cost += tcost #probably nicer to calculate cost per pred instead of pixel of pred
    print(cost / 5000)
    return cost/5000
'''
# ==== OPGAVE 3a ====
def sigmoidGradient(z): 
    # Retourneer hier de waarde van de afgeleide van de sigmoïdefunctie.
    # Zie de opgave voor de exacte formule. Zorg ervoor dat deze werkt met
    # scalaire waarden en met vectoren.
    return sigmoid(z) * (1 - sigmoid(z))
    

# ==== OPGAVE 3b ====
def nnCheckGradients(Theta1, Theta2, X, y): 
    # Retourneer de gradiënten van Theta1 en Theta2, gegeven de waarden van X en van y
    # Zie het stappenplan in de opgaven voor een mogelijke uitwerking.

    Delta2 = np.zeros(Theta1.shape)
    Delta3 = np.zeros(Theta2.shape)
    m = X.shape[0] #voorbeeldwaarde; dit moet je natuurlijk aanpassen naar de echte waarde van m
    
    predicted = predictNumber(Theta1, Theta2, X)
    y = get_y_matrix(y, y.shape[1])

    for i in range(m): 
        a1 = X[i]
        a1 = np.insert(a1, 0, 1, axis=0)
        a2 = sigmoid(np.dot(a1, Theta1.T))
        a2 = np.insert(a2, 0, 1, axis=0)
        a3 = sigmoid(np.dot(a2, Theta2.T))

        #backprop
        s = [0, 0, 0, 0]
        s[3] = predicted[i] - y[i] #or a3 - y[i]
        s[2] = np.dot(Theta2.T, s[3])[:1] * sigmoidGradient(np.dot(a1, Theta1.T))

        s[2] = s[2].reshape((s[2].shape[0], 1))
        a1 = a1.reshape((1, a1.shape[0]))
        Delta2 = Delta2 + np.dot(s[2], a1)

        s[3] = s[3].reshape((s[3].shape[0], 1))
        a2 = a2.reshape((1, a2.shape[0]))

        Delta3 = Delta3 + np.dot(s[3], a2)
        
        # print(Delta3[3])
        # print("Shape of {} is {}".format('s[3]', s[3].shape))
        # print("Shape of {} is {}".format('s[2]', s[2].shape))
        # print("Shape of {} is {}".format('a1', a1.shape))
        # print("Shape of {} is {}".format('a2', a2.shape))
        # print("Shape of {} is {}".format('a3', a3.shape))
        # # print("Shape of {} is {}".format('o', o.shape))
        # # print("Shape of {} is {}".format('z', z.shape))
        # print("Shape of {} is {}".format('Theta2', Theta1.shape))
        # print("Shape of {} is {}".format('Theta2', Theta2.shape))
        # print("Shape of {} is {}".format('Delta2', Delta2.shape))
        # print("Shape of {} is {}".format('Delta3', Delta3.shape))
        # exit()
        
        
    Delta2_grad = Delta2 / m
    Delta3_grad = Delta3 / m
    
    return Delta2_grad, Delta3_grad


'''
        s = [0, 0, 0, 0]
        s[3] = predicted[i] - y[i] #or a3 - y[i]
        s[2] = np.dot(Theta2.T, s[3])[:1] * sigmoidGradient(np.dot(a1, Theta1.T))

        o = s[2].reshape((s[2].shape[0], 1))
        z = a1[:1].reshape((1, a1[:1].shape[0]))
        #Delta2 = Delta2 + np.outer(s[2], a1.T)
        Delta2 = Delta2 + np.dot(o, z.T)
        print("Shape of {} is {}".format('s[3]', s[3].shape))

        #Delta3 = Delta3 + np.outer(s[3], a2.T)
        Delta3 = Delta3 + np.vdot(s[3].T, a2)
        
        print(Delta3[-1])
'''
