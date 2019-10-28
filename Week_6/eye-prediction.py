import argparse
import tensorflow
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import os
from tensorflow import keras
from keras import Sequential
from keras import layers
from pathlib import Path

def plot_image(image, label):
    plt.imshow(image, cmap=plt.cm.binary)
    plt.xlabel(label)
    plt.show()

def nomalize(X):
    max_val = np.amax(X)
    return np.array([val / max_val for val in X])

def shuffle_training_data(training_images, training_labels):
    s = np.arange(0, training_images.shape[0])
    np.random.shuffle(s)
    training_images = training_images[s]
    training_labels = training_labels[s]
    return training_images, training_labels

def create_model():
    model = Sequential()
    model.add(layers.Flatten(input_shape=(75, 75)))
    model.add(layers.Dense(784, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(len(labels), activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, batch_size, epochs):
    model.fit(training_images, training_labels, epochs = epochs, batch_size = batch_size, validation_data=(validation_images, validation_labels))
    return model

def test_model(model):
    pred = np.argmax(model.predict(test_images), axis=1) 
    matrix = tensorflow.math.confusion_matrix(test_labels, pred)
    session = tensorflow.compat.v1.Session()
    with session.as_default(): 
        data = matrix.eval() 
    plt.figure()
    plt.matshow(data)
    plt.show()


#TODO clean this up - shouldn't be using the datasets in such a global manner and data is loaded
#even before the arguments are parsed; slow and inefficient. 

#Load the dataset and split it up into three sets: a training set, a validation set and a test set
directories = list(Path('Fundus-data').glob('*'))
labels = [os.path.basename(directory) for directory in directories]
training_images, training_labels, test_images, test_labels = [], [], [], []

#For every directory (and related label) load in image contents. 
#First 3/4 is used for training, last fourth is used for testing
for label, directory in zip(labels, directories):
    count = len(list(Path(directory).glob('*')))
    for i, image_path in enumerate(Path(directory).glob('*')):
        f = img.imread(image_path)
        if i < count * 0.75:
            training_images.append(f)
            training_labels.append(labels.index(label)) 
        else:
            test_images.append(f)
            test_labels.append(labels.index(label)) 

training_images = np.array(training_images)
training_labels = np.array(training_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

#Randomize the order of the set used for training 
training_images, training_labels = shuffle_training_data(training_images, training_labels)
training_images = nomalize(training_images)

#Split up the training image set into a training set and a validation set. 
validation_images, validation_labels = training_images[600:], training_labels[600:]
training_images, training_labels = training_images[:600], training_labels[:600]

if '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--name')
    parser.add_argument('--epochs', '-e', nargs='?', const=50, type=int)
    parser.add_argument('--batch-size', '-b', nargs = '?', const=124, type=int)
    args = parser.parse_args()

    if not args.name:
        print('Please specify the filename with --name')

    elif args.train:
        model = create_model()
        model = train_model(model, args.batch_size, args.epochs)
        model.save(args.name)
    
    elif args.test:
        tensorflow.compat.v1.disable_eager_execution()
        model = keras.models.load_model(args.name)
        test_model(model)