import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import time
import os, random, sys
import scipy
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model

SHAPE3D = 120
SHAPEART = 300
SHAPE = 475

sprites3d = r"C:\Users\dswhi\.vscode\Pokemon Image Creator\MyScraped3DImages"
spritesart = r"C:\Users\dswhi\.vscode\Pokemon Image Creator\MyScrapedArtImages"
sprites = r"C:\Users\dswhi\.vscode\Pokemon Image Creator\MyScrapedImages"

curr = sprites 
# curr = sprites3d
# curr = spritesart
currshape = SHAPE
# currshape = SHAPE3D
# currshape = SHAPEART

TF_CPP_MIN_LOG_LEVEL=2

def create_cnn(primary = True):
    ''' 
    creates the convolutional neural network classifier 
    returns the classfier for primary or secondary types
    '''

    # sequential layers
    classifier = Sequential()

    # add convolutional layers
    ''' 
    3 Convolutional Layers, maxpool layer follows 2 Conv2D layers
    3 Dropout layers to prevent overfitting
    after flatten, 2 dense layers to return output
    '''
    classifier.add(Conv2D(64, (3, 3),
                          padding = 'same',
                          input_shape = (currshape, currshape, 3),
                          activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Dropout(0.25))
    
    classifier.add(Conv2D(64, (3, 3), padding = 'same', activation = 'relu'))
    classifier.add(Conv2D(64, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Dropout(0.25))

    # flatten output and create fully connected layers
    classifier.add(Flatten())
    classifier.add(Dense(256, input_dim = 4, activation = 'relu'))
    classifier.add(Dropout(0.5))
    
    # one more category for None in secondary type
    categories = 18 if primary else 19

    classifier.add(Dense(categories, activation = 'softmax'))

    # returns built CNN
    return classifier

def train(primary = True, save = True, plot_classifier = False):
    '''
    trains the model for primary or secondary types.
    Supports saving the model and plotting the model.
    Returns the classifier as well as history object for plotting.
    '''

    # get model
    classifier = create_cnn(primary)

    # batch size
    BATCH_SIZE = 32

    # number of training epochs
    EPOCHS = 20
    
    # uses adam optimizer and crossentropy loss function
    classifier.compile(optimizer = 'adam',
                       loss = 'categorical_crossentropy',
                       metrics = ['accuracy'])

    # data augmentation: prevent further overfit by randomly transforming the training images
    train_datagen = ImageDataGenerator(
        rescale = 1./255,
        rotation_range = 10,
        shear_range = 0.2,
        zoom_range = 0.2,
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        horizontal_flip = True,
        vertical_flip = True)
    
    test_datagen = ImageDataGenerator(rescale = 1./255)

    # file path depends on the model
    train = curr+'/type1/train' if primary else '/type2/train' 
    test = curr+'/type1/test' if primary else '/type2/test' 

    # retrieve datasets
    training_set = train_datagen.flow_from_directory(train,
                                                     target_size = (SHAPE, SHAPE),
                                                     batch_size = BATCH_SIZE,
                                                     class_mode = 'categorical')
    
    test_set = test_datagen.flow_from_directory(test,
                                                target_size = (SHAPE, SHAPE),
                                                batch_size = BATCH_SIZE,
                                                class_mode = 'categorical')

    # training
    history = classifier.fit_generator(training_set,
                                       #steps_per_epoch = 752,
                                       #validation_steps = 188,
                                       epochs = EPOCHS, 
                                       validation_data = test_set)

    # save the classifier
    if save:
        if not os.path.exists(os.path.dirname(curr+"/model")):
            os.makedirs(os.path.dirname(curr+"/model"))
        filename = "classifier1" if primary else "classifier2"
        classifier.save(curr+"/model/" + filename + ".h5")
        print("Saved model to disk")

    return classifier, history

if __name__ == "__main__":

    s = True # save model
    plt = False # plot model layers
    
    # build classifier for type 1 and 2
    _, h = train(primary = True, save = s, plot_classifier = plt)    
    _, h2 = train(primary = False, save = s, plot_classifier = plt)