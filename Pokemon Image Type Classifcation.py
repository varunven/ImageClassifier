from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model

import numpy as np
import os,random, sys

SHAPE = 120
curr = "MyScraped3DImages/"

#120, 3D. Other two are 300 and 475. Both require resizing of the images before running on the model

def create_cnn():
    ''' 
    creates and returns the convolutional neural network classifier 
    '''
    classifier = Sequential()

    classifier.add(Conv2D(64, (3, 3),
                          padding = 'same',
                          input_shape = (SHAPE, SHAPE, 3),
                          activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Dropout(0.25))
    
    classifier.add(Conv2D(64, (3, 3), padding = 'same', activation = 'relu'))
    classifier.add(Conv2D(64, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Dropout(0.25))

    classifier.add(Flatten())
    classifier.add(Dense(256, input_dim = 4, activation = 'relu'))
    classifier.add(Dropout(0.5))
    
    classifier.add(Dense(18, activation = 'softmax'))

    return classifier

def train():
    '''
    trains the model for primary or secondary types then saves it to local folder
    '''

    classifier = create_cnn()
    BATCH_SIZE = 64
    EPOCHS = 20
    
    # uses categorical because 19 possible types
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

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

    # file path depends on the primary vs secondary type
    train = 'type1_sorted/train' 
    test = 'type1_sorted/test' 
    train = curr+train
    test = curr+test
    # retrieve datasets
    training_set = train_datagen.flow_from_directory(train, target_size = (SHAPE, SHAPE),
                                                     batch_size = BATCH_SIZE, class_mode = 'categorical')
    
    test_set = test_datagen.flow_from_directory(test, target_size = (SHAPE, SHAPE),
                                                batch_size = BATCH_SIZE, class_mode = 'categorical')

    history = classifier.fit_generator(training_set, epochs = EPOCHS, validation_data = test_set)

    if not os.path.exists(os.path.dirname(curr+"model/")):
        os.makedirs(os.path.dirname(curr+"model/"))
    filename = "classifier1"
    classifier.save(curr+"model/" + filename + ".h5")
    print("Saved model to disk")
    return classifier, history

if __name__ == "__main__":
    #type 1
    _, h = train()