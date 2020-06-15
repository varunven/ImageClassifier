from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras.datasets import imdb
import numpy as np
import os,random, sys

SHAPE = 475
curr = "C:/Users/dswhi/.vscode/Pokemon Image Creator/MyScrapedImages"

def create_cnn(primary):
    if primary:
        train_data = (curr+r"/type1/train")
        test_data = (curr+r"/type1/test") 
    else:
        train_data = (curr+r"/type2/train")
        test_data = (curr+r"/type2/test")
    classifier = Sequential() 
    classifier.add(Conv2D(64, (3, 3), padding = 'same', input_shape = (SHAPE, SHAPE, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    classifier.add(Conv2D(64, (3, 3), padding = 'same', input_shape = (SHAPE, SHAPE, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    classifier.add(Conv2D(64, (3, 3), padding = 'same', input_shape = (SHAPE, SHAPE, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    classifier.add(Flatten())
    classifier.add(Dense(256, activation = 'relu'))
    classifier.add(Dropout(0.5))

    return classifier 

def train(primary):
    if primary:
        train_data = (curr+r"/type1/train")
        test_data = (curr+r"/type1/test") 
    else:
        train_data = (curr+r"/type2/train")
        test_data = (curr+r"/type2/test")
    
    model = create_cnn(primary)

    # batch size, number of training epochs
    BATCH_SIZE = 64
    EPOCHS = 20

    model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

    train_datagen = ImageDataGenerator(rescale = 1./255, rotation_range = 10, shear_range = 0.2,
        zoom_range = 0.2, width_shift_range = 0.1, height_shift_range = 0.1, horizontal_flip = True,
        vertical_flip = True)
    test_datagen = ImageDataGenerator(rescale = 1./255)

    training_set = train_datagen.flow_from_directory(train_data, target_size = (SHAPE, SHAPE),
                                                    batch_size = BATCH_SIZE, class_mode = 'categorical')
    
    test_set = test_datagen.flow_from_directory(test_data, target_size = (SHAPE, SHAPE),
                                                batch_size = BATCH_SIZE, class_mode = 'categorical')
    model.fit_generator(train_data, epochs=EPOCHS, validation_data=test_set)
    if not os.path.exists(os.path.dirname(curr+"/model/")):
        os.makedirs(os.path.dirname(curr+"/model/"))
    filename = "classifier1" if primary else "classifier2"
    model.save(curr+"/model/" + filename + ".h5")
    print("Saved model to disk")

def main():
    # build classifier for type 1 and 2
    _, h = train(primary = True)    
    _, h2 = train(primary = False)

if __name__ == "__main__":
    main()