from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras.models import load_model
from collections import namedtuple
from sklearn import metrics
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import os,random, sys, re
import csv

SHAPE = 120
curr = "MyScraped3DImages/"

Pokemon = namedtuple('Pokemon', 'name, type1, type2')

def types(path):
    ''' 
    Returns info about Pokemon
    '''
    dict = {}
    file = open(path, 'r', encoding='utf-8').read().splitlines()
    for row in file:
        info = row.split(',')
        if len(info)>1:
            mon = info[4]
            dict[mon] = Pokemon(info[0], info[2], info[3])
    return dict

def get_pokemon(filepath):
    type_dict = types('pokemon_3d_with_types.csv')
    filename = filepath.split('/')[-1]
    filename = filename[:-4]
    id = ''
    for mon in type_dict:
        if type_dict[mon].name == filename:
            print(mon)
            id = mon
    return type_dict[id]

def load_models():
    classifier = load_model(curr+"/model/classifier1.h5")
    print("Loaded classifiers from disk")
    return classifier

def load_image(imagepath):
    test_image = image.load_img(imagepath, target_size = (SHAPE, SHAPE))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    return test_image

def stats(type_dict, classifier, primary = True):
    test = curr+'type1/test'
    
    true_types = []
    pred_types =[]
    types = [x for x in os.listdir(test) if os.path.isdir(os.path.join(test, x))]

    for t in types:
        true_type = t.split('/')[-1]
        for img in os.listdir(test + '/' + t):
            imgp = test + '/' + t + '/' + img
            test_img = load_image(imgp)
            pred = classifier.predict_classes(test_img)
            true_types.append(type_dict[true_type])
            pred_types.append(pred[0])
    return np.array(true_types), np.array(pred_types)

def run(evaluate = True, predict = True):    
    classifier = load_models()

    test = curr+'type1/test'
    path = test + '/' + random.choice([x for x in os.listdir(test) if os.path.isdir(os.path.join(test, x))])
    imagepath = path + '/' + random.choice(os.listdir(path))
    test_datagen = ImageDataGenerator(rescale = 1./255)
    test_set = test_datagen.flow_from_directory(test, target_size = (SHAPE, SHAPE), class_mode = 'categorical')
    
    if predict:
        test_image = load_image(imagepath)
        result1 = classifier.predict_classes(test_image)
        predicted_type = [typex for typex, index in test_set.class_indices.items() if index == result1[0]][0]

        pokemon = get_pokemon(imagepath)
        print("The Pokemon " + pokemon.name + " has type " + pokemon.type1)
        print("The predicted type is " + predicted_type)
        
    if evaluate:
        accuracy = classifier.evaluate_generator(test_set)
        print("Loss: ", accuracy[0])
        print("Accuracy: ", accuracy[1])

    return classifier, test_set

def predict_single(pokemon, classifier, test_set):
    test_image = load_image(pokemon)
    result1 = classifier.predict_classes(test_image)
    predicted_type = [type for type, index in test_set.class_indices.items() if index == result1[0]][0]
    test_image.show()
    plt.show()
    return predicted_type

if __name__ == "__main__":
    p = True
    e = True
    run(True, True)