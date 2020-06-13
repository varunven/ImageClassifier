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

SHAPE3D = 120
SHAPEART = 300
SHAPE = 475

sprites3d = r"C:/Users/dswhi/.vscode/Pokemon Image Creator/MyScraped3DImages"
spritesart = r"C:/Users/dswhi/.vscode/Pokemon Image Creator/MyScrapedArtImages"
sprites = r"C:/Users/dswhi/.vscode/Pokemon Image Creator/MyScrapedImages"

curr = sprites 
# curr = sprites3d
# curr = spritesart
currshape = SHAPE
# currshape = SHAPE3D
# currshape = SHAPEART

Pokemon = namedtuple('Pokemon', 'name, type1, type2')

def types(path):
    ''' 
    Reads the csv file to generate a namedtuple Pokemon for each line 
    returns a dictionary of pokemon no. as key and (name, type1, type2) as value
    '''
    file = open(path, 'r').read().splitlines()
    dict = {}
    file.pop(0)
    for f in file:
        info = f.split(',')
        mon = info[4]
        dict[mon] = Pokemon(info[0], info[2], info[3])
    return dict

def get_pokemon(filepath):
    type_dict = types(r'C:/Users/dswhi/.vscode/Pokemon Image Creator/pokemon_3d_with_types.csv')
    ''' returns the Pokemon tuple based on file path '''
    filename = filepath.split('/')[-1]
    id = ''.join(re.findall(r'\b\d+\b', filename))
    return type_dict[id]

def load_models():
    ''' loads the models '''
    classifier = load_model(curr+"/model/classifier.h5")
    print("Loaded classifiers from disk")
    return classifier

def load_image(imagepath):
    ''' helper to load image '''
    test_image = image.load_img(imagepath, target_size = (currshape, currshape))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    return test_image

def stats(type_dict, classifier, primary = True):
    ''' 
    computes manually the true and predicted numpy arrays 
    then computes the precision, recall, and f-measure statistics
    '''
    test = curr+"/test"
    
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
    '''
    function to perform prediction using models
    evaluate determines whether to evaluate model through 
    the Keras evaluation function, computing precision/recall statistics
    predict will sample a random pokemon image to predict
    '''
    
    classifier = load_models()

    # tests
    test = curr
    path = test
    imagepath = path + '/' + random.choice(os.listdir(path))
    # load test sets
    test_datagen = ImageDataGenerator(rescale = 1./255)
    test_set = test_datagen.flow_from_directory(test, target_size = (currshape, currshape), class_mode = 'categorical')
    if predict:
        # predict random pokemon
        test_image = load_image(imagepath)
        result1 = classifier.predict_classes(test_image)
        print(result1)
        predicted_type = [typex for typex, index in test_set.class_indices.items() if index == result1[0]][0]

        pokemon = get_pokemon(imagepath)
        print("The Pokemon " + pokemon.name + " has type " + pokemon.type1)
        print("The predicted type is " + predicted_type)
        
    if evaluate:
        # evaluates models
        print("Primary Type:")
        accuracy = classifier.evaluate_generator(test_set)
       
        v_t, v_p = stats(test_set.class_indices, classifier)
        report = classification_report(v_t, v_p, target_names = list(test_set.class_indices.keys()))
        print(report)
        print("Accuracy: ", accuracy_score(v_t, v_p))

        print("\n ------------------------------- \n")
        print("Evaluation Statistics")
        print("Loss: ", accuracy[0])
        print("Accuracy: ", accuracy[1])

    return classifier, test_set

def predict_single(img, classifier, test_set):
        ''' predicts the type of the pokemon given by "img" '''
        test_image = load_image(img)
        result1 = classifier.predict_classes(test_image)
        predicted_type = [type for type, index in test_set.class_indices.items() if index == result1[0]][0]
        test_image.show()
        plt.show()
        return predicted_type

if __name__ == "__main__":
    p = True
    e = True
    run(evaluate = e, predict = p)