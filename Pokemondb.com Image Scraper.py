import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import requests
from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup
from PIL import Image
import os

def download_image(imageName, imageLink, type1, type2, test):
    type1directory = "C:\\Users\\dswhi\\.vscode\\Pokemon Image Creator\\MyScraped3DImages\\type1\\train\\" + type1 + "\\"
    if test:
        type1directory = "C:\\Users\\dswhi\\.vscode\\Pokemon Image Creator\\MyScraped3DImages\\type1\\test\\" + type1 + "\\"
    if not os.path.exists(os.path.dirname(type1directory)):
        os.makedirs(os.path.dirname(type1directory))
    newImageName =  type1directory + imageName + '.jpg'
    rawImage = requests.get(imageLink, stream=True)
    # save the image received into the file
    with open(newImageName, 'wb') as fd:
        for chunk in rawImage.iter_content(chunk_size=1024):
            fd.write(chunk)
    if type2 != "None":
        type2directory = "C:\\Users\\dswhi\\.vscode\\Pokemon Image Creator\\MyScraped3DImages\\type2\\train\\" + type2 + "\\"
        if test:
            type2directory = "C:\\Users\\dswhi\\.vscode\\Pokemon Image Creator\\MyScraped3DImages\\type2\\test\\" + type2 + "\\"
        if not os.path.exists(os.path.dirname(type2directory)):
            os.makedirs(os.path.dirname(type2directory))
        newImageName =  type2directory + imageName + '.jpg'
        rawImage = requests.get(imageLink, stream=True)
        # save the image received into the file
        with open(newImageName, 'wb') as fd:
            for chunk in rawImage.iter_content(chunk_size=1024):
                fd.write(chunk)

def main():
    df = pd.read_csv(r'C:\Users\dswhi\.vscode\Pokemon Image Creator\pokemon_3d_with_types.csv', encoding='latin-1')
    for i in range(df.shape[0]):
        test = False
        if i%10 == 0:
            test = True
        download_image(df.loc[i, 'name'], df.loc[i, 'url'], df.loc[i, 'type1'], df.loc[i, 'type2'], test)

if __name__=="__main__":
    main()