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

def download_image(imageName, type1, type2, test):
    imageName = str(imageName)
    type1directory = "C:\\Users\\dswhi\\.vscode\\Pokemon Image Creator\\MyScrapedArtImages\\type1\\train\\" + type1 + "\\"
    if test:
        type1directory = "C:\\Users\\dswhi\\.vscode\\Pokemon Image Creator\\MyScrapedArtImages\\type1\\test\\" + type1 + "\\"
    if not os.path.exists(os.path.dirname(type1directory)):
        os.makedirs(os.path.dirname(type1directory))
    newImageName =  type1directory + imageName + '.png'
    rawImage = r"C:\Users\dswhi\.vscode\Pokemon Image Creator\MyScrapedArtImages/" + imageName + ".png"
    pic = "X"
    with open(rawImage, "rb") as img:
        pic = img.read()
    # save the image received into the file
    with open(newImageName, 'wb') as fd:
        fd.write(pic)
    if type2 != "None":
        type2directory = "C:\\Users\\dswhi\\.vscode\\Pokemon Image Creator\\MyScrapedArtImages\\type2\\train\\" + type2 + "\\"
        if test:
            type2directory = "C:\\Users\\dswhi\\.vscode\\Pokemon Image Creator\\MyScrapedArtImages\\type2\\test\\" + type1 + "\\"
        if not os.path.exists(os.path.dirname(type2directory)):
            os.makedirs(os.path.dirname(type2directory))
        newImageName =  type2directory + imageName + '.png'
        rawImage = r"C:\Users\dswhi\.vscode\Pokemon Image Creator\MyScrapedArtImages/" + imageName + ".png"
        pic = "X"
        with open(rawImage, "rb") as img:
            pic = img.read()
        # save the image received into the file
        with open(newImageName, 'wb') as fd:
            fd.write(pic)


def main():
    df = pd.read_csv(r'C:\Users\dswhi\.vscode\Pokemon Image Creator\pokemon_3d_with_types.csv', encoding='latin-1')
    #this will throw an error at 722 because the images only go up till 721
    for i in range(df.shape[0]):
        test = False
        if i%10 == 0:
            test = True
        download_image(df.loc[i, 'dex_num'], df.loc[i, 'type1'], df.loc[i, 'type2'], test)

if __name__=="__main__":
    main()