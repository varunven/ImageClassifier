import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import requests
from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup
import csv

def main():
    url = "https://www.pokemon.com/us/pokedex/"
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    lists = soup.find_all('ul')
    pokemon = []
    for item in lists:
        if len(item) > 1000:
            pokemon = item            
    names = []
    for item in pokemon:
        i = str(item)
        if i != ("\n"):
            i2 = i.split('>')[2]
            names.append(i2)
    for idx, pokemon in enumerate(names):
        pokemon = pokemon.split('-')[1]
        pokemon = pokemon[1:-3]
        names[idx] = pokemon 
        if pokemon == 'Nidoran♀':
            names[idx] = 'Nidoran-female'
        if pokemon == 'Nidoran♂':
            names[idx] = 'Nidoran-male'
        if pokemon == "Farfetch'd":
            names[idx] = 'Farfetchd'
        if pokemon == "Mr. Mime":
            names[idx] = 'Mr-Mime'
        if pokemon == "Mime Jr.":
            names[idx] = 'Mime-Jr'
        if pokemon == "Pory":
            names[idx] = 'Porygon-Z'
        if pokemon == "Type: Null":
            names[idx] = 'Type-Null'
        if pokemon == "Jan":
            names[idx] = 'Jangmo-o'
        if pokemon == "Hak":
            names[idx] = 'Hakamo-o'
        if pokemon == "Ko":
            names[idx] = 'Kommo-o'
        if pokemon == "Tapu Koko" or pokemon == "Tapu Lele" or pokemon == "Tapu Bulu" or pokemon == "Tapu Fini":
            pokemon = pokemon.replace(" ", "-")
            names[idx] = pokemon
        if pokemon == "Sirfetch'd":
            names[idx] = 'Sirfetchd'
        if pokemon == "Mr. Rime":
            names[idx] = 'Mr-Rime'
        if pokemon == "":
            names[idx] = 'Ho-oh'
    pokelist = []
    for pokemon in names:
        name_and_url = {}
        temp_url = url+pokemon
        temp_page = requests.get(temp_url)
        temp_soup = BeautifulSoup(temp_page.content, 'html.parser')
        image = temp_soup.find("meta", property="og:image")
        image = image.get("content")
        name_and_url["name"] = pokemon
        name_and_url["url"]= image
        pokelist.append(name_and_url)
    keys = pokelist[0].keys()
    with open(r'C:\Users\dswhi\.vscode\Pokemon Image Creator\pokemon_list.csv', 'w') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(pokelist)

if __name__=="__main__":
    main()