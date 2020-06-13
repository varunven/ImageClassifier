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
    url = "https://pokemondb.net/pokedex/national"
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    pokemons = soup.find_all('div', class_='infocard ')
    pokemonlist = []
    for item in pokemons:
        i = str(item)
        isplit = i.split('>')
        pokemon = isplit[12][0:-3]
        sprite = isplit[3].split('=')[3][1:-1]
        num = isplit[9][1:4]
        type1 = isplit[16][0:-3]
        type2 = isplit[18][0:-3]
        if type2 == "</s":
            type2 = "None"
        if pokemon == 'Nidoran♀':
            pokemon = 'Nidoran-female'
        if pokemon == 'Nidoran♂':
            pokemon = 'Nidoran-male'
        if pokemon == "Farfetch'd":
            pokemon = 'Farfetchd'
        if pokemon == "Mr. Mime":
            pokemon = 'Mr-Mime'
        if pokemon == "Mime Jr.":
            pokemon = 'Mime-Jr'
        if pokemon == "Type: Null":
            pokemon = 'Type-Null'
        if pokemon == "Tapu Koko" or pokemon == "Tapu Lele" or pokemon == "Tapu Bulu" or pokemon == "Tapu Fini":
            pokemon = pokemon.replace(" ", "-")
            pokemon = pokemon
        if pokemon == "Sirfetch'd":
            pokemon = 'Sirfetchd'
        if pokemon == "Mr. Rime":
            pokemon = 'Mr-Rime'
        to_add = {}
        to_add["name"] = pokemon
        to_add["url"] = sprite
        to_add["type1"] = type1
        to_add["type2"] = type2
        to_add["dex_num"] = num
        pokemonlist.append(to_add)
    keys = pokemonlist[0].keys()
    with open(r'C:\Users\dswhi\.vscode\Pokemon Image Creator\pokemon_3d_with_types.csv', 'w') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(pokemonlist)

if __name__=="__main__":
    main()