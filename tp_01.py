#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 19:19:41 2023

@author: Túlio
"""

from pathlib import Path
import string
from typing import Union

from unidecode import unidecode
from IPython.display import display

BASE_DIR = Path(__file__).resolve().parent

"""
Exercicio 1
"""

def gen_vocabulary(filename: str, text: str) -> "list[str]":
    """
    Gera vocabulario a partir de um arquivo informado, e caso não exista cria
    um novo

    Parameters
    ----------
    filename : str
        Nome do arquivo a ser gerado o vocabulario, caso nao exista cria a 
        partir.
    text : str
        Texto a ser inserido, caso não exista o nome do arquivo inserido.

    Returns
    -------
    final_vector : array of str
        O vocabulário formado pelo texto ou arquivo existente

    """
    # Verifica a exitencia de um arquivo senao cria um novo
    if not (BASE_DIR / filename).exists():
        with open(filename,"w",encoding="utf-8") as file:
            file.write(text)
            file.close()
    # Abre o arquivo caso exista            
    file = open(filename,'r',encoding='utf-8')
    
    # Abre o arquivo como um vetor
    initial_text = file.readlines()
    
    # Vetor temporario para armazenar os valores normalizados
    interim_array = []
    
    for sentence in initial_text:
        # Remove as quebras de linhas e separa as frases por espaco
        aux = sentence.replace('\n',' ').split(' ')
        # Normaliza a frase e add ao array temporario
        interim_array.extend([unidecode(i).lower().translate(str.maketrans('','',string.punctuation)) for i in aux])
    
    # Uso o tipo estruturado set do python para remover duplicatas e depois
    # ordena lexicografica
    final_vector = sorted(set(interim_array))
    # Utilizando essa maneira acabou sobrando uma string vazia
    # no inicio do array
    del final_vector[0]
    
    return final_vector

# =============================================================================
    
"""
Exercicio 2
"""

def gen_bag_of_words(input_1: "list[str]", input_2: Union[str]) -> "list[int]":
    """
    Gera a bag of words do vocabulario informado contra a segunda entrada.

    Parameters
    ----------
    input_1 : "list[str]"
        Vocabulario de entrada.
    input_2 : Union[str]
        Texto de onde gerar a bag of words.

    Returns
    -------
    bag_of_words : vetor de int
        A bag of words propriamente dita.

    """
    # Normalizando o texto para comparacao
    normalize_input_2 = unidecode(input_2).lower().translate(str.maketrans('','',string.punctuation))    
    # Separando os espacos e transformando em array
    vector = normalize_input_2.split(' ')
    
    # Vetor para armazenar a ausencia ou presenca do termo
    bag_of_words = []
    # Dicionario temporario para armazenar a ausencia ou presença de acordo com
    # o termo
    temp_dict = {}
    
    # Percorrendo a entrada e o texto para comparacao
    for word in input_1:
        for i in vector:
            if word == i:
                # Verifica a palavra caso exista adiciona 1 senao 0
                temp_dict[word] = 1
                break
            else:
                temp_dict[word] = 0
    
    # Distribui os valores do dicionario ao array de bag of words
    bag_of_words.extend(temp_dict.values())
    
    return bag_of_words
        
if __name__ == "__main__":
    # Onde executar as funcoes
    hino_tricolor = """
    Salve o Tricolor Paulista
    Amado clube brasileiro
    Tu és forte, tu és grande
    Dentre os grandes, és o primeiro
    Tu és forte, tu és grande
    Dentre os grandes, és o primeiro
    """
    vocabulary = gen_vocabulary('hino_tricolor.txt', hino_tricolor)
    entrada_dois = "Tu és forte, tu és grande Dentre os grandes és o primeiro"
    display(vocabulary)
    bag_of_words = gen_bag_of_words(vocabulary, entrada_dois)    
    display(bag_of_words)
