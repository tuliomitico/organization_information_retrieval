#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 11:22:24 2023

@author: tulio
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent 

sys.path.append(str(BASE_DIR))

import typing as t
import string

from unidecode import unidecode
from IPython.display import display

from Ex1.main import read_files 

def tf_idf(read_vocab, file_path) -> t.NoReturn:
    concat_phrases = []
    interim_array = []
    norm_vocab = []
    tf = {}
    files = read_files(file_path)
    
    # Achata o arquivo de vocabulario para vetor
    for word in read_vocab:
        norm_vocab.append(word.replace('\n',''))
    
    for file in files:
        concat_phrases.append([''.join(file)])
                
    for file in concat_phrases:
        for sentence in file:
            aux = sentence.replace('\n',' ').split(' ')
            interim_array.append([unidecode(i).lower().translate(str.maketrans('','',string.punctuation)) for i in aux])
            
    file = interim_array[0]
    
    for word in norm_vocab:
        tf[word] = 0
    for w in file:
        print(w)
        if w in norm_vocab:
            tf[w] += 1
    display(tf)
            
if __name__=="__main__":
    f = open('../Ex1/vocabulario.txt')
    file = f.readlines()
    tf_idf(file,'../Ex1/data')
    f.close()
    sys.path.pop()