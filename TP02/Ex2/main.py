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

import numpy as np
from unidecode import unidecode
from IPython.display import display

from Ex1.main import read_files 

def tf_idf(read_vocab, file_path) -> t.NoReturn:
    concat_phrases = []
    interim_array = []
    norm_vocab = []
    tf = []
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
    
    collection = len(interim_array)
    
    # Array de dicts para calculo do TF
    for file in interim_array:
        tf_dict = {}
        for word in norm_vocab:
            tf_dict[word] = 0
        for w in file:
            if w in norm_vocab:
                tf_dict[w] += 1
        for i in norm_vocab:
            tf_dict[i] = (1 + np.log2(tf_dict[i])) if tf_dict[i] != 0 else 0
        tf.append(tf_dict)
    
    # Array de dicts para calculo do IDF
    idf_dict = {}
    for w in norm_vocab:
        count = 0
        for file in interim_array:
            if w in file:
                count += 1
        idf_dict[w] = np.log2(collection / count)
        
    tf_idf = []
    for doc in tf:
        doc_tfidf = {}
        for word, freq in doc.items():
            doc_tfidf[word] = freq * idf_dict[word]
        tf_idf.append(doc_tfidf)

    display(tf_idf)
            
if __name__=="__main__":
    f = open('../Ex1/vocabulario.txt')
    file = f.readlines()
    tf_idf(file,'../Ex1/data')
    f.close()
    sys.path.pop()