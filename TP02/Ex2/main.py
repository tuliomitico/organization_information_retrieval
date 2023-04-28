#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 11:22:24 2023

@author: tulio
"""
import sys 
sys.path.append('/home/tulio/Documents/UFU/ORI/organization_information_retrieval/TP02')
print(sys.path)
import typing as t
import string

from unidecode import unidecode
from IPython.display import display

from Ex1.main import read_files 


def tf_idf(read_vocab, file_path) -> t.NoReturn:
    files = read_files(file_path)
    norm_vocab = []
    
    # Achata o arquivo de vocabulario para vetor
    for word in read_vocab:
        norm_vocab.append(word.replace('\n',''))
        
    tf = {}
    
    display(norm_vocab)
    
    interim_array = []
    
    concat_phrases = []
    for file in files:
        concat_phrases.append([''.join(file)])
                
    for file in concat_phrases:
        for sentence in file:
            aux = sentence.replace('\n',' ').split(' ')
            interim_array.append([unidecode(i).lower().translate(str.maketrans('','',string.punctuation)) for i in aux])
            
    file = interim_array[0]
    
    for word in norm_vocab:
        tf[word] = 0
    for i in file:
        print(i)
        if i in norm_vocab:
            tf[word] += 1
            break
    display(tf)
            
            
        
    
if __name__=="__main__":
    f = open('../Ex1/vocabulario.txt')
    file = f.readlines()
    tf_idf(file,'../Ex1/data')
    f.close()
    sys.path.pop()