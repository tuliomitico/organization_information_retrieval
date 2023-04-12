# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 19:19:46 2023

@author: Túlio
"""

from pprint import pprint
from pathlib import Path
import string

from chardet.universaldetector import UniversalDetector
from unidecode import unidecode

# Parte 1

def read_files(path_name: str) -> "list[str]":
    base_dir = Path(__file__).resolve().parent
    detector = UniversalDetector()
    texts = []
    if (base_dir / path_name).exists():
        path = Path(path_name).glob('*.txt')
        for file_name in path:
            # print(str(file_name).ljust(60),end='')
            detector.reset()
            for line in open(file_name,'rb'):
                detector.feed(line)
                if detector.done:
                    break
            detector.close()    
            file = open(file_name,"r",encoding=detector.result['encoding'])
            texts.append(file.readlines())
        file.close()
    return texts
    

# Parte 2

def gen_vocabulary(files: "list[str]"):
    interim_array = []    
    
    for sentence in files[0]:
        aux = sentence.replace('\n',' ').split(' ')
        interim_array.extend([unidecode(i).lower().translate(str.maketrans('','',string.punctuation)) for i in aux])
        
    final_vector = sorted(set(interim_array))
    
    del final_vector[0]
    
    return final_vector

# Parte 3

def create_vocab_file(vocabulary: "list[str]") -> None:
    with open('vocabulario.txt','w',encoding='utf-8') as file:
        for word in vocabulary:
            file.write(word + '\n')
            
# Parte 4

def gen_bag_of_words(vocab_filename: 'str'):            
    path = Path(__file__).resolve().parent
    vocab_file = open(path / vocab_filename,'r',encoding='utf-8')
    
    vocab = vocab_file.readlines()
    norm_vocab = []
    
    for word in vocab:
        norm_vocab.append(word.replace('\n',''))
    # pprint(norm_vocab)
    
    files = read_files('data')
    interim_array = []
    for sentence in files[0]:
        aux = sentence.replace('\n',' ').split(' ')
        interim_array.extend([unidecode(i).lower().translate(str.maketrans('','',string.punctuation)) for i in aux])
        
    bag_of_words= []
    temp_dict = {}
    
    for word in norm_vocab:
        for i in interim_array:
            if word == i:
                temp_dict[word] = 1
                break
            else:
                temp_dict[word] = 0
                
    # pprint(temp_dict)
        
    
if __name__ == "__main__":
    texts = read_files('data')
    vocabulary = gen_vocabulary(texts)
    # pprint(vocabulary[0:49])
    # print('Tamanho da lista',len(vocabulary))
    # create_vocab_file(vocabulary)
    gen_bag_of_words('vocabulario.txt')