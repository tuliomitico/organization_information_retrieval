#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 09:41:46 2023

@author: tulio
"""
import sys
import math
from pathlib import Path
from pprint import pprint

BASE_DIR_MOD = Path(__file__).resolve().parent

sys.path.append(str(BASE_DIR_MOD))

print(BASE_DIR_MOD)

from utils import (
    remove_diacritics,
    read_files,
    tf_idf,
    idf_query,
    tf_query,
    tf_idf_query
)

BASE_DIR = Path(__file__).resolve().parent

# Create a function calculate_similarity that takes three arguments: vocabulario, docs_dir and query. 
# It should print the similarity between the query and the documents sorted by similarity. 
# The documents should be sorted in descending order by similarity.

def calcular_similaridade(vocabulario, docs_dir, query):
    
    query_tf = tf_query(vocabulario, query)
    query_idf = idf_query(vocabulario, docs_dir)
    
    query_tfidf = tf_idf_query(query_tf, query_idf)

    tfidf_values = tf_idf(vocabulario, docs_dir)
    
    sim_dict = {}
    for name,i in tfidf_values.items():
        dot_product = sum(x * y for x, y in zip(map(float,(query_tfidf.values())), i.values()))
        magnitude1 = math.sqrt(sum(x * x for x in query_tfidf.values()))
        magnitude2 = math.sqrt(sum(x * x for x in i.values()))
        similarity = dot_product / (magnitude1 * magnitude2)
        sim_dict[name] = similarity
    
    print("Similaridade com o seguinte termo de consulta: ",'"',*query,'"')
    pprint(sorted(sim_dict.items(),key= lambda item: item[1],reverse=True))
    return sorted(sim_dict.items(),key=lambda item: item[1],reverse=True)

    
    
    


if __name__ == "__main__":
    vocab_file = open(BASE_DIR /  "../Ex2/vocabulario.txt")
    vocab = remove_diacritics(vocab_file.readlines())
    docs_collection = read_files("../Ex2/data")
    calcular_similaridade(vocab,docs_dir=docs_collection,query = ["to" ,"do"])