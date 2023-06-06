#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 09:41:46 2023

@author: tulio
"""
import sys
import time
import math
from pathlib import Path
from IPython.display import display

BASE_DIR_MOD = Path(__file__).resolve().parent

sys.path.append(str(BASE_DIR_MOD))

import pandas as pd
from utils import (
    remove_diacritics,
    read_files,
    tf_idf,
    idf_query,
    tf_query,
    tf_idf_query
)

BASE_DIR = Path(__file__).resolve().parent


def calcular_similaridade(vocabulario, docs_dir, query):

    query = remove_diacritics(query)
    
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
    
    # print("Similaridade com o seguinte termo de consulta: ",'"',*query,'"')
    
    # display(pd.DataFrame((sorted(sim_dict.items(),key= lambda item: item[1],reverse=True))))
    
    # display(sorted(sim_dict.items(),key= lambda item: item[1],reverse=True))
    
    
    return sorted(sim_dict.items(),key=lambda item: item[1],reverse=True)

    
    
    


if __name__ == "__main__":
    vocab_file = open(BASE_DIR /  "vocabulario.txt")
    vocab = remove_diacritics(vocab_file.readlines())
    docs_collection = read_files("../Ex4/data")
    querys = [["love","is","in","the","air"],["take","the","sky"],["make","me","happy"],["sad","song","about","you"],["mind","head","away"]]
    result_query = []
    
    for query in querys:
        start = time.perf_counter()
        results = calcular_similaridade(vocab,docs_dir=docs_collection,query = query)
        end = time.perf_counter() - start
        print(f"Tempo de execução da consulta {query}: ",end)
        # print("Proxima consulta")
        # print(results)
        result_query.extend([("Nome da consulta",' '.join(query)),*results])
    
    dt = pd.DataFrame(result_query)
    dt.to_excel('top10_com_stem.xls')
    