#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 12:18:24 2023

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

from Ex1.main import ( read_files, gen_vocabulary, create_vocab_file )
from Ex2.main import tf_idf

if __name__ == "__main__":
    texts = read_files('data')
    vocab = gen_vocabulary(texts)
    # create_vocab_file(vocab)
    f = open('vocabulario.txt')
    file = f.readlines()
    tf_idf(file,"data")
    # print(texts)
    # print(len(vocab))
    f.close()
    sys.path.pop()