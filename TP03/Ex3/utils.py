import string
from pathlib import Path
import typing as t

from unidecode import unidecode
from chardet.universaldetector import UniversalDetector
import nltk
import numpy as np

def remove_diacritics(words, language="portuguese"):
    nltk.download('stopwords')
    stopwords = nltk.corpus.stopwords.words(language)
    list_stopwords = set(stopwords)
    formatted_words = [unidecode(word).lower().translate(str.maketrans('', '', string.punctuation + "\n")) for word in words]
    word_wo_stopwords = [word for word in formatted_words if word not in list_stopwords]
    return word_wo_stopwords 

def read_files(file_path):
    base_dir = Path(__file__).resolve().parent
    # print(base_dir)
    detector = UniversalDetector()
    texts = {}
    if (base_dir / file_path).exists():
        path = Path(file_path).glob('*.txt')
        for file_name in path:
            # print(file_name.name)
            detector.reset()
            for line in open(file_name,'rb'):
                detector.feed(line)
                if detector.done:
                    break
            detector.close()    
            file = open(file_name,"r",encoding=detector.result['encoding'])
            texts[file_name.name] = file.readlines()
            file.close()
    return texts

def tf_query(vocab,query):
    tf_dict = {}
    for word in vocab:
        tf_dict[word] = 0
    for w in query:
        if w in vocab:
            tf_dict[w] += 1
    return tf_dict

def idf_query(vocab,docs):
    concat_phrases = {}
    interim_dict = {}
    for name,file in docs.items():
        concat_phrases[name] = [''.join(file)]
                
    for name,file in concat_phrases.items():
        for sentence in file:
            aux = sentence.replace('\n',' ').split(' ')
            interim_dict[name] = [unidecode(i).lower().translate(str.maketrans('','',string.punctuation)) for i in aux]
    idf_dict = {}
    n = len(docs.values())
    for w in vocab:
        count = 0
        for doc in interim_dict.values():
            if w in doc:
                count += 1
        idf_dict[w] = np.log2(n/count)
    return idf_dict

def tf_idf_query(tf,idf):
    tf_idf = {}
    for w in tf:
        tf_idf[w] = tf[w] * idf[w]
    return tf_idf

def tf_idf(vocab, files) -> t.NoReturn:
    concat_phrases = {}
    interim_array = {}
    tf = {}
    
    # Achata o arquivo de vocabulario para vetor
    
    for name,file in files.items():
        concat_phrases[name] = [''.join(file)]
                
    for name,file in concat_phrases.items():
        for sentence in file:
            aux = remove_diacritics(sentence.replace('\n',' ').split(' '))
            interim_array[name] = aux
    
    len_collection = len(interim_array.values())
    
    # Array de dicts para calculo do TF
    for name,file in interim_array.items():
        tf_dict = {}
        for word in vocab:
            tf_dict[word] = 0
        for w in file:
            if w in vocab:
                tf_dict[w] += 1
        for i in vocab:
            tf_dict[i] = (1 + np.log2(tf_dict[i])) if tf_dict[i] != 0 else 0
        tf[name] = tf_dict

    # Array de dicts para calculo do IDF
    idf_dict = {}
    for w in vocab:
        count = 0
        for name,file in interim_array.items():
            if w in file:
                count += 1
        idf_dict[w] = np.log2((len_collection / count))
    
    tf_idf = {}
    for name,doc in tf.items():
        doc_tfidf = {}
        for word, freq in doc.items():
            doc_tfidf[word] = freq * idf_dict[word]
        tf_idf[name] = doc_tfidf
 
    return tf_idf
