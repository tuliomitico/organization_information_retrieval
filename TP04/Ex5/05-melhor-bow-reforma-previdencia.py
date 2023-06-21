#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re

import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn import metrics
from sklearn.model_selection import cross_val_predict


# In[2]:


dataset = pd.read_csv(r'/home/tulio/Documents/UFU/ORI/reforma_previdencia_rotulado.csv',sep=';')


# In[3]:


dataset.describe()


# In[4]:


dataset.count()


# In[5]:


dataset["Classificação"].value_counts()


# In[6]:


tweets, sentimentos = dataset['Tweet'], dataset['Classificação']


# In[14]:


vectorizer = CountVectorizer(analyzer='word',ngram_range=(1,2))
freq_tweets = vectorizer.fit_transform(tweets)


# In[15]:


model = MultinomialNB()
model.fit(freq_tweets, sentimentos)


# In[16]:


testes = [
    "Reforma da Previdência é uma uma emenda constitucional", # Neutro
    "Essa reforma da previdência é uma coisa imunda", # Negativo
    "A reforma da previdência será modificada pra economizar menos", # Neutro
    "Só irei aposentar no caixão", # Negativo
    "Isso é regime de escravidão o que essa reforma propões" # Negativo
]
print(testes)


# In[17]:


freq_testes = vectorizer.transform(testes)


# In[18]:


model.predict(freq_testes)


# In[19]:


resultados = cross_val_predict(model, freq_tweets, sentimentos, cv=10)


# In[20]:


metrics.accuracy_score(sentimentos, resultados)


# In[21]:


print(metrics.classification_report(sentimentos, resultados))


# In[23]:


confusion_matrix = metrics.plot_confusion_matrix(model,X=freq_tweets,y_true=sentimentos,values_format='.4g')


# ## Resultado
# 
# Pendente.
