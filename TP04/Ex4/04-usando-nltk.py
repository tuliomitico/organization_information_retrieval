#!/usr/bin/env python
# coding: utf-8

# # Exercício 4
# 
# >4) Remova as stopwords dos tweets usando a biblioteca NLTK (Dica: reuse o código do TP3). Refaça o experimento usando a melhor configuração de BoW e algoritmo de classificação. Discuta os resultados encontrados.

# In[25]:


import re
import string

import nltk
import pandas as pd
from unidecode import unidecode
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn import metrics
from sklearn.model_selection import cross_val_predict


# In[2]:


dataset = pd.read_csv(r'/home/tulio/Documents/UFU/ORI/Tweets_Mg.csv')


# In[3]:


dataset.describe()


# In[4]:


dataset.count()


# In[5]:


dataset['Classificacao'].value_counts()


# In[6]:


tweets, sentimentos = dataset['Text'], dataset['Classificacao']


# In[28]:


def remover_stopwords(text):
    stopwords = nltk.corpus.stopwords.words('portuguese')
    words = text.split()
    formatted_words = [unidecode(word).lower().translate(str.maketrans('', '', string.punctuation + "\n")) for word in words]
    words_wo_stopwords = [word for word in formatted_words if word.lower() not in stopwords]
    return ' '.join(words_wo_stopwords)


# In[29]:


remover_stopwords('É um beluga, coisa imunda!')


# In[30]:


tweets_sem_stopwords = tweets.apply(remover_stopwords)


# In[31]:


vectorizer = CountVectorizer(analyzer='word', ngram_range=(1,2))
freq_tweets = vectorizer.fit_transform(tweets_sem_stopwords)


# In[32]:


model = MultinomialNB()
model.fit(freq_tweets, sentimentos)


# In[34]:


testes = ['O governo de Minas é uma tragédia, muito ruim','Estou muito feliz com o governo de Minas esse ano','O estado de Minas Gerais decretou calamidade financeira!!!','A segurança do estado está deixando a desejar','O governador de Minas é do Novo']
print(testes)


# In[35]:


freq_testes = vectorizer.transform(testes)


# In[36]:


model.predict(freq_testes)


# In[38]:


resultados = cross_val_predict(model,freq_tweets,sentimentos,cv=10)


# In[40]:


metrics.accuracy_score(sentimentos, resultados)


# In[41]:


print(metrics.classification_report(sentimentos, resultados))


# In[43]:


confusion_matrix = metrics.plot_confusion_matrix(model,X=freq_tweets,y_true=sentimentos,values_format='.4g')


# In[ ]:




