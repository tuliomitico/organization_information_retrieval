#!/usr/bin/env python
# coding: utf-8

# # TFIDF e 1-Grama

# ## Importação das libs necessárias

# In[13]:


import re

import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn import metrics
from sklearn.model_selection import cross_val_predict


# ## Carregando o arquivo

# In[14]:


dataset = pd.read_csv(r'/home/tulio/Documents//UFU/ORI/Tweets_Mg.csv')


# In[15]:


dataset.describe()


# In[16]:


dataset.count()


# In[17]:


dataset["Classificacao"].value_counts()


# In[18]:


tweets, sentimentos = dataset['Text'], dataset['Classificacao']


# ## Instanciando o modelo MultinomialNB

# In[19]:


vectorizer = TfidfVectorizer(analyzer='word',ngram_range=(1,1))
tfidf_tweets = vectorizer.fit_transform(tweets)


# In[21]:


model = MultinomialNB()
model.fit(tfidf_tweets, sentimentos)


# In[22]:


testes = ['O governo de Minas é uma tragédia, muito ruim','Estou muito feliz com o governo de Minas esse ano','O estado de Minas Gerais decretou calamidade financeira!!!','A segurança do estado está deixando a desejar','O governador de Minas é do Novo']
print(testes)


# In[23]:


tfidf_testes = vectorizer.transform(testes)


# In[24]:


model.predict(tfidf_testes)


# ## Validação cruzada

# In[27]:


resultados = cross_val_predict(model, tfidf_tweets, sentimentos, cv=10)


# In[28]:


metrics.accuracy_score(sentimentos,resultados)


# In[29]:


print(metrics.classification_report(sentimentos,resultados))


# In[36]:


confusion_matrix = metrics.plot_confusion_matrix(model,X=tfidf_tweets,y_true=sentimentos,values_format='.4g')


# # Resultados
# 
# Utilizando o modelo de vetorização do TF-IDF e unigrama teve resultado ligeiramente menor que o modelo binário, como podemos ver pela acurácia, entretanto obteve uma revogação (*recall*) melhor que o obtido pelo algoritmo anterior. 
