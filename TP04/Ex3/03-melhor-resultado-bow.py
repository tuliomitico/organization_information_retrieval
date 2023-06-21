#!/usr/bin/env python
# coding: utf-8

# # Exercício 3
# 
# > 3) Selecione o melhor resultado encontrado até agora com relação a BoW (TF, TF+bigrama, TF-IDF e TF-IDF+bigrama). Escolha outros três algoritmos de classificação, refaça o experimento e compare os resultados. Preste atenção nas diferenças entre cada uma das classes para cada um dos algoritmos escolhidos.

# In[17]:


import re

import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn import metrics
from sklearn.model_selection import cross_val_predict


# In[18]:


dataset = pd.read_csv(r'/home/tulio/Documents/UFU/ORI/Tweets_Mg.csv')


# In[19]:


dataset.describe()


# In[20]:


dataset.count()


# In[21]:


dataset['Classificacao'].value_counts()


# In[22]:


tweets, sentimentos = dataset['Text'], dataset['Classificacao']


# In[23]:


vectorizer = CountVectorizer(analyzer='word', ngram_range=(1,2))
freq_tweets = vectorizer.fit_transform(tweets)


# In[24]:


rf_model = RandomForestClassifier(random_state=2023)
dt_model = DecisionTreeClassifier(max_depth=3, random_state=2023)
knn_model = KNeighborsClassifier(n_neighbors=10)
svc_model = SVC()
model = MultinomialNB()


# In[25]:


rf_model.fit(freq_tweets,sentimentos)


# In[26]:


dt_model.fit(freq_tweets,sentimentos)


# In[27]:


knn_model.fit(freq_tweets,sentimentos)


# In[28]:


svc_model.fit(freq_tweets,sentimentos)


# In[29]:


model.fit(freq_tweets,sentimentos)


# In[31]:


testes = ['O governo de Minas é uma tragédia, muito ruim','Estou muito feliz com o governo de Minas esse ano','O estado de Minas Gerais decretou calamidade financeira!!!','A segurança do estado está deixando a desejar','O governador de Minas é do Novo']
print(testes)


# In[32]:


freq_testes = vectorizer.transform(testes)


# In[33]:


rf_model.predict(freq_testes)


# In[34]:


dt_model.predict(freq_testes)


# In[35]:


knn_model.predict(freq_testes)


# In[36]:


svc_model.predict(freq_testes)


# In[37]:


model.predict(freq_testes)


# ## Validação Cruzada

# ### Random Forest

# In[39]:


rf_resultados = cross_val_predict(rf_model,freq_tweets,sentimentos,cv=10)


# In[40]:


metrics.accuracy_score(sentimentos,rf_resultados)


# In[42]:


print(metrics.classification_report(sentimentos, rf_resultados))


# In[43]:


confusion_matrix = metrics.plot_confusion_matrix(rf_model,X=freq_tweets,y_true=sentimentos,values_format='.4g')


# ## DecisionTree

# In[44]:


dt_resultados = cross_val_predict(dt_model,freq_tweets,sentimentos,cv=10)


# In[45]:


metrics.accuracy_score(sentimentos,dt_resultados)


# In[46]:


print(metrics.classification_report(sentimentos, dt_resultados))


# In[47]:


confusion_matrix = metrics.plot_confusion_matrix(dt_model,X=freq_tweets,y_true=sentimentos,values_format='.4g')


# ### K-Nearest Neighbors

# In[48]:


knn_resultados = cross_val_predict(knn_model,freq_tweets,sentimentos,cv=10)


# In[49]:


metrics.accuracy_score(sentimentos,knn_resultados)


# In[50]:


print(metrics.classification_report(sentimentos, knn_resultados))


# In[51]:


confusion_matrix = metrics.plot_confusion_matrix(knn_model,X=freq_tweets,y_true=sentimentos,values_format='.4g')


# ### SVC

# In[52]:


svc_resultados = cross_val_predict(svc_model,freq_tweets,sentimentos,cv=10)


# In[53]:


metrics.accuracy_score(sentimentos,svc_resultados)


# In[54]:


print(metrics.classification_report(sentimentos, svc_resultados))


# In[55]:


confusion_matrix = metrics.plot_confusion_matrix(svc_model,X=freq_tweets,y_true=sentimentos,values_format='.4g')


# ### MultinomialNB

# In[56]:


resultados = cross_val_predict(model,freq_tweets,sentimentos,cv=10)


# In[57]:


metrics.accuracy_score(sentimentos, resultados)


# In[58]:


print(metrics.classification_report(sentimentos, resultados))


# In[59]:


confusion_matrix = metrics.plot_confusion_matrix(model,X=freq_tweets,y_true=sentimentos,values_format='.4g')

