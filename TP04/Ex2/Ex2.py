
import nltk
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import cross_val_predict



dataset = pd.read_csv(r"C:\Users\gustavoborges\Arquivos Local\Pessoal\ufu\ORI\TP-4\Tweets_Mg.csv",encoding='utf-8')

dataset.head()

dataset.count()



dataset[dataset.Classificacao=='Neutro'].count()

dataset[dataset.Classificacao=='Positivo'].count()

dataset[dataset.Classificacao=='Negativo'].count()



tweets = dataset['Text'].values
classes = dataset['Classificacao'].values

vectorizer = CountVectorizer(analyzer="word")
freq_tweets = vectorizer.fit_transform(tweets)

modelo = MultinomialNB()
modelo.fit(freq_tweets,classes)


testes = ['O governo de Minas é uma tragédia, muito ruim','Estou muito feliz com o governo de Minas esse ano','O estado de Minas Gerais decretou calamidade financeira!!!','A segurança do estado está deixando a desejar','O governador de Minas é do Novo']
print(testes)



freq_testes = vectorizer.transform(testes)



modelo.predict(freq_testes)


resultados = cross_val_predict(modelo, freq_tweets, classes, cv=10)



metrics.accuracy_score(classes,resultados)


print(metrics.classification_report(classes,resultados))

print (pd.crosstab(classes, resultados, rownames=['Real'], colnames=['Predito'], margins=True))



print('\n<><><><><><><> ngram_range(1,2) <><><><><><><>\n')

vectorizer = CountVectorizer(ngram_range=(1,2))
freq_tweets = vectorizer.fit_transform(tweets)
modelo = MultinomialNB()
modelo.fit(freq_tweets,classes)
resultados = cross_val_predict(modelo, freq_tweets, classes, cv=10)

metrics.accuracy_score(classes,resultados)

print(metrics.classification_report(classes,resultados))

print (pd.crosstab(classes, resultados, rownames=['Real'], colnames=['Predito'], margins=True))

print('\n<><><><><><><> ngram_range(2,2) <><><><><><><>\n')

vectorizer = CountVectorizer(ngram_range=(2,2))
freq_tweets = vectorizer.fit_transform(tweets)
modelo = MultinomialNB()
modelo.fit(freq_tweets,classes)
resultados = cross_val_predict(modelo, freq_tweets, classes, cv=10)

metrics.accuracy_score(classes,resultados)

print(metrics.classification_report(classes,resultados))

print (pd.crosstab(classes, resultados, rownames=['Real'], colnames=['Predito'], margins=True))



