# TP3

O que foi realizado no arquivo jupyter notebook foi o uso da ferramenta NLTK
para tratamento de textos do Machado de Assis, grande escrito brasileiro do 
século 19, de forma que foram feitos pré-processamentos no romace de
Dom Casmurro, removendo _stopwords_, pontuação, acentuação, etc. Desta forma
foi possível formar um vocabulário só com termos que tenham maior peso para
retorno de documentos de uma coleção mais relevante para a consulta de um
usuário, por exemplo. Foram também utilizados *stemming* para que possa remover os afixos das palavras que geralmente formam um verbo ou até mesmo
um substantivo, como qualificar (verbo no infinitivo) e qualidade (substantivo
derivado da ação de qualificar) ficaria como quali após o processamento desta técnica, então identificar variações deste termo facilmente através desta técnica, embora que não seja fácil em empregar em todos os idiomas.

A lemmatização consiste de trazer a forma "raíz" a palavra a ser processada, essa técnica de fato trás uma palavra existente no dicionário daquele idioma em compensação, é um processamento mais demorad que exige maior poder computacional.

A eliminação de stopwords ajuda na recuperação de documentos mais concisos para o usuário de forma a remover palavras que não trazem "significado a pesquisa". Geralmente artigos, pronomes, preposições, conjunções e outros tipos de termos gramaticais são desconsiderados para serem mantidos na consulta original feita pelo usuário.
