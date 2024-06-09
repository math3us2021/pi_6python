# Importando os pacotes a serem utilizados
import pandas as pd
import numpy as np

# importando o arquivo com a base de Filmes
filmes = pd.read_csv('./base/movies_metadata.csv', sep=',')


# importando o arquivo com a base de Avaliações
avaliacoes = pd.read_csv('./base/ratings.csv', sep=',')

#Avaliando as primeiras linhas
filmes.head()

#Avaliando as primeiras linhas
avaliacoes.head()

filmes.info()


# Obtendo os nomes das colunas
nomes_colunas = filmes.columns

# Convertendo para uma lista, se desejado
nomes_colunas_lista = list(nomes_colunas)

# Exibindo os nomes das colunas
print(nomes_colunas)
print(nomes_colunas_lista)


# Filtrando somente as colunas necessários e renomeando nome das variaveis

# Seleciona somente as variaveis que iremos utilizar
filmes = filmes [['budget','genres','id','original_title','original_language','release_date','vote_average','vote_count']]

# Renomeia as variaveis
filmes.rename(columns =
 {
        'budget': 'orcamento',
        'genres': 'generos',
        'id': 'id_filme',
        'original_language': 'linguagem',
        'original_title': 'titulo',
        'release_date': 'data_lancamento',
        'vote_average': 'media_votos',
        'vote_count': 'total_votos'}, inplace = True)

# Exibe as primeiras linhas do arquivo tratado
filmes.head()

avaliacoes.info()

# Obtendo os nomes das colunas
nomes_colunas = avaliacoes.columns

# Convertendo para uma lista, se desejado
nomes_colunas_lista = list(nomes_colunas)

# Exibindo os nomes das colunas
print(nomes_colunas)
print(nomes_colunas_lista)

# Filtrando somente as colunas necessários e renomeando nome das variaveis

# Seleciona somente as variaveis que iremos utilizar
avaliacoes = avaliacoes [['userId', 'movieId', 'rating']]

# Renomeia as variaveis
avaliacoes.rename(columns =
 {
        'userId': 'id_usuario',
        'movieId': 'id_filme',
        'rating': 'avaliacao'}, inplace = True)

# Exibe as primeiras linhas do arquivo tratado
avaliacoes.head()

avaliacoes.info()

# Converter a variavel id_filme em inteiro

# Verificando valores não numéricos na coluna 'id_filme'
non_numeric = filmes[~filmes['id_filme'].apply(lambda x: x.isdigit())]
print("Valores não numéricos encontrados na coluna 'id_filme':")
print(non_numeric)

# Removendo valores não numéricos ou substituindo-os por NaN
filmes['id_filme'] = pd.to_numeric(filmes['id_filme'], errors='coerce')

# Removendo linhas com NaN na coluna 'id_filme' (se preferir, você pode substituir os NaN por um valor específico)
filmes.dropna(subset=['id_filme'], inplace=True)

# Convertendo a coluna 'id_filme' para o tipo inteiro
filmes['id_filme'] = filmes['id_filme'].astype(int)

# Verificando se a conversão foi bem-sucedida
print(filmes.dtypes)

filmes.info()

# Converter a coluna 'data_lancamento' para o tipo datetime
filmes['data_lancamento'] = pd.to_datetime(filmes['data_lancamento'], errors='coerce')

# Verificando se a conversão foi bem-sucedida
print(filmes.dtypes)
print(filmes['data_lancamento'].head())


# Verificando se há valores nulos
filmes.isna().sum()

# Removendo as linhas de algumas colunas que entendo ser importante remover e que não irá afetar pois são poucos
filmes.dropna(subset=['linguagem', 'data_lancamento', 'media_votos', 'total_votos' ], inplace=True)

# Verificando se há valores nulos após a remoção
filmes.isna().sum()

# Verificando se há valores nulos
avaliacoes.isna().sum()

avaliacoes.info()

# Concatenando os dataframes
avaliacoes_e_filmes = avaliacoes.merge(filmes, on = 'id_filme')
avaliacoes_e_filmes.head(5)

#Verificando o novo data frame
avaliacoes_e_filmes.info()


avaliacoes_e_filmes.describe()

# Verificando a quantidade de avaliacoes por usuarios
avaliacoes_e_filmes['id_usuario'].value_counts()

import matplotlib.pyplot as plt

# Contagem de avaliações por usuário
user_counts = avaliacoes_e_filmes['id_usuario'].value_counts()

# Estatísticas descritivas
user_counts_descriptive = user_counts.describe()

print(user_counts_descriptive)

# Distribuição - Histograma
user_counts.hist(bins=50)
plt.xlabel('Número de Avaliações')
plt.ylabel('Número de Usuários')
plt.title('Distribuição do Número de Avaliações por Usuário')
plt.show()


# Contagem de avaliações por usuário
user_counts = avaliacoes_e_filmes['id_usuario'].value_counts()

# Estatísticas descritivas
user_counts_descriptive = user_counts.describe()
print("Estatísticas Descritivas:\n", user_counts_descriptive)

# Calcular percentis
percentiles = user_counts.quantile([0.75, 0.90, 0.95])
print("Percentis:\n", percentiles)

# Distribuição - Histograma
user_counts.hist(bins=50)
plt.xlabel('Número de Avaliações')
plt.ylabel('Número de Usuários')
plt.title('Distribuição do Número de Avaliações por Usuário')
plt.show()

# Verificando a coluna de média dos votos
avaliacoes_e_filmes['media_votos'].describe()

# Verificando a coluna total dos votos
avaliacoes_e_filmes['total_votos'].describe()

# Define o número mínimo de votos
m = avaliacoes_e_filmes['total_votos'].quantile(0.80)

# Calcula a média geral de votos de todos os filmes
C = avaliacoes_e_filmes['media_votos'].mean()

# Função para calcular a Classificação Ponderada
def weighted_rating(x, m=m, C=C):
    v = x['total_votos']
    R = x['media_votos']
    return (v / (v + m) * R) + (m / (v + m) * C)

# Aplica a função ao DataFrame
avaliacoes_e_filmes['avaliacao_ponderada'] = avaliacoes_e_filmes.apply(weighted_rating, axis=1)

# Ordena os filmes pela Classificação Ponderada
avaliacoes_e_filmes = avaliacoes_e_filmes.sort_values('avaliacao_ponderada', ascending=False)

avaliacoes_e_filmes.head()


# Filtrando as linhas com 'avaliacao_ponderada' acima de 6.5
avaliacoes_e_filmes2 = avaliacoes_e_filmes[avaliacoes_e_filmes['avaliacao_ponderada'] > 6.5]

# Exibindo o DataFrame filtrado
print(avaliacoes_e_filmes2)

avaliacoes_e_filmes2.head()

# Vamos descartar os valores duplicados, para que não tenha problemas de termos o mesmo usuário avaliando o mesmo filme
# diversas vezes
avaliacoes_e_filmes2.drop_duplicates(['id_usuario','id_filme'], inplace = True)

# Visualizando se houve alteração na quantidade de registros
avaliacoes_e_filmes2.shape

# Agora precisamos fazer um PIVOT. O que queremos é que cada ID_USUARIO seja uma variavel com o respectivo valor de nota
# para cada filme avaliado
filmes_pivot = avaliacoes_e_filmes2.pivot_table(columns = 'id_usuario', index = 'id_filme', values = 'avaliacao_ponderada')


# Avaliar o arquivo transformado para PIVOT
filmes_pivot.head(10)


# Os valores que são nulos iremos preencher com ZERO
filmes_pivot.fillna(0, inplace = True)
filmes_pivot.head()

# Vamos importar o csr_matrix do pacote SciPy
# Esse método possibilita criarmos uma matriz sparsa
from scipy.sparse import csr_matrix


# Vamos transformar o nosso dataset em uma matriz sparsa
filmes_sparse = csr_matrix(filmes_pivot)

# Tipo do objeto
type(filmes_sparse)

# Vamos importar o algoritmo KNN do SciKit Learn
from sklearn.neighbors import NearestNeighbors

# Criando e treinando o modelo preditivo
modelo = NearestNeighbors(algorithm = 'brute')
modelo.fit(filmes_sparse)

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import pickle

# Supondo que filmes_pivot é o seu DataFrame de entrada
# filmes_pivot = ...

# Transformar o DataFrame em uma matriz esparsa
filmes_sparse = csr_matrix(filmes_pivot.values)

# Treinar o modelo KNN
modelo = NearestNeighbors(algorithm='brute')
modelo.fit(filmes_sparse)

# Salvar a matriz esparsa
with open('filmes_sparse.pkl', 'wb') as f:
    pickle.dump(filmes_sparse, f)

# Salvar o modelo KNN
with open('modelo_knn.pkl', 'wb') as f:
    pickle.dump(modelo, f)



