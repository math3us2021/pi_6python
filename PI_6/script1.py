# Importando os pacotes a serem utilizados
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import pickle

# Define o número mínimo de votos e calcula a média geral de votos
def initialize_params(df):
    m = df['total_votos'].quantile(0.80)
    C = df['media_votos'].mean()
    return m, C

# Função para calcular a Classificação Ponderada
def weighted_rating(x, m, C):
    v = x['total_votos']
    R = x['media_votos']
    return (v / (v + m) * R) + (m / (v + m) * C)

# Função para processar um chunk de dados
def process_chunk(chunk, filmes, m, C):
    chunk = chunk.merge(filmes, on='id_filme')
    chunk['avaliacao_ponderada'] = chunk.apply(lambda x: weighted_rating(x, m, C), axis=1)
    chunk = chunk[chunk['avaliacao_ponderada'] > 6.5]
    chunk.drop_duplicates(['id_usuario', 'id_filme'], inplace=True)
    return chunk

# Carregar a base de filmes e preprocessar
filmes = pd.read_csv('./base/movies_metadata.csv', sep=',', dtype={'id': str}, low_memory=False)
filmes = filmes[['budget', 'genres', 'id', 'original_title', 'original_language', 'release_date', 'vote_average', 'vote_count']]
filmes.rename(columns={
    'budget': 'orcamento',
    'genres': 'generos',
    'id': 'id_filme',
    'original_language': 'linguagem',
    'original_title': 'titulo',
    'release_date': 'data_lancamento',
    'vote_average': 'media_votos',
    'vote_count': 'total_votos'}, inplace=True)
filmes['id_filme'] = pd.to_numeric(filmes['id_filme'], errors='coerce')
filmes.dropna(subset=['id_filme'], inplace=True)
filmes['id_filme'] = filmes['id_filme'].astype(int)
filmes['data_lancamento'] = pd.to_datetime(filmes['data_lancamento'], errors='coerce')
filmes.dropna(subset=['linguagem', 'data_lancamento', 'media_votos', 'total_votos'], inplace=True)

# Inicializar parâmetros para o cálculo da classificação ponderada
m, C = initialize_params(filmes)

# Processar avaliações em chunks
chunk_size = 100000
avaliacoes_chunks = pd.read_csv('./base/ratings.csv', sep=',', chunksize=chunk_size)

chunks = []
for chunk in avaliacoes_chunks:
    chunk.rename(columns={
        'userId': 'id_usuario',
        'movieId': 'id_filme',
        'rating': 'avaliacao'}, inplace=True)
    chunk = chunk[['id_usuario', 'id_filme', 'avaliacao']]
    chunk = process_chunk(chunk, filmes, m, C)
    chunks.append(chunk)

avaliacoes_e_filmes2 = pd.concat(chunks)

# Fazer o PIVOT
filmes_pivot = avaliacoes_e_filmes2.pivot_table(columns='id_usuario', index='id_filme', values='avaliacao_ponderada')
filmes_pivot.fillna(0, inplace=True)

# Transformar em matriz esparsa
filmes_sparse = csr_matrix(filmes_pivot.values)

# Criar e treinar o modelo preditivo KNN
modelo = NearestNeighbors(algorithm='brute')
modelo.fit(filmes_sparse)

# Salvar os modelos
with open('filmes_sparse.pkl', 'wb') as f:
    pickle.dump(filmes_sparse, f)

with open('modelo_knn.pkl', 'wb') as f:
    pickle.dump(modelo, f)

print("Arquivos salvos com sucesso!")
