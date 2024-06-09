from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Carregar os modelos salvos
with open('filmes_sparse.pkl', 'rb') as f:
    filmes_sparse = pickle.load(f)

with open('modelo_knn.pkl', 'rb') as f:
    modelo_knn = pickle.load(f)

# Carregar os índices dos filmes
# filmes_indices = list(filmes_sparse.indices)
filmes_indices = list(range(filmes_sparse.shape[0]))  # Correção para obter o índice correto do DataFrame original



@app.route('/indices', methods=['GET'])
def get_indices():
    # Converter todos os índices para int
    int_filmes_indices = [int(idx) for idx in filmes_indices]
    return jsonify(int_filmes_indices)



@app.route('/predict', methods=['POST'])

# def predict():
#     data = request.json
#     filme_id = data.get('id')
#     print(filme_id)
#     return jsonify(filmes_indices), 200

def predict():
    data = request.json
    filme_id = data.get('id')
    print(filme_id)

    # Verificar se o filme_id está nos índices
    if filme_id not in filmes_indices:
        return jsonify({"error": "Filme não encontrado"}), 404

    # Encontrar a posição do filme na matriz esparsa
    filme_idx = filmes_indices.index(filme_id)

    # Obter as recomendações
    distances, indices = modelo_knn.kneighbors(filmes_sparse[filme_idx], n_neighbors=10)

    # Obter os IDs dos filmes recomendados
    recommended_filme_ids = [int(filmes_indices[i]) for i in indices.flatten()]  # Conversão para int

    return jsonify({"recommended_ids": recommended_filme_ids})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
