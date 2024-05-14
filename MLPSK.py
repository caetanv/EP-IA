
import numpy as np
from sklearn.neural_network import MLPClassifier
# Dados de treinamento - vetores de pixels para cada letra do alfabeto (representação simplificada)
# Vamos criar vetores para as letras de 'A' a 'Z'

# Vetores de pixels para cada letra (exemplos simplificados)
vetores_letras = {
    'A': [1, 1, 1, -1, -1, -1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1, -1,
          1, 1, 1, -1, -1, -1, 1, -1, 1, -1, -1, -1],
    'B': [1, 1, 1, 1, -1, -1, 1, -1, 1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, -1,
          1, -1, -1, -1, 1, -1, 1, 1, 1, 1, -1, -1],
    'C': [1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1, -1,
          1, -1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1],
    'D': [1, 1, 1, 1, -1, -1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1, -1,
          1, -1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1],
    'E': [1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, -1,
          1, -1, -1, -1, 1, -1, -1, -1, 1, 1, -1, -1],
    # Adicione mais letras aqui conforme necessário
    'Z': [1, 1, 1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1, -1,
          1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1]
}

# Criando os arrays de features (X) e rótulos (y) a partir dos vetores de pixels e letras
X = []
y = []

for letra, vetor in vetores_letras.items():
    X.append(vetor)
    y.append(letra)

X = np.array(X)
y = np.array(y)

# Criando e treinando o modelo MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10000, alpha=1e-4,
                    solver='sgd', verbose=10, random_state=1,
                    learning_rate_init=0.1)

mlp.fit(X, y)

# Fazendo previsões para todas as letras do alfabeto
letras_alfabeto = 'ABCDEZ'
for letra in letras_alfabeto:
    vetor_letra = vetores_letras[letra]
    previsao = mlp.predict([vetor_letra])
    letra_prevista = previsao[0]
    print(f"Letra: {letra} - Letra prevista: {letra_prevista}")