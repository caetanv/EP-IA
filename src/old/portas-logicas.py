import MLP as multilp
import numpy as np

# Definição dos dados de entrada (X) e saída (y) para cada problema lógico

# Dados para a porta lógica AND
X_and = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]).T  # Transposta para ter dimensões [n_entradas, n_amostras]

y_and = np.array([
    [0],
    [0],
    [0],
    [1]
]).T  # Transposta para ter dimensões [n_saidas, n_amostras]

# Dados para a porta lógica OR
X_or = X_and  # Usa o mesmo conjunto de entradas que o problema AND

y_or = np.array([
    [0],
    [1],
    [1],
    [1]
]).T  # Transposta para ter dimensões [n_saidas, n_amostras]

# Dados para a porta lógica XOR
X_xor = X_and  # Usa o mesmo conjunto de entradas que o problema AND

y_xor = np.array([
    [0],
    [1],
    [1],
    [0]
]).T  # Transposta para ter dimensões [n_saidas, n_amostras]

# Definição das camadas da rede neural
input_size = 2  # Duas entradas
hidden_sizes = [3]  # Uma camada oculta com 3 neurônios
output_size = 1  # Uma saída

# Taxa de aprendizado
learning_rate = 0.1

# Número de épocas para treinamento
epochs = 10000

# Função para treinar e testar o MLP com um problema lógico
def train_and_test_mlp(X, y, problem_name):
    # Cria uma instância do Multilayer Perceptron
    mlp = multilp.MultilayerPerceptron(input_size, hidden_sizes, output_size, learning_rate)

    # Treina o MLP nos dados
    mlp.train(X, y, epochs)

    # Faz previsões com o MLP treinado
    predictions = mlp.predict(X)

    # Imprime os resultados
    print(f"\nProblema lógico: {problem_name}")
    print("Previsões:")
    print(predictions)
    print("\nValores reais:")
    print(y)

# Testa o MLP com os problemas lógicos AND, OR e XOR
train_and_test_mlp(X_and, y_and, "AND")
train_and_test_mlp(X_or, y_or, "OR")
train_and_test_mlp(X_xor, y_xor, "XOR")