import MLP as multilp
import readers as r
import numpy as np



# Definição dos dados de entrada (X) e saída (y)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]).T  # Transposta para ter dimensões [n_entradas, n_amostras]

y = np.array([
    [0],
    [1],
    [1],
    [0]
]).T  # Transposta para ter dimensões [n_saidas, n_amostras]

reader = r.Reader()
X = reader.caracteres_reader("src\X.csv")
Y = reader.y_npy('caracteres-completo\Y_classe.npy')

# Defina o tamanho das camadas de entrada, ocultas e de saída
input_size = 2
hidden_sizes = [3, 3]  # Duas camadas ocultas com 3 neurônios cada
output_size = 1

# Cria o MLP com os tamanhos definidos
mlp = multilp.MultilayerPerceptron(input_size, hidden_sizes, output_size, learning_rate=0.1)

# Treina o MLP em 10.000 épocas
mlp.train(X, y, epochs=10000)

# Faz previsões com o MLP treinado
predictions = mlp.predict(X)

# Imprime as previsões
print("Previsões:")
print(predictions)

# Compare as previsões com os valores reais
print("\nValores reais:")
print(y)