import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import MLP as multilp

# Importa o arquivo CSV com os dados de entrada (X) e saída (y)
X = pd.read_csv('X.csv', header=None).values
y = pd.read_csv('Y_letra.csv', header=None).values

# Transforma os rótulos em codificação one-hot
label_binarizer = LabelBinarizer()
y = label_binarizer.fit_transform(y.flatten())


# Definição da estrutura da rede neural
input_size = 120  # 120 neurônios na camada de entrada (correspondente ao vetor de pixels)
hidden_sizes = [50, 50]  # Duas camadas ocultas com 50 neurônios cada
output_size = 26  # 26 neurônios na camada de saída (correspondente às letras do alfabeto)

# Taxa de aprendizado
learning_rate = 0.01

# Número de épocas para treinamento
epochs = 1000

# Cria uma instância do Multilayer Perceptron
mlp = multilp.MultilayerPerceptron(input_size, hidden_sizes, output_size, learning_rate)


# Tente treinar o MLP nos dados
try:
    mlp.train(X, y, epochs)
except Exception as e:
    print(f"Ocorreu um erro durante o treinamento do MLP: {e}")


# Função para converter previsões para letras
def predictions_to_letters(predictions):
    # Converte as previsões em índices
    predicted_indices = np.argmax(predictions, axis=0)
    # Converte os índices para letras
    predicted_letters = label_binarizer.inverse_transform(predicted_indices)
    return predicted_letters

# Faz previsões com o MLP treinado
predictions = mlp.predict(X)

# Converte as previsões para letras
predicted_letters = predictions_to_letters(predictions)

# Imprime algumas previsões
for i in range(5):  # Mostra apenas as 5 primeiras previsões como exemplo
    print(f"Letra prevista: {predicted_letters[i]}, Letra real: {y[i]}")