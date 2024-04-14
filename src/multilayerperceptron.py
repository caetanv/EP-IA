import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class MultilayerPerceptron:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.01):
        """
        Inicializa o Multilayer Perceptron com os tamanhos das camadas e a taxa de aprendizado.
        
        input_size: O tamanho da camada de entrada.
        hidden_sizes: Uma lista com os tamanhos de cada camada oculta (por exemplo, [3, 3]).
        output_size: O tamanho da camada de saída.
        learning_rate: A taxa de aprendizado para o treinamento.
        """
        # Inicializa os parâmetros
        self.learning_rate = learning_rate
        
        # Cria uma lista com os tamanhos de todas as camadas, começando pela entrada, depois as ocultas, e por fim a saída
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        # Inicializa os pesos e biases para cada camada
        self.weights = []
        self.biases = []
        
        # Inicializa os pesos e vies para cada camada
        for i in range(1, len(self.layer_sizes)):
            weight_matrix = np.random.randn(self.layer_sizes[i], self.layer_sizes[i - 1])
            bias_vector = np.random.randn(self.layer_sizes[i], 1)
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)
    
    def sigmoid(self, z):
        """Função de ativação sigmoide."""
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        """Derivada da função de ativação sigmoide."""
        sig = self.sigmoid(z)
        return sig * (1 - sig)
    
    def softmax(self, z):
        """Função de ativação softmax para a camada de saída."""
        exp_z = np.exp(z - np.max(z, axis=0))
        return exp_z / exp_z.sum(axis=0, keepdims=True)
    
    def forward(self, X):
        """
        Executa a propagação direta e retorna as ativações e as entradas ponderadas (zs).
        
        X: Matriz de entrada (dimensões: [n_entradas, n_amostras]).
        """
        activations = [X]
        zs = []
        
        # Propagação direta pelas camadas
        for i in range(len(self.weights)):

            z = np.dot(self.weights[i], activations[-1]) + self.biases[i]
            zs.append(z)
            
            if i == len(self.weights) - 1:
                # Última camada: ativação softmax para classificação
                activation = self.softmax(z)
            else:
                # Outras camadas: função de ativação sigmoide
                activation = self.sigmoid(z)
            
            activations.append(activation)
        
        return activations, zs
    
    def backpropagation(self, X, y):
        """
        Executa a backpropagation e retorna os gradientes para os pesos e vies.
        
        X: Matriz de entrada (dimensões: [n_entradas, n_amostras]).
        y: Matriz de saída esperada (dimensões: [n_saidas, n_amostras]).
        """
        # Propagação direta para obter ativações e zs
        activations, zs = self.forward(X)
        
        # Inicializa listas para armazenar os gradientes
        delta_w = [np.zeros(w.shape) for w in self.weights]
        delta_b = [np.zeros(b.shape) for b in self.biases]
        
        # Calcula o erro na camada de saída
        error = activations[-1] - y
        delta = error
        
        # Calcula gradientes para a última camada
        delta_w[-1] = np.dot(delta, activations[-2].T)
        delta_b[-1] = np.sum(delta, axis=1, keepdims=True)
        
        # Retropropaga para as camadas anteriores
        for i in range(2, len(self.layer_sizes)):
            z = zs[-i]
            activation_prev = activations[-i - 1]
            sp = self.sigmoid_derivative(z)
            delta = np.dot(self.weights[-i + 1].T, delta) * sp
            
            delta_w[-i] = np.dot(delta, activation_prev.T)
            delta_b[-i] = np.sum(delta, axis=1, keepdims=True)
        
        return delta_w, delta_b
    
    def update_params(self, delta_w, delta_b):
        """
        Atualiza os pesos e vies com base nos gradientes calculados.
        
        delta_w: Lista de gradientes para os pesos.
        delta_b: Lista de gradientes para os vies.
        """
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * delta_w[i]
            self.biases[i] -= self.learning_rate * delta_b[i]
    
    def train(self, X, y, epochs=1000):
        """
        Treina o MLP com os dados fornecidos por um número de épocas.
        
        X: Matriz de entrada (dimensões: [n_entradas, n_amostras]).
        y: Matriz de saída esperada (dimensões: [n_saidas, n_amostras]).
        epochs: Número de épocas para treinar.
        """
        for epoch in range(epochs):
            # Executa backpropagation
            delta_w, delta_b = self.backpropagation(X, y)
            
            # Atualiza os parâmetros
            self.update_params(delta_w, delta_b)
            
            # Calcula a perda (erro quadrático médio)
            loss = np.mean(np.square(activations[-1] - y))
            
            # Mostra o progresso do treinamento a cada 100 épocas
            if epoch % 100 == 0:
                print(f"Época {epoch + 1}/{epochs}, Perda: {loss}")
    
    def predict(self, X):
        """
        Faz previsões com o MLP para uma matriz de entrada.
        
        X: Matriz de entrada (dimensões: [n_entradas, n_amostras]).
        """
        activations, _ = self.forward(X)
        return np.argmax(activations[-1], axis=0)
    
def load_data(input_csv, label_csv):
    """
    Carrega os dados de entrada e rótulos dos arquivos CSV.
    
    input_csv: Caminho para o arquivo CSV com os dados de entrada.
    label_csv: Caminho para o arquivo CSV com os rótulos.
    """
    # Carrega os dados de entrada
    data = pd.read_csv(input_csv, header=None).values
    # Carrega os rótulos
    labels = pd.read_csv(label_csv, header=None).values.ravel()
    
    return data, labels

def encode_labels(labels):
    """
    Codifica os rótulos em one-hot encoding.
    
    labels: Lista de rótulos.
    """
    encoder = OneHotEncoder(sparse_output=False)
    labels_encoded = encoder.fit_transform(labels.reshape(-1, 1))
    return labels_encoded, encoder

def main():
    # Carrega os dados
    input_csv = 'X.csv'
    label_csv = 'Y_letra.csv'
    
    data, labels = load_data(input_csv, label_csv)
    
    # Codifica os rótulos em one-hot encoding
    labels_encoded, encoder = encode_labels(labels)
    
    # Define as dimensões da rede neural
    input_size = data.shape[1]
    hidden_sizes = [32, 16]  # Dimensões das camadas ocultas
    output_size = labels_encoded.shape[1]  # Número de classes (26 letras)
    
    # Cria o MLP
    mlp = MultilayerPerceptron(input_size, hidden_sizes, output_size, learning_rate=0.01)
    
    # Treina o MLP
    mlp.train(data, labels_encoded, epochs=1000)
    
    # Teste o MLP com novos dados (você pode fornecer novos dados para testar)
    new_data = data[:5]  # Por exemplo, usar as primeiras 5 linhas dos dados de treinamento
    predictions = mlp.predict(new_data)
    
    # Decodifica as previsões para letras usando o encoder
    predicted_letters = encoder.inverse_transform(predictions.reshape(-1, 1))
    print(f"Previsões: {predicted_letters}")

if __name__ == "__main__":
    main()