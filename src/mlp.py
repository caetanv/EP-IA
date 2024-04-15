import numpy as np

class MultilayerPerceptron:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.01):
        """
        Inicializa o Multilayer Perceptron com o tamanho das camadas e a taxa de aprendizado.
        
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
    
    def forward(self, X):
        """
        Executa a propagação direta (forward pass) e retorna a saída.
        
        X: Matriz de entrada (dimensões: [n_entradas, n_amostras]).
        """
        activations = [X]
        zs = []
        
        # Propagação para frente pelas camadas
        for i in range(len(self.weights)):
            z = np.dot(self.weights[i], activations[-1]) + self.biases[i]
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        
        return activations, zs
    
    def backpropagation(self, X, y):
        """
        Executa a backpropagation e retorna os gradientes para os pesos e biases.
        
        X: Matriz de entrada (dimensões: [n_entradas, n_amostras]).
        y: Matriz de saída esperada (dimensões: [n_saidas, n_amostras]).
        """
        # Propagação direta para obter as ativações e z's
        activations, zs = self.forward(X)
        
        # Inicializa listas para armazenar os gradientes
        delta_w = [np.zeros(w.shape) for w in self.weights]
        delta_b = [np.zeros(b.shape) for b in self.biases]
        
        # Calcula o erro na camada de saída
        error = activations[-1] - y
        delta = error * self.sigmoid_derivative(zs[-1])
        
        # Atualiza os gradientes para a última camada
        delta_w[-1] = np.dot(delta, activations[-2].T)
        delta_b[-1] = np.sum(delta, axis=1, keepdims=True)
        
        # Retropropagação para as camadas anteriores
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
        Atualiza os pesos e biases da rede neural com base nos gradientes.
        
        delta_w: Lista de gradientes para os pesos.
        delta_b: Lista de gradientes para os biases.
        """
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * delta_w[i]
            self.biases[i] -= self.learning_rate * delta_b[i]
    
    def train(self, X, y, epochs=10000):
        """
        Treina a rede neural em um conjunto de dados por um número de épocas.
        
        X: Matriz de entrada (dimensões: [n_entradas, n_amostras]).
        y: Matriz de saída esperada (dimensões: [n_saidas, n_amostras]).
        epochs: Número de épocas para treinar.
        """
        for epoch in range(epochs):
            try:
                # Executa backpropagation
                delta_w, delta_b = self.backpropagation(X, y)
                
                # Atualiza os parâmetros
                self.update_params(delta_w, delta_b)
                
                # Calcula a perda (Erro Quadrático Médio - MSE)
                loss = np.mean(np.square(self.forward(X)[0][-1] - y))
                
                # Imprime o progresso do treinamento
                print(f"Época {epoch + 1}/{epochs}, Perda: {loss}")
                
            except Exception as e:
                # Imprime o erro
                print(f"Ocorreu um erro durante o treinamento do MLP na época {epoch}: {e}")
                print(f"Dimensões de X: {X.shape}, y: {y.shape}")
                break
    
    def predict(self, X):
        """
        Faz previsões com a rede neural para um conjunto de entradas.
        
        X: Matriz de entrada (dimensões: [n_entradas, n_amostras]).
        """
        activations, _ = self.forward(X)
        return activations[-1]