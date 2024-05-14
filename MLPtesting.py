import numpy as np
import pandas as pd
import csv
import os

# Definição da classe MLP
class MLP:
    def __init__(self, input_size, hidden_size, output_size, alpha=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.alpha = alpha

        # Inicialização dos pesos
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        # Camada oculta
        hidden_input = np.dot(x, self.weights_input_hidden)
        hidden_output = self.sigmoid(hidden_input)

        # Camada de saída
        output_input = np.dot(hidden_output, self.weights_hidden_output)
        output = self.sigmoid(output_input)

        return output

    def train(self, X, y, epochs=1000, early_stopping=False, patience=10):
        best_mse = float('inf')
        patience_count = 0

        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)

            # Backpropagation
            output_error = y - output
            output_delta = output_error * output * (1 - output)

            hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
            hidden_delta = hidden_error * self.sigmoid(output) * (1 - self.sigmoid(output))

            # Atualização dos pesos
            self.weights_hidden_output += np.dot(self.sigmoid(output).T, output_delta) * self.alpha
            self.weights_input_hidden += np.dot(X.T, hidden_delta) * self.alpha

            # Calcular MSE
            mse = np.mean(output_error ** 2)

            # Verificar Early Stopping
            if early_stopping:
                if mse < best_mse:
                    best_mse = mse
                    patience_count = 0
                else:
                    patience_count += 1
                    if patience_count >= patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break

        return mse

    def predict(self, x):
        return self.forward(x)

    def save_weights(self, filename='weights.csv'):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Weights Input to Hidden'])
            writer.writerows(self.weights_input_hidden)
            writer.writerow(['Weights Hidden to Output'])
            writer.writerows(self.weights_hidden_output)

    def load_weights(self, filename='weights.csv'):
        if os.path.exists(filename):
            with open(filename, newline='') as csvfile:
                reader = csv.reader(csvfile)
                weights_input_hidden = []
                weights_hidden_output = []
                current_row = None
                for row in reader:
                    if len(row) == 0:
                        continue
                    if row[0].startswith('Weights'):
                        current_row = row[0]
                        continue
                    if current_row == 'Weights Input to Hidden':
                        weights_input_hidden.append([float(val) for val in row])
                    elif current_row == 'Weights Hidden to Output':
                        weights_hidden_output.append([float(val) for val in row])

            self.weights_input_hidden = np.array(weights_input_hidden)
            self.weights_hidden_output = np.array(weights_hidden_output)

# Carregar dados de exemplo
def load_example_data():
    X_example = np.array([
        [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1],
        [-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1],
        [1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1]
    ])

    y_example = np.array([
        [1, 0, 0],  # Representa a letra A
        [0, 1, 0],  # Representa a letra B
        [0, 0, 1]   # Representa a letra C
    ])

    return X_example, y_example

# Função para decodificar a previsão
def decode_prediction(prediction):
    alphabet = "ABC"
    index = np.argmax(prediction)
    return alphabet[index]

# Exemplo de utilização
if __name__ == "__main__":
    # Carregar os dados de exemplo
    X_example, y_example = load_example_data()

    # Inicializar e treinar o MLP
    mlp = MLP(input_size=120, hidden_size=10, output_size=3, alpha=0.1)
    mse = mlp.train(X_example, y_example, epochs=1000, early_stopping=True, patience=20)

    # Fazer previsões
    for i in range(len(X_example)):
        prediction = mlp.predict(X_example[i])
        decoded_letter = decode_prediction(prediction)
        print(f"Vetor {i + 1}: Letra prevista = {decoded_letter}")