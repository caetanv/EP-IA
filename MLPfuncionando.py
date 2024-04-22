import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Inicialização dos pesos
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)

    def sigmoid(self, x):
        # Clipagem dos valores de entrada para evitar overflow
        clipped_x = np.clip(x, -500, 500)  # Valores de corte escolhidos arbitrariamente

        # Função sigmoidal com os valores de entrada clipados
        return 1 / (1 + np.exp(-clipped_x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, X, y, epochs=1000, learning_rate=0.1):
        for epoch in range(epochs):
            # Forward pass
            hidden_input = np.dot(X, self.weights_input_hidden)
            hidden_output = self.sigmoid(hidden_input)

            output_input = np.dot(hidden_output, self.weights_hidden_output)
            output = self.sigmoid(output_input)

            # Cálculo do erro
            output_error = y - output
            mse = np.mean(np.square(output_error))

            # Backpropagation
            output_delta = output_error * self.sigmoid_derivative(output)
            hidden_error = output_delta.dot(self.weights_hidden_output.T)
            hidden_delta = hidden_error * self.sigmoid_derivative(hidden_output)

            # Atualização dos pesos
            self.weights_hidden_output += hidden_output.T.dot(output_delta) * learning_rate
            self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate

            # Log da perda (loss)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {mse}')

    def predict(self, X):
            
            hidden_input = np.dot(X, self.weights_input_hidden)
            hidden_output = self.sigmoid(hidden_input)

            output_input = np.dot(hidden_output, self.weights_hidden_output)
            output = self.sigmoid(output_input)

            return output

    def save_weights(self):
        with open('pesos.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Pesos da camada de entrada para oculta'])
            writer.writerows(self.weights_input_hidden)
            writer.writerow(['Pesos da camada oculta para saída'])
            writer.writerows(self.weights_hidden_output)

    def load_weights(self):
        with open('pesos.csv', newline='') as csvfile:
            reader = csv.reader(csvfile)
            weights_input_hidden = []
            weights_hidden_output = []
            current_row = None
            for row in reader:
                if len(row) == 0:
                    continue
                if row[0].startswith('Pesos'):
                    current_row = row[0]
                    continue
                if current_row == 'Pesos da camada de entrada para oculta':
                    weights_input_hidden.append([float(val) for val in row])
                elif current_row == 'Pesos da camada oculta para saída':
                    weights_hidden_output.append([float(val) for val in row])

        self.weights_input_hidden = np.array(weights_input_hidden)
        self.weights_hidden_output = np.array(weights_hidden_output)


# Função para transformar os rótulos de letras em códigos binários
def one_hot_encode(labels):
    unique_labels = np.unique(labels)
    num_labels = len(unique_labels)
    encoding = np.zeros((len(labels), num_labels))
    for i, label in enumerate(labels):
        index = np.where(unique_labels == label)[0][0]
        encoding[i, index] = 1
    return encoding

# Leitura dos dados de entrada e remoção de valores vazios
def ler_dados_entrada(nome_arquivo):
    dados = []
    rotulos = []
    with open(nome_arquivo, 'r') as arquivo:
        linhas = arquivo.readlines()
        for linha in linhas:
            valores = linha.strip().split(',')
            valores = [int(valor) for valor in valores if valor.strip()]  # Remover valores vazios
            if valores:  # Se a linha não estiver vazia após a remoção dos valores vazios
                dados.append(valores)
    return np.array(dados)

# Leitura dos rótulos de letras
def ler_rotulos_letras(nome_arquivo):
    with open(nome_arquivo, 'r') as arquivo:
        rotulos = arquivo.readlines()
    return [rotulo.strip() for rotulo in rotulos]



# Obter número de camadas escondidas e épocas
num_camadas_escondidas = int(input("Digite o número de camadas escondidas: "))
num_epocas = int(input("Digite o número de épocas: "))

# Carregar os dados
X = ler_dados_entrada('x26.txt')
rotulos_letras = ler_rotulos_letras('y26.txt')

# Verificar se o número de amostras de entrada é igual ao número de rótulos
if len(X) != len(rotulos_letras):
    print("Erro: O número de amostras de entrada não é igual ao número de rótulos.")
    exit()

# Transformar os rótulos de letras em códigos binários
y = one_hot_encode(rotulos_letras)

# Treinamento do MLP
mlp = MLP(input_size=120, hidden_size=num_camadas_escondidas, output_size=26)
mlp.train(X, y, epochs=num_epocas)


exemplo = [[ 1,  1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1,  1,  1, -1,  1,  1,  1,  1, -1, -1,  1,  1,  1,  1,  1, -1,  1,  1,  1,  1,  1, -1,  1,  1,  1,  1,  1, -1,  1,  1,  1,  1, -1, -1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1,  1, -1,  1,  1,  1,  1, -1, -1,  1,  1,  1,  1,  1, -1,  1,  1,  1,  1,  1, -1, -1,  1,  1,  1,  1, -1,  1,  1,  1,  1,  1, -1, -1,  1,  1,  1,  1, -1,  1,  1,  1,  1, -1, -1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1 ]]
print(exemplo)
previsao = mlp.predict(exemplo)
letra_prevista_index = np.argmax(previsao)
letra_prevista = chr(ord('a') + letra_prevista_index)  # Converter o índice para a letra correspondente
print("Letra prevista:", letra_prevista)