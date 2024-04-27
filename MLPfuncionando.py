import numpy as np
import csv
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

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

        # Função (ou funções) de ativação dos neurônios
        # Função sigmoidal com os valores de entrada clipados
        return 1 / (1 + np.exp(-clipped_x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # Loop do procedimento de treinamento
    def train(self, X, y, epochs=1000, learning_rate=0.1, early_stopping=True, patience=20, min_delta=1e-4):
        best_mse = float('inf')
        count = 0

        for epoch in range(epochs):
            # Forward pass
            # Camada escondida da rede neural.
            hidden_input = np.dot(X, self.weights_input_hidden)
            hidden_output = self.sigmoid(hidden_input)

            # Camada de saída da rede neural.
            output_input = np.dot(hidden_output, self.weights_hidden_output)
            output = self.sigmoid(output_input)

            # Cálculo do erro
            # Procedimentos de cálculo de erro na camada de saída
            output_error = y - output
            mse = np.mean(np.square(output_error))

            # Critério de parada do treinamento
            # Verificação de parada antecipada
            if early_stopping and epoch > 0:
                if mse > best_mse - min_delta:
                    count += 1
                else:
                    best_mse = mse
                    count = 0

                if count >= patience:
                    print(f'Early stopping at epoch {epoch}, Best MSE: {best_mse}')
                    break

            # Backpropagation
            # Procedimento de cálculo de informação de erro para a retropropagação
            output_delta = output_error * self.sigmoid_derivative(output)
            # Procedimento de cálculo de contribuição de erro na camada escondida
            hidden_error = output_delta.dot(self.weights_hidden_output.T)
            hidden_delta = hidden_error * self.sigmoid_derivative(hidden_output)

            # Atualização dos pesos
            # Taxa de aprendizado.
            #  Conjunto de pesos sinápticos em cada camada
            # Atualização dos pesos sinápticos em cada camada.
            self.weights_hidden_output += hidden_output.T.dot(output_delta) * learning_rate
            self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate

            # Log da perda (loss)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, MSE: {mse}')

        return mse

    # Treinamento com Parada antecipada e cross validation
    def train(self, X, y, epochs=1000, learning_rate=0.1, use_cross_validation=False, num_folds=5, early_stopping=False, patience=10):
        if use_cross_validation:
            fold_size = len(X) // num_folds
            mse_values = []

            for fold in range(num_folds):
                print(f"Fold {fold + 1}")

                # Dividir os dados em conjunto de treinamento e validação
                start_idx = fold * fold_size
                end_idx = (fold + 1) * fold_size
                X_train = np.concatenate([X[:start_idx], X[end_idx:]], axis=0)
                y_train = np.concatenate([y[:start_idx], y[end_idx:]], axis=0)
                X_val = X[start_idx:end_idx]
                y_val = y[start_idx:end_idx]

                mse = self._train_single_fold(X_train, y_train, X_val, y_val, epochs, learning_rate, early_stopping, patience)
                mse_values.append(mse)

            print(f"Mean MSE across {num_folds} folds: {np.mean(mse_values):.4f}")

        else:
            self._train_single_fold(X, y, X, y, epochs, learning_rate, early_stopping, patience)

    def _train_single_fold(self, X_train, y_train, X_val, y_val, epochs, learning_rate, early_stopping, patience):
        best_mse = float('inf')
        patience_count = 0

        for epoch in range(epochs):
            # Forward pass
            hidden_input = np.dot(X_train, self.weights_input_hidden)
            hidden_output = self.sigmoid(hidden_input)
            output_input = np.dot(hidden_output, self.weights_hidden_output)
            output = self.sigmoid(output_input)

            # Backpropagation
            output_error = y_train - output
            output_delta = output_error * output * (1 - output)
            hidden_error = output_delta.dot(self.weights_hidden_output.T)
            hidden_delta = hidden_error * hidden_output * (1 - hidden_output)

            # Update weights
            self.weights_hidden_output += hidden_output.T.dot(output_delta) * learning_rate
            self.weights_input_hidden += X_train.T.dot(hidden_delta) * learning_rate

            # Calculate MSE on validation set
            if (epoch + 1) % 100 == 0:
                val_pred = self.predict(X_val)
                mse = np.mean((y_val - val_pred) ** 2)
                print(f"Epoch {epoch + 1}, Validation MSE: {mse:.4f}")

                # Early stopping
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

    def predict(self, X):
            
            hidden_input = np.dot(X, self.weights_input_hidden)
            hidden_output = self.sigmoid(hidden_input)

            #Camada de saída da rede neural.
            output_input = np.dot(hidden_output, self.weights_hidden_output)
            output = self.sigmoid(output_input)

            return output

    def save_weights(self, filename = 'pesos.csv'):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Pesos da camada de entrada para oculta'])
            writer.writerows(self.weights_input_hidden)
            writer.writerow(['Pesos da camada oculta para saída'])
            writer.writerows(self.weights_hidden_output)

    def load_weights(self, filename='pesos.csv'):
        with open(filename, newline='') as csvfile:
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


    def plot_confusion_matrix(self, y_true, y_pred):
        # Calcular a matriz de confusão
        # Procedimento de cálculo da matriz de confusão
        num_classes = self.output_size
        confusion_matrix = np.zeros((num_classes, num_classes))
        for true_label, pred_label in zip(y_true, y_pred):
            confusion_matrix[true_label, pred_label] += 1

        # Plotar a matriz de confusão
        plt.figure(figsize=(10, 8))
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, range(num_classes))
        plt.yticks(tick_marks, range(num_classes))
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

# Função para transformar os rótulos de letras em códigos binários
# Procedimento de cálculo da resposta da rede em termos de reconhecimento do caractere
def one_hot_encode(labels):
    unique_labels = np.unique(labels)
    num_labels = len(unique_labels)
    encoding = np.zeros((len(labels), num_labels))
    for i, label in enumerate(labels):
        index = np.where(unique_labels == label)[0][0]
        encoding[i, index] = 1
    return encoding

# Leitura dos dados de entrada e remoção de valores vazios
# Camada de entrada da rede neural.
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

def split_data(X, y, num_folds):
    fold_size = len(X) // num_folds
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    X_folds = []
    y_folds = []
    for i in range(num_folds):
        start = i * fold_size
        end = (i + 1) * fold_size if i < num_folds - 1 else len(X)
        X_folds.append(X_shuffled[start:end])
        y_folds.append(y_shuffled[start:end])
    return X_folds, y_folds

def cross_validation(X, y, num_folds, input_size, hidden_size, output_size, epochs=1000, learning_rate=0.1):
    mlp = MLP(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    X_folds, y_folds = split_data(X, y, num_folds)
    mse_values = []
    for i in range(num_folds):
        X_train = np.concatenate([X_folds[j] for j in range(num_folds) if j != i])
        y_train = np.concatenate([y_folds[j] for j in range(num_folds) if j != i])
        X_test = X_folds[i]
        y_test = y_folds[i]
        mse = mlp.train(X_train, y_train, epochs=epochs, learning_rate=learning_rate)
        mse_values.append(mse)
        print(f'Fold {i+1}, MSE: {mse}')
        # Evaluate on test set (you may need to adapt this based on your needs)
        hidden_input = np.dot(X_test, mlp.weights_input_hidden)
        hidden_output = mlp.sigmoid(hidden_input)
        output_input = np.dot(hidden_output, mlp.weights_hidden_output)
        output = mlp.sigmoid(output_input)
        y_pred = np.argmax(output, axis=1)
        y_true = np.argmax(y_test, axis=1)
        mlp.plot_confusion_matrix(y_true, y_pred)
    return mse_values


# Obter número de camadas escondidas e épocas
num_camadas_escondidas = int(input("Digite o número de camadas escondidas: "))
num_epocas = int(input("Digite o número de épocas: "))
tx_aprendizado = float(input("Digite a taxa de treinamento: "))
parada_antecipada_str = input("Parada antecipada? true or false ")
# Converter a entrada para um valor booleano
if parada_antecipada_str == "true":
    parada_antecipada = True
else:
    parada_antecipada = False
if parada_antecipada:
    pat = int(input("Patience? "))
else:
    pat=20
validacao_cruzada_str = input("Validação Cruzada? true or false ")
if validacao_cruzada_str == "true":
    validacao_cruzada = True
else:
    validacao_cruzada = False
if validacao_cruzada:
    num_vezes = int(input("Num Folds: "))
else:
    num_vezes = 5

# Carregar os dados
X = ler_dados_entrada('x26.txt')
rotulos_letras = ler_rotulos_letras('y26.txt')

# Verificar se o número de amostras de entrada é igual ao número de rótulos
if len(X) != len(rotulos_letras):
    print("Erro: O número de amostras de entrada não é igual ao número de rótulos.")
    exit()

# Transformar os rótulos de letras em códigos binários
y_encoded = one_hot_encode(rotulos_letras)


# Carregar os pesos treinados
mlp = MLP(input_size=120, hidden_size=num_camadas_escondidas, output_size=26)
mlp.load_weights()

# Treinamento da MLP com Hyperparametros
mlp.train(X, y_encoded, epochs=num_epocas, learning_rate=tx_aprendizado, use_cross_validation=validacao_cruzada, num_folds=num_vezes, early_stopping=parada_antecipada, patience=pat)

# Salvar Pesos em csv
mlp.save_weights()

# Letra B
X_example = [[ 1,  1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1,  1,  1, -1,  1,  1,  1,  1, -1, -1,  1,  1,  1,  1,  1, -1,  1,  1,  1,  1,  1, -1,  1,  1,  1,  1,  1, -1,  1,  1,  1,  1, -1, -1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1,  1, -1,  1,  1,  1,  1, -1, -1,  1,  1,  1,  1,  1, -1,  1,  1,  1,  1,  1, -1, -1,  1,  1,  1,  1, -1,  1,  1,  1,  1,  1, -1, -1,  1,  1,  1,  1, -1,  1,  1,  1,  1, -1, -1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1]]  # Vetor de entrada de exemplo
y_example = ["B"]  # Rótulo de exemplo
previsao = mlp.predict(X_example)
letra_prevista_index = np.argmax(previsao)
letra_prevista = chr(ord('a') + letra_prevista_index)  # Converter o índice para a letra correspondente
print("Letra prevista:", letra_prevista)

# Apresentar a matriz de confusão
# mlp.plot_confusion_matrix(X, Y)
