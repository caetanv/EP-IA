import numpy as np
import csv
from pathlib import Path
import os
import matplotlib.pyplot as plt

# https://github.com/EdgarLiraa/IA-Multilayer-Perceptron/blob/main/rede.py

class MLP:
    def __init__(self, input_size, hidden_size, output_size, weights_filename='weights.csv'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Inicialização dos pesos e biases
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = np.zeros(hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_output = np.zeros(output_size)

        self.weights_filename = weights_filename

        self.Valores_MSE = []
        self.épocas = []
        
        # Carregar os pesos salvos se o arquivo existir
        #self.load_weights()

    def sigmoid(self, x):
        # Clipagem dos valores de entrada para evitar overflow
        clipped_x = np.clip(x, -500, 500)  # Valores de corte escolhidos arbitrariamente

        # Função (ou funções) de ativação dos neurônios
        # Função sigmoidal com os valores de entrada clipados
        return 1 / (1 + np.exp(-clipped_x))

    # Função que calcula a derivada da sigmoide
    def sigmoid_derivative(self, x):
        return x * (1 - x)


    # Treinamento com Parada antecipada e cross validation
    def train(self, X, y, epochs=1000, learning_rate=0.1, use_cross_validation=False, num_folds=5, early_stopping=False, patience=10):
        if use_cross_validation:
            self.cross_validation(X,y,num_folds, self.input_size, self.hidden_size, self.output_size, epochs, learning_rate)
        else:
            self._train_single_fold(X, y, X, y, epochs, learning_rate, early_stopping, patience)

    def _train_single_fold(self, X_train, y_train, X_val, y_val, epochs, learning_rate, early_stopping, patience):
        best_mse = float('inf')
        patience_count = 0

        for epoch in range(epochs):

            output, hidden_output = self.forward_pass(X_train)
            
            # Backpropagation
            self.back_propagation(X_train,y_train,learning_rate,output,hidden_output)
            
            # Calculate MSE on validation set
            if (epoch + 1) % 100 == 0:
                val_pred = self.predict(X_val)
                mse = np.mean((y_val - val_pred) ** 2)
                self.Valores_MSE.append(mse)
                self.épocas.append(epoch)
                print(f"Epoch {epoch + 1}, Validation MSE: {mse:.4f}")

                # Early stoppingot(self.sigmoid(output).T, ou
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

    def save_weights(self):
        with open(self.weights_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Pesos da camada de entrada para oculta'])
            writer.writerows(self.weights_input_hidden)
            writer.writerow(['Pesos da camada oculta para saída'])
            writer.writerows(self.weights_hidden_output)

    def load_weights(self):
        if os.path.exists(self.weights_filename):
            with open(self.weights_filename, newline='') as csvfile:
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
            print("Pesos carregados", self.weights_filename)


    
    def forward_pass(self, X_train):
        # Calcular a entrada líquida para a camada oculta
        hidden_input = np.dot(X_train, self.weights_input_hidden) + self.bias_hidden
        # Aplicar a função de ativação
        hidden_output = self.sigmoid(hidden_input)
        
        # Calcular a entrada líquida para a camada de saída
        output_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
        # Aplicar a função de ativação (pode ser softmax para classificação)
        output = self.sigmoid(output_input)

        return output, hidden_output

    def back_propagation(self, X, y, learning_rate, output, hidden_output):

        if len(y) == len(output):
            # Calcula o erro na camada de saída
            output_error = y - output
            # Calcula o gradiente da camada de saída usando a derivada da função sigmoidal
            output_delta = output_error * self.sigmoid_derivative(output)

            # Calcula o erro na camada oculta retrocedendo o erro da camada de saída
            hidden_error = output_delta.dot(self.weights_hidden_output.T)
            # Calcula o gradiente da camada oculta usando a derivada da função sigmoidal
            hidden_delta = hidden_error * self.sigmoid_derivative(hidden_output)

            # Atualiza os pesos e biases da camada de saída
            self.weights_hidden_output += hidden_output.T.dot(output_delta) * learning_rate
            self.bias_output += np.sum(output_delta, axis=0) * learning_rate

            # Atualiza os pesos e biases da camada oculta
            self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
            self.bias_hidden += np.sum(hidden_delta, axis=0) * learning_rate


    def calculate_confusion_matrix(self, y_true, y_pred, num_classes):
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
        for true_label, pred_label in zip(y_true, y_pred):
            confusion_matrix[true_label, pred_label] += 1
        return confusion_matrix

    def plot_confusion_matrix(self, confusion_matrix, classes):
        num_classes = len(classes)
        plt.figure(figsize=(8, 6))
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')

        for i in range(num_classes):
            for j in range(num_classes):
                plt.text(j, i, confusion_matrix[i, j], ha='center', va='center', color='white' if confusion_matrix[i, j] > (confusion_matrix.max() / 2) else 'black')

        plt.show()

    def cross_validation(self, X, y, num_folds, input_size, hidden_size, output_size, epochs=1000, learning_rate=0.1):
        folds_X, folds_y = split_data(X, y, num_folds)
        accuracies = []
        for i in range(num_folds):
            # Separa os dados em conjunto de treinamento e validação
            X_train = np.concatenate([folds_X[j] for j in range(num_folds) if j != i])
            y_train = np.concatenate([folds_y[j] for j in range(num_folds) if j != i])
            X_val = folds_X[i]
            y_val = folds_y[i]

            # Treina o modelo
            self.train(X_train, y_train, epochs=epochs, learning_rate=learning_rate)

            # Avalia o modelo no conjunto de validação
            predictions = self.predict(X_val)
            true_labels = np.argmax(y_val, axis=1)
            accuracy = np.mean(np.argmax(predictions, axis=1) == true_labels)
            accuracies.append(accuracy)

            print(f"Fold {i + 1} Accuracy: {accuracy:.4f}")

            print (f"Acurácia do modelo: {np.mean(accuracies)}")
        return accuracies


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

def load_data(file_x, file_y):
    return np.array(ler_dados_entrada(file_x)), np.array(ler_rotulos_letras(file_y))

# Leitura dos dados de entrada e remoção de valores vazios
# Camada de entrada da rede neural.
def ler_dados_entrada(nome_arquivo):
    dados = []
    atual = os.getcwd()
    os.chdir('caracteres-completo')
    with open(nome_arquivo, 'r') as arquivo:
        linhas = arquivo.readlines()
        for linha in linhas:
            valores = linha.strip().split(',')
            valores = [int(valor) for valor in valores if valor.strip()]  # Remover valores vazios
            if valores:  # Se a linha não estiver vazia após a remoção dos valores vazios
                dados.append(valores)
    
    os.chdir(atual)
    return np.array(dados)


# Leitura dos rótulos de letras
def ler_rotulos_letras(nome_arquivo):
    atual = os.getcwd()
    os.chdir('caracteres-completo')
    with open(nome_arquivo, 'r') as arquivo:
        rotulos = arquivo.readlines()
    os.chdir(atual)
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


def letra_para_indice(letra):
    if 'A' <= letra <= 'Z':
        return ord(letra) - ord('A')
    elif 'a' <= letra <= 'z':
        return ord(letra) - ord('a')
    else:
        raise ValueError("Caractere fornecido não é uma letra do alfabeto.")


def letras_para_indices(vetor_letras):
    indices = []
    for letra in vetor_letras:
        if 'A' <= letra <= 'Z' or 'a' <= letra <= 'z':
            indice = letra_para_indice(letra)
            indices.append(indice)
        else:
            raise ValueError(f"Caractere '{letra}' não é uma letra do alfabeto.")
    return indices

def transformar_rotulos(labels):
    return [letras_para_indices(label) for label in labels]

def plot_gráfico(listax, listay):

        plt.plot(listax, listay, marker='o')

        plt.title('Gráfico de MSE em ralação às épocas')
        plt.xlabel('Valor do MSE')
        plt.ylabel('Épocas`')

        plt.show()

if __name__ == "__main__":
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

    # Carregar dados de treinamento
    #X_train, y_train = load_data('X.txt', 'Y_letra.txt')
    X_train, y_train = load_data('X26.txt', 'Y26.txt')

    # Verificar se o número de amostras de entrada é igual ao número de rótulos
    if len(X_train) != len(y_train):
        print("Erro: O número de amostras de entrada não é igual ao número de rótulos.")
        exit()

    # Transformar os rótulos de letras em códigos binários
    y_encoded = one_hot_encode(y_train)


    # Carregar os pesos treinados
    mlp = MLP(input_size=120, hidden_size=num_camadas_escondidas, output_size=26)


    # Treinamento da MLP com Hyperparametros
    mlp.train(X_train, y_encoded, epochs=num_epocas, learning_rate=tx_aprendizado, use_cross_validation=validacao_cruzada, num_folds=num_vezes, early_stopping=parada_antecipada, patience=pat)

    # Salvar Pesos em csv
    mlp.save_weights()


    i = 0
    y_pred = []
    classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'] # Exemplo de classes (substitua pelas suas)
    num_classes = len(classes)
    y_true = letras_para_indices(y_train)

    for item in X_train:
        print("Valor em binario",item)
        previsao = mlp.predict(item)
        letra_prevista_index = np.argmax(previsao)
        letra_prevista = chr(ord('a') + letra_prevista_index)
        print("Letra Prevista:", letra_prevista)
        print("Letra Real", y_train[i])
        y_pred.append(letra_prevista_index)
        i = i + 1


    print(len(X_train))
    print(len(y_true))

    y = mlp.Valores_MSE
    x = mlp.épocas

    plot_gráfico(x,y)
    confusion_matrix = mlp.calculate_confusion_matrix(y_true,y_pred,num_classes)
    mlp.plot_confusion_matrix(confusion_matrix, classes)