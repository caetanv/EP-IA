import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from datetime import datetime

class MLP:
    def __init__(self, input_size, hidden_size, output_size, weights_filename='pesos_finais.csv'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Inicialização dos pesos e biases
        # Multiplicando pesos por 0.01 para evitar a saturação dos vetores gradientes
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
        self.bias_hidden = np.zeros(hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.01
        self.bias_output = np.zeros(output_size)

        self.weights_filename = weights_filename

        # Listas que auxiliam na construção dos gráficos
        self.valores_MSE_train = []
        self.valores_MSE_val = []
        self.epocas = []
        self.accuracies_train = []
        self.accuracies_val = []
        self.cont_fold = []

        # Carregar os pesos salvos se o arquivo existir
        #self.load_weights()

    def sigmoid(self, x):
        # Clipagem dos valores de entrada para evitar overflow
        clipped_x = np.clip(x, -500, 500)  # Valores de corte escolhidos arbitrariamente

        # Função (ou funções) de ativação dos neurônios
        # Função sigmoidal com os valores de entrada clipados
        return 1 / (1 + np.exp(-clipped_x))

    # Método que calcula a derivada da sigmoide
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    # Método de treinamento que dependendo da entrada do usuário, realiza  (ou não) a cross validation
    def train(self, X, y, X_val, y_val, epochs=1000, learning_rate=0.1, use_cross_validation=False, num_folds=5, early_stopping=False, patience=10):      
        if use_cross_validation:
            self.cross_validation(X,y,num_folds, self.input_size, self.hidden_size, self.output_size, epochs, learning_rate)

        else:
            self._train_single_fold(X, y, X_val, y_val, epochs, learning_rate, early_stopping, patience)

        print (f'Acurácia do modelo: {(np.max(self.accuracies_val) * 100):.1f}%')

        
    # Método de treinamento específica para o caso de um único fold
    def _train_single_fold(self, X_train, y_train, X_val, y_val, epochs, learning_rate, early_stopping, patience):
        best_mse = 0.01
        patience_count = 0
        mse_val = 0

        for epoch in range(epochs):

            output, hidden_output = self.forward_pass(X_train)
            
            # Backpropagation
            self.back_propagation(X_train,y_train,learning_rate,output,hidden_output)
            
            # Define o intervalo em que erro e acurácia são definidos
            if (epoch + 1) % 100 == 0:
                # Calcula o erro para conjunto de teste
                val_pred_train = self.predict(X_train)
                mse_train = np.mean((y_train - val_pred_train) ** 2)
                self.valores_MSE_train.append(mse_train)

                #Calcula o erro para conjunto de validação
                val_pred_val = self.predict(X_val)
                mse_val = np.mean((y_val - val_pred_val) ** 2)
                self.valores_MSE_val.append(mse_val)

                # Avalia o modelo no conjunto de treino
                predictions_train = self.predict(X_train)
                true_labels_train = np.argmax(y_train, axis=1)
                accuracy_train = np.mean(np.argmax(predictions_train, axis=1) == true_labels_train)
                self.accuracies_train.append(accuracy_train)

                # Avalia o modelo no conjunto de validação
                predictions_val = self.predict(X_val)
                true_labels_val = np.argmax(y_val, axis=1)
                accuracy_val = np.mean(np.argmax(predictions_val, axis=1) == true_labels_val)
                self.accuracies_val.append(accuracy_val)

                # Guarda o valor das épocas correspondentes
                self.epocas.append(epoch)
                # Exibe no terminal , respectivamente, a época, o erro quadrático médio do conjunto de treino, o erro quadrático médio do conjunto de validação, a acurácia do conjunto de treino e a acurácia do conjunto de validação na respectiva época
                print(f"Época {epoch + 1}: MSE Treino: {mse_train:.4f}, MSE Validação: {mse_val:.4f}, Acurácia de Treino: {accuracy_train:.4f}, Acurácia de Validação: {accuracy_val:.4f}")

            # Parada antecipada, onde só muda caso houver uma mudança substancial no erro (até 3 casas decimais)
            if early_stopping:
                if mse_val > best_mse:
                    best_mse = mse_val
                    patience_count = 0
                else:
                    patience_count += 1
                    if patience_count >= patience:
                        print(f"Parada antecipada na época {epoch + 1}")
                        break
        
        #Funções para criação dos gráficos
        self.gráfico_MSE(self.epocas, self.valores_MSE_train, self.epocas, self.valores_MSE_val)
        self.gráfico_acc(self.epocas, self.accuracies_train, self.epocas, self.accuracies_val)
        
        return

    # Método de previsão da rede neural
    def predict(self, X):
            
        hidden_input = np.dot(X, self.weights_input_hidden)
        hidden_output = self.sigmoid(hidden_input)

        #Camada de saída da rede neural.
        output_input = np.dot(hidden_output, self.weights_hidden_output)
        output = self.sigmoid(output_input)

        return output

    # Método que converte a previsão numérica em uma letra do alfabeto
    def prever_letra(self, item):
        return chr(ord('a') + np.argmax(mlp.predict(item)))

    # Método para salvar os pesos em um arquivo
    def save_weights(self, filename):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Pesos da camada de entrada para oculta'])
            writer.writerows(self.weights_input_hidden)
            writer.writerow(['Pesos da camada oculta para saida'])
            writer.writerows(self.weights_hidden_output)

    # Método para carregar os pesos de um arquivo
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
                    elif current_row == 'Pesos da camada oculta para saida':
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


    # Método para dividir os dados de folds de tamanhos iguais
    def split_data(self, X, y, num_folds):
        num_alfabetos = len(X) // 26     # Obtem a quantidade total de alfabetos
        a = num_alfabetos // num_folds   # Divide os alfabetos em  k-folds de partes iguais
        fold_size = a * 26               # Obtem as letras dos alfabetos em cada fold

        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        X_folds = []
        y_folds = []
        for i in range(num_folds):
            start = (i * fold_size) + 1
            end = (i + 1) * fold_size if i < num_folds - 1 else len(X)
            X_folds.append(X_shuffled[start:end])
            y_folds.append(y_shuffled[start:end])
        return X_folds, y_folds

    # Método que calcula a matriz de confusão
    def calculate_confusion_matrix(self, y_true, y_pred, num_classes):
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
        for true_label, pred_label in zip(y_true, y_pred):
            confusion_matrix[true_label, pred_label] += 1
        return confusion_matrix

    # printar acurácia e precisão, recall e f1score
    def print_accuracy_precision_recall_f1(self, confusion_matrix):
        accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
        precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
        recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
        f1_score = 2 * (precision * recall) / (precision + recall)
        
        print(f"Accuracy: {accuracy:.2f}")
        
        for idx in range(len(precision)):
            print(f"Class {idx}:")
            print(f"  Precision: {precision[idx]:.2f}")
            print(f"  Recall: {recall[idx]:.2f}")
            print(f"  F1 Score: {f1_score[idx]:.2f}")


    # Método para exibir a matriz
    def plot_confusion_matrix(self, confusion_matrix, classes):
        num_classes = len(classes)
        plt.figure(figsize=(8, 6))
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Matriz de Confusão')
        plt.colorbar()
        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        plt.xlabel('Valores preditos')
        plt.ylabel('Valores reais')

        for i in range(num_classes):
            for j in range(num_classes):
                plt.text(j, i, confusion_matrix[i, j], ha='center', va='center', color='white' if confusion_matrix[i, j] > (confusion_matrix.max() / 2) else 'black')

        # Obter a data atual
        current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # Nome do arquivo com a data atual
        filename = f'grafico_matriz_{current_date}.png'
        dir = 'gráficos/matriz'

        filepath = os.path.join(dir, filename)

        if not os.path.exists(dir):
            os.makedirs(dir)
        # Salvar o gráfico
        plt.savefig(filepath)
        plt.show()

    def calculate_metrics(self,confusion_matrix):
        FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
        FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
        TP = np.diag(confusion_matrix)
        TN = confusion_matrix.sum() - (FP + FN + TP)
        
        FP = FP.sum()
        FN = FN.sum()
        TP = TP.sum()
        TN = TN.sum()
        
        # Calculate percentages
        total = FP + FN + TP + TN
        FP_rate = FP / total
        FN_rate = FN / total
        TP_rate = TP / total
        TN_rate = TN / total
        
        print(f"True Positives: {TP} ({TP_rate:.2%})")
        print(f"True Negatives: {TN} ({TN_rate:.2%})")
        print(f"False Positives: {FP} ({FP_rate:.2%})")
        print(f"False Negatives: {FN} ({FN_rate:.2%})")


    def cross_validation(self, X, y, num_folds, input_size, hidden_size, output_size, epochs, learning_rate):
        folds_X, folds_y = self.split_data(X, y, num_folds)
        mlps = []
        for i in range(num_folds):
            # Separa os dados em conjunto de treinamento e validação
            X_train = np.concatenate([folds_X[j] for j in range(num_folds) if j != i])
            y_train = np.concatenate([folds_y[j] for j in range(num_folds) if j != i])
            X_val = folds_X[i]
            y_val = folds_y[i]

            # Treina o modelo
            mlp = MLP(self.input_size, self.hidden_size, self.output_size, self.weights_filename)
            mlp.train(X_train, y_train, X_val, y_val, epochs=epochs, learning_rate=learning_rate, use_cross_validation=False)
            mlp.cont_fold.append(i+1)
            mlps.append(mlp)
            
            print(f"Fim do Fold {i + 1}")

        # Verifica qual mlp possui maior acurácia nos dados de validação
        max = 0 
        for m in range(len(mlps)):
            if np.max(mlps[m].accuracies_val) > max:
                max = np.max(mlps[m].accuracies_val)
                z = m

        # Incorpora os melhores pesos a partir da mlp com maior acurácia
        self.weights_input_hidden = mlps[z].weights_input_hidden
        self.bias_hidden = mlps[z].bias_hidden
        self.weights_hidden_output = mlps[z].weights_hidden_output
        self.bias_output = mlps[z].bias_output
        self.accuracies_val = mlps[z].accuracies_val
        self.cont_fold = mlps[z].cont_fold
        
        return
    
    # Método para criar gráfico de MSE em função das épocas
    def gráfico_MSE(self, x_train, y_train, x_val, y_val):
        plt.plot(x_train, y_train, color='purple', label='Treino')
        plt.plot(x_val, y_val, color='orange', label='Validação')
        plt.title('Gráfico de MSE em relação às épocas')
        plt.xlabel('Épocas')
        plt.ylabel('Valor do MSE')
        plt.legend()

        # Obter a data atual
        current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        dir = 'gráficos/MSE'

        # Nome do arquivo com a data atual
        filename = f'grafico_MSE_{self.cont_fold}_{current_date}.png'
        filepath = os.path.join(dir, filename)
        
        if not os.path.exists(dir):
            os.makedirs(dir)

        # Salvar o gráfico
        plt.savefig(filepath)
        plt.show()

    # Método para criar gráfico de acurácia em função dos 'fold'
    def gráfico_acc(self, x_train , y_train, x_val, y_val):
        plt.plot(x_train, y_train, color='purple', label='Treino')
        plt.plot(x_val, y_val, color='orange', label='Validação')
        plt.title('Gráfico de acurácia em relação às épocas')
        plt.xlabel('Épocas')
        plt.ylabel('Acurácia')
        plt.legend()

        # Obter a data atual
        current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # Nome do arquivo com a data atual
        dir = 'gráficos/acurácia'
        filename = f'grafico_ACC_{self.cont_fold}_{current_date}.png'

        filepath = os.path.join(dir, filename)

        if not os.path.exists(dir):
            os.makedirs(dir)
        # Salvar o gráfico
        plt.savefig(filepath)
        plt.show()



# Classe com funções para auxiliar na leitura dos dados de entrada
class Util_Functions:
    def __init__(self):
        pass

    # Método para transformar os rótulos de letras em códigos binários
    # Procedimento de cálculo da resposta da rede em termos de reconhecimento do caractere
    def one_hot_encode(self, labels):
        unique_labels = np.unique(labels)
        num_labels = len(unique_labels)
        encoding = np.zeros((len(labels), num_labels))
        for i, label in enumerate(labels):
            index = np.where(unique_labels == label)[0][0]
            encoding[i, index] = 1
        return encoding

    def load_data(self, file_x, file_y):
        return np.array(self.ler_dados_entrada(file_x)), np.array(self.ler_rotulos_letras(file_y))

    # Leitura dos dados de entrada e remoção de valores vazios
    # Camada de entrada da rede neural.
    def ler_dados_entrada(self, nome_arquivo):
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
    def ler_rotulos_letras(self, nome_arquivo):
        atual = os.getcwd()
        os.chdir('caracteres-completo')
        with open(nome_arquivo, 'r') as arquivo:
            rotulos = arquivo.readlines()
        os.chdir(atual)
        return [rotulo.strip() for rotulo in rotulos]

    # Transforma uma letra em um índice de intervalo [0:25]
    def letra_para_indice(self, letra):
        if 'A' <= letra <= 'Z':
            return ord(letra) - ord('A')
        elif 'a' <= letra <= 'z':
            return ord(letra) - ord('a')
        else:
            raise ValueError("Caractere fornecido não é uma letra do alfabeto.")

    # Transforma um vetor de letras em um vetor de índices de intervalo [0:25]
    def letras_para_indices(self, vetor_letras):
        indices = []
        for letra in vetor_letras:
            if 'A' <= letra <= 'Z' or 'a' <= letra <= 'z':
                indice = self.letra_para_indice(letra)
                indices.append(indice)
            else:
                raise ValueError(f"Caractere '{letra}' não é uma letra do alfabeto.")
        return indices

class Programa:

  def __init__(self): # Valores definidos apenas para inicialização do programa, após isso, os valores são alterados
      self.num_camadas_escondidas = 30
      self.num_epocas = 3000
      self.tx_aprendizado = 0.01
      self.parada_antecipada = False
      self.validacao_cruzada = False
      self.num_vezes = 0
      self.pat = 0
      self.mlp = None
      self.avaliação = 0
      
  def carregar_hyperparametros(self):
    # Entrada do usuário para obter número de camadas escondidas, épocas, taxa de treinamento, parada antecipada e validação cruzada
    self.num_camadas_escondidas = int(input("Digite o número de neurônios na camadas escondidas: "))
    self.num_epocas = int(input("Digite o número de épocas: "))
    self.tx_aprendizado = float(input("Digite a taxa de treinamento: "))
    parada_antecipada_str = input("Parada antecipada? true or false ")
    # Converter a entrada para um valor booleano
    if parada_antecipada_str == "true":
        self.parada_antecipada = True
    else:
        self.parada_antecipada = False

    # Obter valor de 'Patience' do usuário para a parada antecipada
    if self.parada_antecipada: 
        self.pat = int(input("Patience? "))
    else:
        self.pat=20

    # Seleção da estratégia de avaliação do classificador
    self.avaliação = int(input("Escolha a estratégia para o classificador:\n[0] Hold-out\n[1] Validação cruzada\n[2] Hold-out e Validação cruzada\n"))
    
  def carregar_mlp(self):
    util = Util_Functions()

    # Carrega os dados e separa-os em dados de treinamento e dados de validação de acordo com a avaliação escolhida pelo usuário
    if self.avaliação == 2: # Cross Validation e Hold-out
        self.validacao_cruzada = True
        self.num_vezes = int(input("Num Folds: ")) # Obtem quantidade de folds
        # Dados separados em 70% para treinamento e 30% para validação:
        X_train, y_train = util.load_data('X_treinamento.txt', 'Y_treinamento.txt')
        X_val, y_val = util.load_data('X_validação.txt', 'Y_validação.txt')     

    elif self.avaliação == 1: # Cross Validation
        self.validacao_cruzada = True
        self.num_vezes = int(input("Num Folds: ")) # Obtem quantidade de folds
        X_train, y_train = util.load_data('X_CV.txt', 'Y_CV.txt') # Conjunto de dados contendo todos os alfabetos menos os 5 últimos
        X_val, y_val = util.load_data('X_CV.txt', 'Y_CV.txt')

    elif self.avaliação == 0: # Hold-out
        self.num_vezes = 5
        self.validacao_cruzada = False
        # Dados separados em 70% para treinamento e 30% para validação:
        X_train, y_train = util.load_data('X_treinamento.txt', 'Y_treinamento.txt')
        X_val, y_val = util.load_data('X_validação.txt', 'Y_validação.txt') 

    X_test, y_test = util.load_data('X_verificação_final.txt', 'Y_verificação_final.txt') # Conjunto de dados de teste final da mlp contendo os últimos 5 alfabetos dos dados totais

    # Verificar se o número de amostras de entrada é igual ao número de rótulos
    if len(X_train) != len(y_train):
        print("Erro: O número de amostras de entrada não é igual ao número de rótulos.")
        exit()

    # Transformar os rótulos de letras em códigos binários
    y_encoded = util.one_hot_encode(y_train)
    y_val_encoded = util.one_hot_encode(y_val)

    # Carregar os pesos treinados
    self.mlp = MLP(input_size=120, hidden_size=self.num_camadas_escondidas, output_size=26)

    # Salvar pesos iniciais em csv
    self.mlp.save_weights('pesos_iniciais.csv')

    # Treinamento da MLP com Hyperparametros
    self.mlp.train(X_train, y_encoded, X_val, y_val_encoded, epochs=self.num_epocas, learning_rate=self.tx_aprendizado, use_cross_validation=self.validacao_cruzada, num_folds=self.num_vezes, early_stopping=self.parada_antecipada, patience=self.pat)

    # Salvar Pesos finais em csv
    self.mlp.save_weights('pesos_finais.csv')

    i = 0
    y_pred = []
    num_classes = 26
    classes = [chr(i + ord('A')) for i in range(num_classes)]
    y_true = util.letras_para_indices(y_test)

    # Representação de cada letra do alfabeto com o vetor binário e demonstração do resultado em comparação à amostra real
    for item in X_test:
        #print("Valor em binario",item)
        previsao = self.mlp.predict(item)
        letra_prevista_index = np.argmax(previsao)
        letra_prevista = chr(ord('a') + letra_prevista_index)
        #print("Letra Prevista:", letra_prevista)
        #print("Letra Real", y_train[i])
        y_pred.append(letra_prevista_index)
        i = i + 1

    print(len(X_train)) # Quantidade de letras treinadas
    print(len(y_true))  # Quantidade de letras no conjunto de teste final da mlp
    # Exibição dos dados de entrada do usuário (Número de neurônios na camada escondida, Número de épocas, taxa de treinamento)
    print("Número de neurônios na camada escondida: " + str(self.num_camadas_escondidas) + "\nNúmero de épocas: " + str(self.num_epocas) + "\nTaxa de treinamento: " + str(self.tx_aprendizado))

    # Cálculo da matriz e criação da matriz de confusão
    confusion_matrix = self.mlp.calculate_confusion_matrix(y_true,y_pred,num_classes)
    self.mlp.print_accuracy_precision_recall_f1(confusion_matrix)
    self.mlp.calculate_metrics(confusion_matrix)
    self.mlp.plot_confusion_matrix(confusion_matrix, classes)

  def iniciar_programa(self):
    self.carregar_hyperparametros()
    self.carregar_mlp()


if __name__ == "__main__":
    programa = Programa()
    programa.iniciar_programa()