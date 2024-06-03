'''
ACH2016 - Inteligência Artificial                
   EACH-USP - Primeiro Semestre de 2024                           
   Turma 04 - Prof. Sarajane Marques Peres                  
                                                                 
  Primeiro Exercicio-Programa                                   
                                                                 
  Guilherme Fernandes Aliaga - 13672432
  Marcos Vilela Rezende Júnior - 13729806
  Vinicius Kazuo Inagaki - 13747967
  Vitor Caetano da Silva - 9276999
  
'''

import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from datetime import datetime
from utilfunctions import Utilfunctions
from mlp import MLP

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
    util = Utilfunctions()

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