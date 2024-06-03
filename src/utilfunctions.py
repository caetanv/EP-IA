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

# Classe com funções para auxiliar na leitura dos dados de entrada
class Utilfunctions:
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
