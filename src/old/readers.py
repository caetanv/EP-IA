import csv
import numpy as np
# inicia a classe dos leitores
class Reader:

    def __init__(self):
        return

    #leitor para os problemas de portas lógicas
    def porta_logica_reader(self, file_path):
    
        entradas, saidas = [], []
    
        with open(file_path, mode = 'r', encoding= 'utf-8-sig') as and_file:
            csv_reader = csv.reader(and_file)

            for row in csv_reader:
                entradas.append([int(row[0]), int(row[1])])
                saidas.append(int(row[2]))

        return entradas, saidas
    
    #leitor para caracteres representados por números
    def caracteres_reader(self, file_path):

        entradas = []
        aux = []

        with open(file_path, mode = 'r', encoding= 'utf-8-sig') as file:
            csv_reader = csv.reader(file)

            for row in csv_reader:
                for line in row:
                    aux.append(int(line))

                entradas.append(aux)
                aux = []

        return entradas

    #leitor para caracteres representados por letras
    def read_letters (self, file_path):

        letters = []

        with open(file_path, mode='r') as file:
            csv_reader = csv.reader(file)

            for row in csv_reader:
                for line in row:
                    letters.append(line)

        return letters

#continuar...
    
'''
reader = Reader()

entradaCSV, saidaCSV = reader.porta_logica_reader("portas-logicas/problemAND.csv")

entradaY = reader.read_letters('caracteres-completo/Y_letra.csv')

print (entradaY)
'''