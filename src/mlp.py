import random, copy

class MLP:

    def __init__(self, amostras, saidas_esperadas, num_camada_escondida, taxa_aprendizado, epocas, bias):
        self.amostras = amostras #todas as amostras
        self.saidas_esperadas = saidas_esperadas #saídas esperadas do MLP
        self.num_camada_escondida = num_camada_escondida #número de neurônios na camada escondida
        self.taxa_aprendizado = taxa_aprendizado #taxa de aprendiado (valor entre 0 e 1)
        self.epocas = epocas #número de épocas
        self.bias = bias #bias
        self.num_amostras = len(amostras) #número de amostras na camada de entrada
        self.num_elementos_amostra = len(amostras) #número de elementos para cada amostra
        self.pesos_entrada = [[] for amostra in range(self.num_amostras)] #vetor/matriz de pesos

        #falta CAMADA ESCONDIDA
        #falta PESOS DA CAMADA ESCONDIDA

    def treinar(self):

        #adiciona o bias para camada de entrada
        for amostra in range(self.num_amostras):
            self.amostras.insert(0, self.bias)

        #iniciar o vetor de pesos com valores aleatórios
        for i in range(self.num_elementos_amostra):
            for j in range(self.num_camada_escondida):
                self.pesos_entrada[i].append(random.random())

        #inserir bias no vetor de pesos
        #self.pesos_entrada #ERRO: o numero de pesos deve ser igual a quantidade de pesos da camada escondida CONTINUAR DEPOIS)

        #CONTINUA......

mlp= MLP ([1, 2, 3, 4], [2], 3, 1, 2, 1) #debug
mlp.treinar()

print ('x')