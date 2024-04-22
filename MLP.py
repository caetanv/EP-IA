import random
import math

class Neuronio():
    def __init__(self, pesos, taxa_aprendizado):
        self.pesos = pesos
        self.taxa_aprendizado = taxa_aprendizado

class CamadaMLP(object):
    """ Classe base para representar uma camada de uma rede neural multilayer perceptron"""

    def __init__(self, quantidade_neuronios):
        self.quantidade_neuronios = quantidade_neuronios
        self.neuronios = []
        self.bias = random.uniform(0,1)
        self.bias_peso = 0

    def sigmoide(self, valor_in):
        """ Retorna o valor da função de ativação para o valor y_in """
        return 1 / (1 + math.exp(-valor_in))

    def inicializa_pesos(self, quantidade_observacoes, taxa_de_aprendizado):
        """ Inicializa os pesos dos neuronios da camada de acordo com o número de observações na base de treino """
        self.bias_peso = random.uniform(0,1)
        
        for _ in range(self.quantidade_neuronios):
            pesos = {col_num: random.uniform(0, 1) for col_num in range(quantidade_observacoes)}
            #MUDAR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            
            self.neuronios.append(Neuronio(pesos, taxa_de_aprendizado))

            

    def calcula_saida(self, amostra): #self, amostra, pesos
        """ Calcula a saída da camada de acordo com os valores de entrada X"""
        saidas = []  # lista da saída de cada neuronio da camada
        valores_ins = []  # lista da soma ponderada de cada neuronio da camada
        
        for neuronio in self.neuronios:
            valor_in = 0
            valor_in = self.bias * self.bias_peso
            for col_num in range(len(amostra)):
                valor_in += amostra[col_num] * neuronio.pesos[col_num]

            saidas.append(self.sigmoide(valor_in))
            valores_ins.append(valor_in)
        
        return valores_ins, saidas


class CamadaEntrada(CamadaMLP):
    def __init__(self, quantidade_neuronios):
        super().__init__(quantidade_neuronios)
        


class CamadaEscondida(CamadaMLP):
    def __init__(self, quantidade_neuronios):
        super().__init__(quantidade_neuronios)



class CamadaSaida(CamadaMLP):
    def __init__(self, quantidade_neuronios):
        super().__init__(quantidade_neuronios)


class MultilayerPerceptron(object):
    """ Classe que representa uma rede neural multilayer perceptron 

    Parâmetros:
    n_entrada: int
        Número de neurônios na camada de entrada
    n_escondida: int
        Número de neurônios na camada escondida
    n_saida: int
        Número de neurônios na camada de saída
    taxa_de_aprendizado: float
        Taxa de aprendizagem utilizada no treinamento
    """

    def __init__(self, quantidade_entrada, quantidade_escondida, quantidade_saida, 
                 taxa_de_aprendizado=0.1, limiar: float = 0.0001):
        self.camada_entrada = CamadaEntrada(quantidade_entrada)
        self.camada_escondida = CamadaEscondida(quantidade_escondida)
        self.camada_saida = CamadaSaida(quantidade_saida)
        self.taxa_de_aprendizado = taxa_de_aprendizado
        self.limiar = limiar

    def sigmoide(self, valor_in: float):
        """ Retorna o valor da função de ativação para o valor y_in """
        return 1 / (1 + math.exp(-valor_in))

    def derivada_sigmoide(self, y_in: float):
        """ Retorna o valor da derivada da função de ativação para o valor y_in """
        sig = self.sigmoide(y_in)
        return sig* (1 - sig)
    
    def altera_pesos(self, camada, taxas_correcao):
        """ Altera os pesos dos neuronios da camada de acordo com as correções """
        for neuronio in range(len(camada.neuronios)):
            for peso in range (len(camada.neuronios[neuronio].pesos)):
                camada.neuronios[neuronio].pesos[peso] += taxas_correcao[peso][neuronio]

    def altera_bias(self, camada, taxas_correcao):
        for k in range(len(camada.neuronios)):
            camada.bias_peso += taxas_correcao[k]

    
    def calcula_erro_quadratico_medio(self, erros: list):
        # Calcula o erro quadrático médio da época
        erro_medio = 0
        for erro in erros:
            erro_atual = sum(erro) ** 2
            erro_medio += erro_atual

        return erro_medio / len(erros)

    def feedForward(self, x):
        # Calcula a saída da camada escondida
        entradas_escondida, saidas_escondida = self.camada_escondida.calcula_saida(x)

        # Calcula a saída da camada de saída
        entradas_saida, saidas_saida = self.camada_saida.calcula_saida(saidas_escondida)

        return entradas_escondida, entradas_saida, saidas_escondida, saidas_saida
    
    def backPropagation(self, x, y, entradas_escondida, saidas_escondida, entradas_saida, saidas_saida):
        #cálculo da variação do erro da camada de saída
        variacao_erro_saida = []
        taxas_correcao_pesos_saida = [[] for a in range (self.camada_escondida.quantidade_neuronios)]
        delta_in = []
        variacao_erro_escondida = []
        taxas_correcao_pesos_escondida = [[] for b in range (self.camada_entrada.quantidade_neuronios)]
        taxas_correcao_bias_escondida = []
        taxas_correcao_bias_saida = []

        for j in range(self.camada_escondida.quantidade_neuronios):
            for k in range(self.camada_saida.quantidade_neuronios):
                variacao_erro_saida.append((y[k] - saidas_saida[k]) * self.derivada_sigmoide(entradas_saida[k]))

                taxas_correcao_pesos_saida[j].append(saidas_escondida[j] * variacao_erro_saida[k] * self.taxa_de_aprendizado)
                taxas_correcao_bias_saida.append(self.taxa_de_aprendizado * variacao_erro_saida[k])
        
        for j_ in range(self.camada_escondida.quantidade_neuronios):
            soma = 0
            for k_ in range(self.camada_saida.quantidade_neuronios):
                #inicia calculos necessários para a taxa de correcao da escondida
                soma += variacao_erro_saida[k_] * self.camada_saida.neuronios[k_].pesos[j_]

            delta_in.append(soma)
            variacao_erro_escondida.append(delta_in[j_] * self.derivada_sigmoide(entradas_escondida[j_]))

        for i in range (self.camada_entrada.quantidade_neuronios):
            for j__ in range (self.camada_escondida.quantidade_neuronios):
                taxas_correcao_pesos_escondida[i].append(self.taxa_de_aprendizado * variacao_erro_escondida[j__] * x[i])
                taxas_correcao_bias_escondida.append(self.taxa_de_aprendizado * variacao_erro_escondida[j__])

        return taxas_correcao_pesos_saida, taxas_correcao_pesos_escondida, taxas_correcao_bias_saida, taxas_correcao_bias_escondida
   
        

    def erro_validacao(self, X: list, y_real: list):
        """ Calcula o erro de validação """
        erro_validacao = 0

        for x, y_r in zip(X, y_real):
            entradas_escondida, entradas_saida, saidas_escondida, saidas_saida = self.feedForward(x)
            """Forward
            
            _, saidas_escondida = self.camada_escondida.calcula_saida(
                x)

            
            y_ins, saidas_saida = self.camada_saida.calcula_saida(
                saidas_escondida)
            """
            #self.backPropagation(x, y_real, entradas_escondida, saidas_escondida, entradas_saida, saidas_saida)
            """Backpropagation"""
            # Calcula o erro e correção da camada de saída
            erro_saida, _, _ = self.camada_saida.calcula_correcao(
                y_real=y_r, y_calculado=saidas_saida, valores_in=entradas_saida, saidas_escondida=saidas_escondida)

            erro_epoca = self.calcula_erro_quadratico_medio(erro_saida)
            erro_validacao += erro_epoca
        return erro_validacao

    def deve_parar(self, erro_geral: float, qtde_epocas_sem_melhora: int):
        """ Verifica se a rede deve parar de treinar """
        return erro_geral < self.limiar or qtde_epocas_sem_melhora > 50

    def treina(self, X: list, Y: list, epocas: int = 200, conj_validacao: tuple = None, valor_min_validacao: float = 0.0001):
        """ Treina a rede neural multilayer por meio do algoritmo de backpropagation na sua forma de gradiente descendente """
      
        self.camada_escondida.inicializa_pesos(
            self.camada_entrada.quantidade_neuronios, taxa_de_aprendizado=self.taxa_de_aprendizado)
        self.camada_saida.inicializa_pesos(
            self.camada_escondida.quantidade_neuronios, taxa_de_aprendizado=self.taxa_de_aprendizado)
    

        qtde_epocas_sem_melhora = 0
        for epoca in range(epocas):
            erro_epoca = 0
            erro_validacao = 0
            
            for x, y in zip(X, Y):
                
                entradas_escondida, entradas_saida, saidas_escondida, saidas_saida = self.feedForward(x)
                
                taxas_correcao_pesos_saida, taxas_correcao_pesos_escondida, taxas_correcao_bias_saida, taxas_correcao_bias_escondida = self.backPropagation(x, y, 
                        entradas_escondida, saidas_escondida, entradas_saida, saidas_saida)
                
                # Faz a alteração dos pesos da camada de saída e da camada escondida
                self.altera_pesos(self.camada_saida, taxas_correcao_pesos_saida)
                self.altera_pesos(self.camada_escondida, taxas_correcao_pesos_escondida)

                self.altera_bias(self.camada_saida, taxas_correcao_bias_saida)
                self.altera_bias(self.camada_escondida, taxas_correcao_bias_escondida)

                # Calcula o erro geral
                #erro_quad = self.calcula_erro_quadratico_medio(erro_saida)
                #erro_epoca += erro_quad

            #print(f"Epoca: {epoca} - Erro: {erro_epoca} - Erro de validação: {erro_validacao}")

                print (f"Epoca: {epoca} - saída: {saidas_saida}")
            #if self.deve_parar(erro_epoca, qtde_epocas_sem_melhora): 
             #   break
    
    
    def prediz(self, X: list):
        predito = []

        for x in X:
            _, saidas_escondida = self.camada_escondida.calcula_saida(X)
            _, saidas_saida = self.camada_saida.calcula_saida(saidas_escondida)

        return saidas_saida
'''   
    def gera_matriz_de_confusao(self, X: list, y: list):
        letras = ["A", "B", "C", "D", "E", "J", "K", "L", "V", "Y", "Z"]

        # Pega a quantidade de letras
        tamanho_da_matriz = len(y[0])

        # Gera uma matriz quadrada
        matriz_de_confusao = [
            [0 for _ in range(tamanho_da_matriz)] for _ in range(tamanho_da_matriz)]

        # Limiar de distância entre neurônios
        limiar = 0.2

        
        for x, y_ in zip(X, y):
            # Obtem o indice do neurônio ativado no rótulo
            indice_esperado = y_.index(1)

            # Faz o feedforwarding para obter a predição
            _, saidas_escondida = self.camada_escondida.calcula_saida(x)
            _, saidas_saida = self.camada_saida.calcula_saida(saidas_escondida)

            # Ordenamos os neurônios de acordo com a sua ativação
            saidas_copia = saidas_saida.copy()
            saidas_copia.sort(reverse=True)

            # Pegamos os dois neurônios que mais ativaram
            primeiro_neuronio, segundo_neuronio, *_ = saidas_copia

            # Iremos calcular se os neurônios chegaram em resultados parecidos
            diferenca_aceitavel = (primeiro_neuronio -
                                   segundo_neuronio) > limiar

            # Se a diferença entre os neurônios foi pequena, então ambos ativaram
            # portanto, a rede não conseguiu predizer uma letra para a entrada
            if diferenca_aceitavel is False:
                primeira_letra = saidas_saida.index(primeiro_neuronio)
                segunda_letra = saidas_saida.index(segundo_neuronio)
                print(
                    f"Não consegui prever entre a letra: {letras[primeira_letra]} e {letras[segunda_letra]}, o esperado era {letras[indice_esperado]}")
                continue

            # Caso contrário, o primeiro neurônio foi quem ativou
            # e, iremos buscar o indice dele no vetor
            indice_predito = saidas_saida.index(primeiro_neuronio)

            print(
                f"A letra predita é {letras[indice_predito]} e o esperado era: {letras[indice_esperado]}")

            # Somamos +1 na matriz
            matriz_de_confusao[indice_esperado][indice_predito] += 1

        return matriz_de_confusao
''' 