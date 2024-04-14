import random
from perceptron import Perceptron
from gerenciador_logs import GerenciadorLogs


class CamadaMLP(object):
    """ Classe base para representar uma camada de uma rede neural multilayer perceptron"""

    def __init__(self, n_neuronios: int):
        self.n_neuronios = n_neuronios
        self.neuronios = []

    def inicializa_pesos(self, n_observacoes: int, tx_aprendizagem: float):
        """ Inicializa os pesos dos neuronios da camada de acordo com o número de observações na base de treino """
        for _ in range(self.n_neuronios):
            pesos = {col_num: random.uniform(-0.5, 0.5)
                     for col_num in range(n_observacoes)}
            bias = random.random()
            self.neuronios.append(Perceptron(pesos, bias, tx_aprendizagem))

    def calcula_saida(self, X: list):
        """ Calcula a saída da camada de acordo com os valores de entrada X"""
        saidas = []  # lista da saída de cada neuronio da camada
        valores_ins = []  # lista da soma ponderada de cada neuronio da camada

        for neuronio in self.neuronios:
            valor_in, saida = neuronio.calcula_saida(X)
            saidas.append(saida)
            valores_ins.append(valor_in)
        return valores_ins, saidas


class CamadaEntrada(CamadaMLP):
    def __init__(self, n_neuronios: int):
        super().__init__(n_neuronios)


class CamadaEscondida(CamadaMLP):
    def __init__(self, n_neuronios: int):
        super().__init__(n_neuronios)

    def calcula_correcao(self, valores_in: list, entradas: list, termos_inf_erro_saida: list):
        """ Calcula o erro e as devidas correções dos pesos dos neuronios da camada escondida """
        correcoes = {}
        for indice, (neuronio, valor_in, termo_inf_erro_saida) in enumerate(zip(self.neuronios, valores_in, termos_inf_erro_saida)):
            termo_inf_erro_saida = sum(termo_inf_erro_saida) if isinstance(
                termo_inf_erro_saida, list) else termo_inf_erro_saida
            termo_inf_erro = sum(termo_inf_erro_saida * neuronio.pesos[indice_peso]
                                 for indice_peso in neuronio.pesos) * neuronio.derivada_funcao_ativacao(valor_in)

            correcoes_pesos, correcao_bias = neuronio.calcula_correcao(
                termo_inf_erro, entradas)
            correcoes[indice] = correcoes_pesos, correcao_bias
        return correcoes


class CamadaSaida(CamadaMLP):
    def __init__(self, n_neuronios: int):
        super().__init__(n_neuronios)

    def calcula_correcao(self, y_real: list, y_calculado: list, valores_in: list, saidas_escondida: list):
        """ Calcula o erro e as devidas correções da camada de saída """
        erros = []
        termos_inf_erro = []
        correcoes = {}

        y_real = y_real if isinstance(y_real, list) else [y_real]

        for indice, (neuronio, y_r, y_c, valor_in) in enumerate(zip(self.neuronios, y_real, y_calculado, valores_in)):
            erro, termo_inf_erro = neuronio.calcula_erro(y_r, y_c, valor_in)
            correcoes_pesos, correcao_bias = neuronio.calcula_correcao(
                erro, saidas_escondida)
            erros.append(erro)
            termos_inf_erro.append(termo_inf_erro)
            correcoes[indice] = correcoes_pesos, correcao_bias

        return erros, correcoes, termos_inf_erro


class MultilayerPerceptron(object):
    """ Classe que representa uma rede neural multilayer perceptron 

    Parâmetros:
    n_entrada: int
        Número de neurônios na camada de entrada
    n_escondida: int
        Número de neurônios na camada escondida
    n_saida: int
        Número de neurônios na camada de saída
    tx_aprendizagem: float
        Taxa de aprendizagem utilizada no treinamento
    """

    def __init__(self, n_entrada: int, n_escondida: int, n_saida: int, tx_aprendizagem=0.1, limiar: float = 0.0001, gerenciador_logs: GerenciadorLogs = None):
        self.camada_entrada = CamadaEntrada(n_entrada)
        self.camada_escondida = CamadaEscondida(n_escondida)
        self.camada_saida = CamadaSaida(n_saida)
        self.tx_aprendizagem = tx_aprendizagem
        self.limiar = limiar
        self.gerenciador_logs = gerenciador_logs

    def altera_pesos(self, camada: CamadaMLP, correcoes: list):
        """ Altera os pesos dos neuronios da camada de acordo com as correções """
        for neuronio, correcao in zip(camada.neuronios, correcoes):
            neuronio.altera_pesos(correcoes[correcao])

    def checa_melhora_validacao(self, erros_validacao: list):
        """ Checa se a taxa de erro da validacao aumentou """
        if len(erros_validacao) < 2:
            return False
        return erros_validacao[-1] < erros_validacao[-2]
    
    def calcula_erro_quadratico_medio(self, erros: list):
        # Calcula o erro quadrático médio da época
        erro_medio = 0
        for erro in erros:
            erro_atual = sum(erro) ** 2
            erro_medio += erro_atual

        return erro_medio / len(erros)

    def erro_validacao(self, X: list, y_real: list):
        """ Calcula o erro de validação """
        erro_validacao = 0

        for x, y_r in zip(X, y_real):
            """Forward"""
            # Calcula a saída da camada escondida
            _, saidas_escondida = self.camada_escondida.calcula_saida(
                x)

            # Calcula a saída da camada de saída
            y_ins, saidas_saida = self.camada_saida.calcula_saida(
                saidas_escondida)

            """Backpropagation"""
            # Calcula o erro e correção da camada de saída
            erro_saida, _, _ = self.camada_saida.calcula_correcao(
                y_real=y_r, y_calculado=saidas_saida, valores_in=y_ins, saidas_escondida=saidas_escondida)

            erro_epoca = self.calcula_erro_quadratico_medio(erro_saida)
            erro_validacao += erro_epoca
        return erro_validacao

    def deve_parar(self, erro_geral: float, qtde_epocas_sem_melhora: int):
        """ Verifica se a rede deve parar de treinar """
        return erro_geral < self.limiar or qtde_epocas_sem_melhora > 50

    def treina(self, X: list, y: list, epocas: int = 200, conj_validacao: tuple = None, valor_min_validacao: float = 0.0001):
        """ Treina a rede neural multilayer por meio do algoritmo de backpropagation na sua forma de gradiente descendente """

        self.camada_escondida.inicializa_pesos(
            self.camada_entrada.n_neuronios, tx_aprendizagem=self.tx_aprendizagem)
        self.camada_saida.inicializa_pesos(
            self.camada_escondida.n_neuronios, tx_aprendizagem=self.tx_aprendizagem)

        
        qtde_epocas_sem_melhora = 0
        for epoca in range(epocas):
            erro_epoca = 0
            erro_validacao = 0
            
            for x, y_ in zip(X, y):
                """Forward"""
                # Calcula a saída da camada escondida
                z_ins, saidas_escondida = self.camada_escondida.calcula_saida(
                    x)

                # Calcula a saída da camada de saída
                y_ins, saidas_saida = self.camada_saida.calcula_saida(
                    saidas_escondida)

                """Backpropagation"""
                # Calcula o erro e correção da camada de saída
                erro_saida, correcao_saida, termos_inf_erro_saida = self.camada_saida.calcula_correcao(
                    y_real=y_, y_calculado=saidas_saida, valores_in=y_ins, saidas_escondida=saidas_escondida)

                # Calcula o erro e a e correção da camada escondida
                correcao_escondida = self.camada_escondida.calcula_correcao(
                    valores_in=z_ins, entradas=x, termos_inf_erro_saida=termos_inf_erro_saida)

                # Faz a alteração dos pesos da camada de saída e da camada escondida
                self.altera_pesos(self.camada_saida, correcao_saida)
                self.altera_pesos(self.camada_escondida, correcao_escondida)

                # Calcula o erro geral
                erro_quad = self.calcula_erro_quadratico_medio(erro_saida)
                erro_epoca += erro_quad

            # Calcula o erro de validação se houver conjunto de validação
            if conj_validacao is not None:
                erro_val = self.erro_validacao(
                    conj_validacao[0], conj_validacao[1])
                melhoria = (erro_validacao - erro_val) < valor_min_validacao # checa se a melhoria em relação à epoca passada é maior que o valor mínimo tolerado
   
                qtde_epocas_sem_melhora = 0 if (melhoria or epoca == 0) else qtde_epocas_sem_melhora + 1
                erro_validacao = erro_val

            self.gerenciador_logs.adiciona_log(self.__class__.__name__, x,  saidas_saida, self.camada_escondida.neuronios,
                                               self.camada_saida.neuronios, self.tx_aprendizagem, epoca, -1, self.limiar, epocas, erro_epoca, erro_validacao)

            print(f"Epoca: {epoca} - Erro: {erro_epoca} - Erro de validação: {erro_validacao}")

            if self.deve_parar(erro_epoca, qtde_epocas_sem_melhora): 
                break
    
    
    def prediz(self, X: list):
        predito = []

        for x in X:
            _, saidas_escondida = self.camada_escondida.calcula_saida(x)
            _, saidas_saida = self.camada_saida.calcula_saida(saidas_escondida)
            predito.append(saidas_saida)

        return predito
    
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

    def to_json(self):
        """Função que desserializa a rede neural e transforma em um objeto json"""
        modelo = {
            'tx_aprendizagem': self.tx_aprendizagem,
            'camada_entrada':  self.camada_entrada.n_neuronios,
            'camada_escondida': {
                'n_neuronios': self.camada_escondida.n_neuronios,
                'neuronios': [
                    {
                        'pesos': neuronio.pesos,
                        'bias': neuronio.bias
                    } for neuronio in self.camada_escondida.neuronios

                ]
            },
            'camada_saida': {
                'n_neuronios': self.camada_saida.n_neuronios,
                'neuronios': [
                    {
                        'pesos': neuronio.pesos,
                        'bias': neuronio.bias
                    } for neuronio in self.camada_saida.neuronios
                ]
            }
        }
        return modelo

    def from_json(modelo):
        """Função que serializa a rede neural e transforma em um objeto python"""
        mlp = MultilayerPerceptron(
            modelo['camada_entrada'],
            modelo['camada_escondida']['n_neuronios'],
            modelo['camada_saida']['n_neuronios'],
            modelo['tx_aprendizagem']
        )

        for neuronio in modelo['camada_escondida']['neuronios']:
            pesos = {int(indice): valor for indice,
                     valor in neuronio['pesos'].items()}
            mlp.camada_escondida.neuronios.append(
                Perceptron(pesos, neuronio['bias'], mlp.tx_aprendizagem))
        for neuronio in modelo['camada_saida']['neuronios']:
            pesos = {int(indice): valor for indice,
                     valor in neuronio['pesos'].items()}
            mlp.camada_saida.neuronios.append(
                Perceptron(pesos, neuronio['bias'], mlp.tx_aprendizagem))
        return mlp
