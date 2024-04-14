import math


class Perceptron(object):
    def __init__(self, pesos: dict, bias: float, tx_aprendizagem: float):
        self.pesos = pesos
        self.bias = bias
        self.tx_aprendizagem = tx_aprendizagem

    def funcao_ativacao(self, valor_in: float):
        """ Retorna o valor da função de ativação para o valor y_in """
        return 1 / (1 + math.exp(-valor_in))

    def derivada_funcao_ativacao(self, y_in: float):
        """ Retorna o valor da derivada da função de ativação para o valor y_in """
        return self.funcao_ativacao(y_in) * (1 - self.funcao_ativacao(y_in))

    def calcula_saida(self, X: list):
        """ Calcula a saída do perceptron para o valor X """
        valor_in = self.bias + sum([X[col_num] * self.pesos[col_num]
                                    for col_num in range(len(X))])
        return valor_in, self.funcao_ativacao(valor_in)

    def calcula_erro(self, y_real: list or float, y_calculado: list or float, valor_in: list or float):
        """ Calcula o termo de informação do erro para o valor y_real e y_calculado """
        erros = []
        termos_inf_erro = []

        if not isinstance(y_real, list):
            erro = y_real - y_calculado
            termo_inf_erro = erro * \
                self.derivada_funcao_ativacao(valor_in)
            return [erro], [termo_inf_erro]

        for y in y_real:
            erro = y - y_calculado
            termo_inf_erro = erro * self.derivada_funcao_ativacao(valor_in)
            erros.append(erro)
            termos_inf_erro.append(termo_inf_erro)

        return erros, termos_inf_erro

    def calcula_correcao(self, termos_inf_erro: list or float, saida_camada: list or float):
        """ Calcula a correção dos pesos do perceptron """
        correcoes = []
        termo_inf_erro = sum(termos_inf_erro) if isinstance(
            termos_inf_erro, list) else termos_inf_erro

        for j in self.pesos:
            correcao = self.tx_aprendizagem * termo_inf_erro * saida_camada[j]
            correcoes.append(correcao)

        correcoes_bias = self.tx_aprendizagem * termo_inf_erro

        return correcoes, correcoes_bias

    def altera_pesos(self, correcao: dict):
        """ Altera os pesos do perceptron """
        for col_num in range(len(self.pesos)):
            self.pesos[col_num] += correcao[0][col_num]
            self.bias += correcao[1]

    def converte_json(self):
        """ Retorna o objeto em formato JSON """
        return {
            "media_pesos": sum(self.pesos.values()) / len(self.pesos),
            "bias": self.bias,
            "tx_aprendizagem": self.tx_aprendizagem
        }
