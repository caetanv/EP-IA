from MLP import MultilayerPerceptron
import readers

X = readers.dataX

y = readers.dataY

mlp = MultilayerPerceptron(len(X[0]), 30, len(y[0]), taxa_de_aprendizado=0.1, limiar=0.0001)
mlp.treina(X, y, epocas=20)
print(mlp.prediz(X[0]))