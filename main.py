from MLP import MultilayerPerceptron
import readers

X = readers.dataX

y = readers.dataY

x_teste = []
y_teste = []

for col in range(26*5):
    x_teste.append(readers.dataX[col])

for col2 in range(26*5):
    y_teste.append(readers.dataY[col2])

teste1 = [
    [-1,-1],
    [-1, 1],
    [1,-1],
    [1,1]
    ]

teste2 = [
    [-1],
    [-1],
    [-1],
    [1]
]
#mlp = MultilayerPerceptron(len(X[0]), 30, len(y[0]), taxa_de_aprendizado=0.1, limiar=0.0001)
#mlp.treina(x_teste, y_teste, epocas=500)

mlpAND = MultilayerPerceptron(len(teste1[0]), 3, len(teste2[0]), taxa_de_aprendizado=0.5)
mlpAND.treina(teste1, teste2, epocas=300)

#print(mlp.prediz(X[0]))
print(mlpAND.prediz(teste1[0]))

print('debug')