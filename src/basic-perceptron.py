import numpy as np

ms = []
mt = []
pt = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
w = []
b = []
y_in = []
y = []
ms = np.array(ms)
zigma = 0
threshold = 0.1

def op(file):
    matrix = []
    lines = [line.rstrip('\n') for line in open(file)]
    for yline, line in enumerate(lines):
        for xline, ch in enumerate(line):
            matrix.append(1) if ch == 'o' else matrix.append(-1)
    print(matrix)
    matrixS(matrix)

def openfile(file):
    matriz = []
    with open(file, 'r') as arquivo:
        linhas = arquivo.readlines()
        for linha in linhas:
            valores = linha.strip().replace(' ','').split(',')
            for valor in valores:
                if valor != '' :
                    linha_matriz = int(valor)
                    matriz.append(linha_matriz)
    print(matriz)
    matrixS(matriz)

def matrixS(m):
    global ms
    if len(ms) == 0: ms = m
    else: ms = np.vstack([ms, m])

def matrixT(a):
    global mt
    matrix = []
    for i in range(len(pt)):
        if pt[i] == a : matrix.append(1)
        else : matrix.append(-1)
    if len(mt) == 0: mt = matrix
    else: mt = np.vstack([mt, matrix])
    
def train():
    Trained = False
    global zigma,w,b,y_in,y,mt
    if len(w) == 0:
        w = np.full([len(pt), len(ms)], 0.5)
        b = np.full([len(pt)], 0.5)
        y_in = np.zeros([len(pt)])
        y = np.zeros([len(pt)])
    for train in range(10000):
        for h in range(np.shape(ms)[0]):
            for i in range(len(pt)):
                for j in range(len(ms)):
                    zigma = zigma + ms[h][j] * w[i][j]
                y_in[i] = b[i] + zigma
                zigma = 0
                if y_in[i]>threshold : y[i] = 1
                elif -threshold<y_in[i]<threshold : y[i] = 0
                elif y_in[i]<-threshold : y[i] = -1

            for z in range(len(pt)):
                if y[z] != mt[h][z]: 
                    true = 1
                    break

            if true == 1:
                for i in range(len(pt)):
                    for j in range(len(ms)):
                        w[i][j] = w[i][j] + mt[h][i] * ms[h][j]
                    b[i] = b[i] + mt[h][i]
            else: 
                Trained = True
                break
        if Trained == True:
            break

def guess(file):
    g = 0
    mg = []
    lines = [line.rstrip('\n') for line in open(file)]
    for yline, line in enumerate(lines):
        for xline, ch in enumerate(line):
            mg.append(1) if ch == 'o' else mg.append(-1)
        
    yt = np.zeros(len(pt))
    for i in range(len(pt)):
        for j in range(len(mt)):
            g = g + mg[j]*w[i][j]
        yt[i] = g
        if yt[i]>threshold : yt[i] = 1
        else : yt[i] = -1
    return yt

def check(file):
    alphabet = ''
    yg = guess(file)
    for i in range(len(pt)):
        if yg[i] == 1 : 
            alphabet = pt[i]
            break
    return alphabet

""" Input Training Data """
import pdb
openfile('X.txt')


#matrixT('a')
#op('A2.txt')
#matrixT('a')
#op('B1.txt')
#matrixT('b')
train()

""" Guess The Alphabet """
gs = check('A3.txt')
print(gs)
gs = check('B1.txt')
print(gs)
gs = check('G3.txt')
print(gs)