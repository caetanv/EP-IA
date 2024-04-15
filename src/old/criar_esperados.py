import csv
alfabeto = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 
            'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
data = [[] for _ in range(len(alfabeto))]

#criando arrays do modelo [true, false, false, ... , false] para cada letra do alfabeto

for i in range (len(alfabeto)):
    for j in range(len(data)):
        if i == j:
            data[i].append(1)
        else:
            data[i].append(-1)

#print para teste
for i in range(len(data)):
    print(data[i])
    print('\n')