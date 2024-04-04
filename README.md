# TRABALHO EP IA - MLP e CNN

## DATAS DE ENTREGA:

### MLP 21 de Maio

### CNN 25 de Junho

# Conteúdo das Entregas

## OBJETIVO 1 (MLP)
Implementar uma rede neural artificial Multilayer Perceptron (MLP) - sem fazer uso de bibliotecas especializadas em redes neurais artificiais - com uma camada escondida e treinada com o algoritmo Backpropagation em sua versão de Gradiente Descendente - algoritmo de treinamento discutido em sala de aula. 

- Dados para treinamento e teste: OR,AND,XOR,Caracteres Completo

### Arquivos de Saída
- hiperparametros finais da arquitetura da rn e hiperparametros de inicialização (parameters.py)
- pesos iniciais 
- pesos finais
- erro cometido pela rede em cada iteração
- saídas produzidas pela rede

### Primeiro Video (MLP)
- Treinamento sem e com validação cruzada e parada anteciopada para o conjunto de dados
- Estudo dos parâmetros
- Teste da MLP

### Segundo Video (MLP)
- Apresentação dos conceitos usados no código: camada de entrada da rn, camada de saida, camada escondida
- função de ativação dos neuronios, tx de aprendizado, termos de regularização ou procedimentos de otimização adaptativos
- loop do treinamento, critério de parada
- conjunto de pesos em cada camada, atualização dos pesos em cada camada
- procedimentos de calculo de erro na camada de saida, procedimento de calculo de informação de erro para a retropropagação, procedimento de calculo de contribuição de erro na acamada escondida, procedimento de calculo da resposta de rede em termos de reconhecimento do caractere, procedimento de calculo da matriz de confusão


## OBJETIVO 2 (CNN)
Implementar uma rede neural artificial Convolution Neural Network (CNN)
O conjunto de dados escolhido deve ser usado para teste da rede neural, considerando
duas tarefas:
● Uma tarefa multiclasse, ou seja, é esperado que a rede neural saiba responder corretamente para todas as classes do conjunto.
● Uma tarefa de classificação binária, ou seja, apenas duas classes, ou duas composições de várias classes, devem ser usados no treinamento e a rede deverá reconhecer essas duas classes. Você pode variar os pares de classes em seus testes. 

### Arquivos de Saída
- hiperparametros da arquitetura da rede e inicialização
- pesos iniciais da rede
- pesos finais da rede
- erro cometido pela rede a cada iteração
- saídas produzidas pela rede para cada um dos dados

### Terceiro Video (CNN)
- Avaliação em termos de funcionamento
- Modelagem das entradas da CNN
- Camadas de aplicação de kernel e pooling
- Modelagem das camadas densas e camadas de saída
- Lógica do treinamento da estrutura completa
- Critério de parada do treinamento
- Procedimentos e cálculo do erro na camada de saída
- Procedimento de cálculo da resposta da rede em termos de reconhecimento de caractere
- Teste da CNN para o conjunto de dados MNIST
- Procediemento de calculo da matriz de confusão


### Descrição das imagens
Caracteres Completo X_png contem os arquivos de imagem em dimensão 10x12 em png dos caracteres em ordem alfabética. Repetindo a cada 26 caracteres o alfabeto. (1326 / 26) (51 alfabetos)

X.txt contém cada caractere separado por linha em vetores de 1,-1 para cada bit da matriz do caractere

Y_letra.txt valor da letra correspondente ao caractere


### Modelagem da Arquitetura
120 Neurônios de Entrada (Referente aos 12x10 pixels do png) (Recebem 1 ou -1)
26 Neurônios de Saída (Referente aos 26 rótulos do alfabeto) (Saem com 1 e -1)


Classes: 
- Classe para o Neuronio Perceptron
 Métodos : inicialização, treinamento, teste, função de ativação, matriz de confusão
- Classe para a arquitetura da rede neural
 Métodos : inicialização, rodar
- Classe para os hiperparâmetros
 Métodos : get_parameters
- Classe para geração de arquivos de saida
 Métodos : write_output
- Classe para leitura dos arquivos para continuação dos testes
 Métodos : read_file

#### Bibliografia:

### MLP
https://github.com/caetanv/multilayer-perceptron

https://github.com/123epsilon/Neural-Network-in-Python

https://github.com/manoharmukku/multilayer-perceptron-in-python

https://colab.research.google.com/github/bentrevett/pytorch-image-classification/blob/master/1_mlp.ipynb

https://github.com/caetanv/perceptron-fausset 

https://github.com/caetanv/Perceptron-Adaline-implementation

### Perceptron
https://github.com/rfguri/perceptron

https://github.com/whoisraibolt/Single-Layer-Perceptron

https://github.com/Hello-World-Blog/Perceptron

https://github.com/aonurakman/Simple-Char-Recognition-w-Perceptron/tree/main


### Backpropagation
https://github.com/hvanhaagen/backpropagation/tree/main

https://github.com/xprathamesh/Backpropagation-in-MLP


### Funcao OR 
https://github.com/ZahidHasan/Perceptron

### CNN
https://github.com/caetanv/cnn-facial-landmark

https://github.com/caetanv/Convolution-Neural-Network-

https://github.com/caetanv/Conjunctivitis-image-detection

https://github.com/caetanv/CNN-for-Text-Classification

https://github.com/caetanv/DigitRecognizer




