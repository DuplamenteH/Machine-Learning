# -*- coding: ut
"""
Editor Spyder

Este é um arquivo de script temporário.
"""
"""
CARREGANDO OS DADOS.
zootopia
0,0,0,0,0,0,0,1,1,1,1,0,1,110,27.74456356
"""
import pandas as pd
import matplotlib.pyplot as plt  
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

filmes = pd.read_csv('movies_multilinear_reg.csv')


#Só queremos as features e, então, 
#separamos em dados de treino e teste

filmes_atributos = filmes[filmes.columns[2:17]]
filmes_bilheteria = filmes[filmes.columns[17:]]
treino, teste, treino_marcacoes, teste_marcacoes =  train_test_split(filmes_atributos, 
                                                                     filmes_bilheteria,
                                                                     test_size = 0.3)

#Criando o modelo
modelo = LinearRegression()
modelo.fit(treino, treino_marcacoes)

score = modelo.score(teste, teste_marcacoes)
score


#Teste
#previsão do valor da bilheteria com o modelo criado.
#Titulo,Documentary,Sci-Fi,Mystery,Horror,Romance,Thriller,Crime,Fantasy,Comedy,Animation,Children,Drama,Adventure,Duracao,Investimento,Bilheteria
zootopia = modelo.predict([[0,0,0,0,0,0,0,1,1,1,1,0,1,110,27.74456356]])
zootopia

macaco = modelo.predict([[0,1,0,0,0,0,0,0,0,0,0,0,0,150,5]])
macaco

filmeNome = [1,1,0,0,0,0,0,1,0,0,0,0,0,210,7]
caso3=modelo.predict([filmeNome])
print(caso3)