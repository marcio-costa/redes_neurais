# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 22:20:57 2019

@author: marci
"""

import numpy as np

entradas = np.array([[0,0], [0,1], [1,0], [1,1]])
saidas = np.array([0, 0, 0, 1])
pesos = np.array([0.0, 0.0])
taxa_aprendizagem = 0.1

def stepFunction(soma):
    if (soma >= 1):
        return 1
    return 0

def calculaSaida(registro):
    s = registro.dot(pesos)
    return stepFunction(s)

def treinar():
    errototal = 1
    while (errototal != 0):
        errototal = 0
        for i in range(len(saidas)):
            saidacalculada = calculaSaida(np.asarray(entradas[i]))
            erro = abs(saidas[i] - saidacalculada)
            errototal += erro
            for j in range(len(pesos)):
                pesos[j] = pesos[j] + (taxa_aprendizagem * entradas[i][j] * erro)
                print(f'Peso atualizado: {str(pesos[j])}')
        print(f'Total de erros: {str(errototal)}')
        
treinar()
print('Rede neural treinada')
print(calculaSaida(entradas[0]))
print(calculaSaida(entradas[1]))
print(calculaSaida(entradas[2]))
print(calculaSaida(entradas[3]))