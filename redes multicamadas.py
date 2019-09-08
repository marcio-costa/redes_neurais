# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 19:08:01 2019

@author: marci
"""

import numpy as np

def sigmoid(soma):
    return 1 / (1 + np.exp(-soma))


def sigmoidDerivada(sig):
    return sig * (1 - sig)

a = sigmoid(0.5)
b = sigmoidDerivada(a)
print(b)

entradas = np.array([[0,0],
                     [0,1],
                     [1,0],
                     [1,1]])

saidas = np.array([[0],[1],[1],[0]])

pesos0 = np.array([[-0.424, -0.740, -0.961],
                   [0.358, -0.577, -0.469]])
pesos1 = np.array([[-0.017, -0.893, -0.148]])

epocas = 100

for _ in range(epocas):
    camadasEntrada = entradas
    somaSinapse0 = np.dot(camadasEntrada, pesos0)
    camadaOculta = sigmoid(somaSinapse0)

    somaSinapse1 = np.dot(camadaOculta, pesos1)
    camadaSaida = sigmoid(somaSinapse1)

    erroCamadaSaida = saidas - camadaSaida
    mediaAbsoluta = np.mean(np.abs(erroCamadaSaida))