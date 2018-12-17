# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 13:54:45 2018
Python 3.7
@author: Priyansh
"""

import numpy as np
from load_mnist import mnist
from load_mnist import one_hot
import math



def relu(Z):
    
    return np.maximum(0, Z)



def reluGradient(Z):
    
    return (Z > 0).astype(int)



def softmax(Z):
    
    return (np.exp(Z) / (np.sum(np.exp(Z), axis=0, keepdims=True)))



def meProp(dZ, k):
    
    dZabs = np.abs(dZ)
    dZsort = np.argsort(dZabs, axis=0)
    for i in range(dZ.shape[1]):
        dZ[dZsort[0:(dZ.shape[0]-k),i],i] = 0.0
    
    return dZ, dZsort



def initParameter(lDim):
    
    parameters = {}           

    for l in range(1, len(lDim)):
        parameters['W' + str(l)] = np.random.randn(lDim[l], lDim[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((lDim[l], 1))
        
    return parameters



def initParameterAdam(parameters):
    
    L = len(parameters) // 2
    v = {}
    s = {}
    
    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
        v["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])

        s["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
        s["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])
    
    return v, s



def initRandomMiniBatch(X, Y, miniBatchSize):
    
    miniBatch = []
    perm = np.random.permutation(X.shape[1])
    tempX = X[:,perm]
    tempY = Y[:,perm]
    nMiniBatch = math.floor(X.shape[1]/miniBatchSize)
    for i in range(nMiniBatch):
        miniBatchX = tempX[:, i * miniBatchSize:(i + 1) * miniBatchSize]
        miniBatchY = tempY[:, i * miniBatchSize:(i + 1) * miniBatchSize]
        miniBatch.append((miniBatchX, miniBatchY))
    return miniBatch



def lForwardProp(Aprev, cache, l, W, b, activation):
    
    if activation == 'relu':
        Z = np.dot(W, Aprev) + b
        A = relu(Z)
        
    elif activation == 'softmax':
        Z = np.dot(W, Aprev) + b
        A = softmax(Z)
    
    cache['Z' + str(l)] = Z
    cache['A' + str(l)] = A    
    
    return A, cache



def forwardPropagation(X, parameters):

    cache = {}
    A = X
    L = len(parameters) // 2        
    cache['A' + str(0)] = X       
    
    for l in range(1, L):
        Aprev = A 
        A, cache = lForwardProp(Aprev, cache, l,
                                parameters['W' + str(l)], 
                                parameters['b' + str(l)], 
                                activation='relu')

    AL, cache = lForwardProp(A, cache, L,
                             parameters['W' + str(L)], 
                             parameters['b' + str(L)],
                             activation='softmax')
            
    return AL, cache




def costNN(AL, Y):
    
    return (np.asscalar(- ( np.sum( np.sum( np.multiply(Y, np.log(AL)) , axis=0, keepdims=True) , axis=1, keepdims=True) ) / Y.shape[1] ))



def lBackProp(dA, parameters, cache, l, m, k):
    
    dAprev = []
    dZ = np.multiply(dA, reluGradient(cache['Z' + str(l+1)]))
    
    #meProp
    dZmeProp, _ = meProp(dZ, k)
    dZ = dZmeProp

    
    dW = np.dot(dZ, cache['A' + str(l)].T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    if l!=0:
        dAprev = np.dot(parameters['W' + str(l+1)].T, dZ)
    return dW, db, dAprev



def backPropagation(AL, Y, parameters, cache, k):

    gradients = {}
    m = Y.shape[1]
    L = len(parameters) // 2
    
    dZL = AL - Y
    gradients['dW' + str(L)] = np.dot(dZL, cache['A' + str(L-1)].T) / m
    gradients['db' + str(L)] = np.sum(dZL, axis=1, keepdims=True) / m
    gradients['dA' + str(L-1)] = np.dot(parameters['W' + str(L)].T, dZL)
    
    for l in reversed(range(L-1)):
        dW, db, dAprev = lBackProp(gradients['dA' + str(l+1)], parameters, cache, l, m, k[l])
        gradients["dW" + str(l + 1)] = dW
        gradients["db" + str(l + 1)] = db
        gradients["dA" + str(l)] = dAprev

    return gradients



def optimizerAdam(parameters, gradients, alpha, v, s, tAdam):
    
    L = len(parameters) // 2   
              
    vCorrected = {}                    
    sCorrected = {}             

    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-08       
    
    for l in range(L):

        v["dW" + str(l + 1)] = beta1 * v["dW" + str(l + 1)] + (1 - beta1) * gradients['dW' + str(l + 1)]
        v["db" + str(l + 1)] = beta1 * v["db" + str(l + 1)] + (1 - beta1) * gradients['db' + str(l + 1)]

        s["dW" + str(l + 1)] = beta2 * s["dW" + str(l + 1)] + (1 - beta2) * np.power(gradients['dW' + str(l + 1)], 2)
        s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + (1 - beta2) * np.power(gradients['db' + str(l + 1)], 2)

        vCorrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - np.power(beta1, tAdam))
        vCorrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - np.power(beta1, tAdam))

        sCorrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1 - np.power(beta2, tAdam))
        sCorrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - np.power(beta2, tAdam))

        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - alpha * vCorrected["dW" + str(l + 1)] / np.sqrt(s["dW" + str(l + 1)] + epsilon)
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - alpha * vCorrected["db" + str(l + 1)] / np.sqrt(s["db" + str(l + 1)] + epsilon)

    return parameters, v, s  
    




def predictClass(X, parameters):
    
    AL, _ = forwardPropagation(X, parameters)
    
    return AL.argmax(axis=0), AL



def accuracy(Y, Ypred):
    
    return (np.sum(Y == Ypred) / Y.shape[1])*100



def errorDev(Xdev, Ydev, parameters):
    
    AL, _ = forwardPropagation(Xdev, parameters)
    
    return costNN(AL, Ydev)
    


def neuralNetwork(X, Y, lDim, nEpoch, alpha, Xdev, Ydev, k, miniBatchSize):
        
    parameters = initParameter(lDim)
    
    v, s = initParameterAdam(parameters)
   
    costTrain = []
    costDev = []
    tAdam = 0
    
    for i in range(nEpoch):
    
        
        miniBatch = initRandomMiniBatch(X, Y, miniBatchSize)
        
        for (miniBatchX, miniBatchY) in miniBatch:
            
            AL, cache = forwardPropagation(miniBatchX, parameters)
            
            costTr = costNN(AL, one_hot(miniBatchY.astype(int), 10))
            
            gradients = backPropagation(AL, one_hot(miniBatchY.astype(int), 10), parameters, cache, k)
                    
            tAdam = tAdam + 1
            parameters, v, s = optimizerAdam(parameters, gradients, alpha, v, s, tAdam)
        
        costTrain.append(costTr)
        
        costD = errorDev(Xdev, one_hot(Ydev.astype(int), 10), parameters)
        
        costDev.append(costD)
        
        Ypred, _ = predictClass(Xdev, parameters)
        
        print((accuracy(Ydev, Ypred)))
        
           
    return parameters, costTrain, costDev


def main():
    
    trainDevX, trainDevY, testX, testY = \
            mnist(noTrSamples=50000,noTsSamples=5000,\
            digit_range=[0,1,2,3,4,5,6,7,8,9],\
            noTrPerClass=5000, noTsPerClass=500) 
    
    perm = np.random.permutation(trainDevX.shape[1])
    trainDevX = trainDevX[:,perm]
    trainDevY = trainDevY[:,perm]
    devX = trainDevX[:,45000:50000]
    devY = trainDevY[:,45000:50000]
    trainX = trainDevX[:,0:45000]
    trainY = trainDevY[:,0:45000]      
    
    
    
    #trainY = one_hot(trainY.astype(int), 10)
    #devY = one_hot(devY.astype(int), 10)
    #testY = one_hot(testY.astype(int), 10)
    
    
    lDim = [784, 500, 10]
    miniBatchSize = 100
    k = [[50],[100]]
    nEpoch = 200
    alpha = 0.001
    costHL = []
    accTest = []
    errDev = []
    errTest = []
    
    for i in range(len(k)):
        print(k[i])
        parameters, costTrain, costDev, t = neuralNetwork(trainX, trainY, lDim, nEpoch, alpha, devX, devY, k[i], miniBatchSize)
        costHL.append((costTrain, costDev))
        errDev.append(costDev[nEpoch-1])
        Ypred, _ = predictClass(devX, parameters)
        accDev.append(accuracy(devY, Ypred))
        Ypred, AL = predictClass(testX, parameters)
        accTest.append(accuracy(testY, Ypred))
        errTest.append(costNN(AL, one_hot(testY.astype(int), 10)))

    
if __name__ == "__main__":
    main()