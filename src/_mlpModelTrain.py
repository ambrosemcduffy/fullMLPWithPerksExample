
import numpy as np

import _mlpModel as model
import dataProcessor
from dataEvaluation import getAccuracy


def train(X, Y, XTest, YTest, layerDims, learningRate, numIteration, useTanh, useL2=False, dropOut=False, lambd=0.7, keepProb=0.5, earlyStop=False):
    parameters = model.initializeParameters(layerDims)
    m = X.shape[1]

    accTrain_prev = 0
    accTest_prev = 0
    velocity = model.initializeAdam(parameters)
    #velocity = model.initializeMomentum(parameters)
    for i in range(numIteration):

        if dropOut:
            AL, cache = model.forward_Dropout(X, parameters, useSoftmax=False, UseSigmoid=True, useTanh=False,  keepProb=0.5)
        else:
            AL, cache = model.forward(X, parameters, useSoftmax=True, UseSigmoid=False, useTanh=False)
        
        if useL2:
            cost = model.crossEntropy(m, Y, AL) + model.l2Loss(parameters, lambd=lambd)
        else:
            #cost = model.crossEntropy(m, Y, AL)
            cost = model.softmaxCost(m, Y, AL)
            #cost = model.meanSquaredError(Y, AL)
        
        if dropOut:
            grads = model.backpropDrop(AL, Y, cache, parameters, m, useTanh=useTanh, lambd=lambd, useL2=useL2, keepProb=keepProb)
        else:
            grads = model.backprop(AL, Y, cache, parameters, m, useTanh=useTanh, lambd=lambd, useL2=useL2)
        
        #parameters = model.optimization(parameters, grads, learningRate)
        parameters, velocity = model.optimizationWithAdam(parameters, grads, learningRate, velocity, i+1)

        #parameters, velocity = model.optimizationWithMomentum(parameters, grads, learningRate, velocity)
        #parameters, velocity = model.optimizationWithRMSPROP(parameters, grads, learningRate, velocity)
        if i % 100 == 0:
            
            alTest, _ = model.forward(XTest, parameters, True, useTanh)
            costTest = model.softmaxCost(YTest.shape[1], YTest, alTest)
            accTrain = getAccuracy(AL, yTrain)
            accTest = getAccuracy(alTest, yTest)

            if accTrain >= 80 and accTest >= 80:
                if accTrain > accTrain_prev and accTest > accTest_prev:
                    print("Saving out weights..")
                    np.savez("weights/epoch3000.npz", **parameters)
                if earlyStop:
                    break
                accTrain_prev = accTrain
                accTest_prev = accTest
            print("epoch: {} -- trainsetError {} --testError {} -- train acc {} -- testSet Acc {}".format(i, cost, costTest, accTrain, accTest))
    return parameters


# answer = input("Would you like to update Dataset?\n")
# if answer.lower() == "yes":
#     images, labels = dataProcessor.importImages("images")
#     dataProcessor.saveData(images, labels)

# obtaining our datset
xTrain, yTrain, xTest, yTest = dataProcessor.getDataset(flattenImages=True)

# Obtaining dimensions and Outputting
if len(xTrain.shape) == 3:
    nx, m = xTrain.shape
    ny = yTrain.shape[0]
    
    print("\nNumber of features: {}".format(nx))
    print("Number of training Examples: {}\n".format(m))
else:
    print("Training Set Dimensions: {}".format(xTrain.shape))
    print("Training labels Dimensions: {}\n".format(yTrain.shape))
    print("Test Set Dimensions: {}".format(xTest.shape))
    print("Test labels Dimensions: {}".format(yTest.shape))


# Transposing labels
yTrain = model.toOneHot(yTrain, 5).T
yTest = model.toOneHot(yTest, 5).T

# Training our model
parameters = train(xTrain, yTrain, xTest, yTest, [xTrain.shape[0], 32, 128, 1024, 5],  0.001, 3000, useTanh=False, useL2=False, dropOut=False, lambd=0.7, keepProb=0.7, earlyStop=False)
