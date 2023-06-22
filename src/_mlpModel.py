import numpy as np


def crossEntropy(m, y, yhat):
    epsilon = 1e-10
    return (-1.0/m) * np.sum(y*np.log(yhat+epsilon) + ((1-y) * np.log(1-yhat+epsilon)))


def softmaxCost(m, y, yhat):
    epsilon = 1e-10
    yhat = np.clip(yhat, epsilon, 1-epsilon)
    loss = (-1.0/m) * np.sum(y * np.log(yhat))
    return loss


def l2Loss(parameters, lambd=0.7):
    L = len(parameters) // 2
    reg_loss = 0
    for l in range(1, L+1):
        w = parameters["w"+str(l)]
        reg_loss += np.sum(w**2)
    l2Loss = lambd * reg_loss
    return l2Loss


def meanSquaredError(y, yhat):
    m = y.shape[1]
    return (1.0/m) * (np.sum((y-yhat)**2))


def optimization(parameters, grads, learningRate):
    L = len(parameters)//2
    for l in range(1, L+1):
        parameters["w"+str(l)] = parameters["w"+str(l)] - learningRate * grads["dw"+str(l)]
        parameters["b"+str(l)] = parameters["b"+str(l)] - learningRate * grads["db"+str(l)]
    return parameters


def optimizationWithAdam(parameters, grads, learningRate, velocity, t):
    L = len(parameters)//2
    vdw_prev, vdb_prev, sdw_prev, sdb_prev = velocity


    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

        

    for l in range(1, L+1):
        vdw = (beta1 * vdw_prev["dw"+str(l)]) + (1-beta1) * grads["dw"+str(l)]
        vdb = (beta1 * vdb_prev["db"+str(l)]) + (1-beta1) * grads["db"+str(l)]

        sdw = (beta2 * sdw_prev["dw"+str(l)]) + (1-beta2) * (grads["dw"+str(l)] **2) 
        sdb = (beta2 * sdb_prev["db"+str(l)]) + (1-beta2) * (grads["db"+str(l)] **2)


        # bias correction
        #vdw = vdw/(1-beta1**t) + epsilon
        #vdb = vdb/(1-beta1**t) + epsilon
        #sdw = sdw/(1-beta2**t) + epsilon
        #sdb = sdb/(1-beta2**t) + epsilon

        parameters["w"+str(l)] = parameters["w"+str(l)] - learningRate * (vdw/(np.sqrt(sdw + epsilon)))
        parameters["b"+str(l)] = parameters["b"+str(l)] - learningRate * (vdb/(np.sqrt(sdb + epsilon)))

        
        
        vdw_prev["dw"+str(l)] = vdw
        vdb_prev["db"+str(l)] = vdb

        sdw_prev["dw"+str(l)] = sdw
        sdb_prev["db"+str(l)] = sdb

    return parameters, (vdw_prev, vdb_prev, sdw_prev, sdb_prev)


def optimizationWithRMSPROP(parameters, grads, learningRate, velocity):
    L = len(parameters)//2
    vdw_prev = {}
    vdb_prev = {}
    beta = 0.9
    epsilon = 1e-8

    vdw_prev, vdb_prev = velocity

    for l in range(1, L+1):
        vdw = (beta * vdw_prev["dw"+str(l)]) + (1-beta) * (grads["dw"+str(l)] **2)
        vdb = beta * vdb_prev["db"+str(l)] + (1-beta) * (grads["db"+str(l)] **2)

        parameters["w"+str(l)] = parameters["w"+str(l)] - learningRate * (grads["dw"+str(l)]/np.sqrt(vdw+epsilon))
        parameters["b"+str(l)] = parameters["b"+str(l)] - learningRate * (grads["db"+str(l)]/np.sqrt(vdb+epsilon))
        vdw_prev["dw"+str(l)] = vdw
        vdb_prev["db"+str(l)] = vdb

    return parameters, velocity


def optimizationWithMomentum(parameters, grads, learningRate, velocity):
    L = len(parameters)//2
    beta = 0.9
    vdw_prev, vdb_prev = velocity
    for l in range(1, L+1):
        vdw = (beta * vdw_prev["dw"+str(l)]) + (1-beta) * grads["dw"+str(l)]
        vdb = beta * vdb_prev["db"+str(l)] + (1-beta) * grads["db"+str(l)]
        
        # updating weights
        parameters["w"+str(l)] = parameters["w"+str(l)] - learningRate * vdw
        parameters["b"+str(l)] = parameters["b"+str(l)] - learningRate * vdb
        
        vdw_prev["dw"+str(l)] = vdw
        vdb_prev["db"+str(l)] = vdb

    return parameters, velocity


def initializeAdam(parameters):
    vdw_prev = {}
    vdb_prev = {}

    sdw_prev = {}
    sdb_prev = {}
    L = len(parameters)//2
    for l in range(1, L+1):
        vdw_prev["dw"+str(l)] = np.zeros_like(parameters["w"+str(l)])
        vdb_prev["db"+str(l)] = np.zeros_like(parameters["b"+str(l)])

        sdw_prev["dw"+str(l)] = np.zeros_like(parameters["w"+str(l)])
        sdb_prev["db"+str(l)] = np.zeros_like(parameters["b"+str(l)])
    return (vdw_prev, vdb_prev, sdw_prev, sdb_prev)


def initializeMomentum(parameters):
    vdw_prev = {}
    vdb_prev = {}

    L = len(parameters)//2
    for l in range(1, L+1):
        vdw_prev["dw"+str(l)] = np.zeros_like(parameters["w"+str(l)])
        vdb_prev["db"+str(l)] = np.zeros_like(parameters["b"+str(l)])

    return (vdw_prev, vdb_prev)


def backprop(AL, Y, caches, parameters, m, useTanh=False, lambd=0.7, useL2=False):
    L = len(parameters) // 2
    grads = {}
    dZL = 2*(AL-Y)

    keep_prob = 0.5

    A_prev, ZL, _ = caches[L-1]
    if useL2:
        grads['dw' +str(L)] = (1.0/m) * np.dot(dZL, A_prev.T) + (lambd/m) * parameters["w"+str(L)]
    else:
        grads['dw' +str(L)] = (1.0/m) * np.dot(dZL, A_prev.T)
    
    grads['db' + str(L)] = (1.0/m)  * np.sum(dZL, axis=1, keepdims=True)
    dZ_prev = dZL


    for l in reversed(range(1, L)):
        A_prev, Z, A = caches[l-1]
        dAL = np.dot(parameters["w"+str(l+1)].T, dZ_prev)
        
        if l != 1:
            if useTanh:
                dZ = np.multiply(dAL, 1-np.power(A, 2))
            else:
                dZ = np.where(A > 0, dAL, 0)
        else:
            dZ = dAL
        
        if useL2:
            grads["dw"+str(l)] = (1.0/m) * np.dot(dZ, A_prev.T) + (lambd/m) * parameters["w"+str(l)]
        else:
            grads["dw"+str(l)] = (1.0/m) * np.dot(dZ, A_prev.T)
        grads['db'+str(l)] = (1.0/m) * np.sum(dZ, axis=1, keepdims=True)

        dZ_prev = dZ
    return grads


def backpropDrop(AL, Y, caches, parameters, m, useTanh=False, lambd=0.7, useL2=False, keepProb=0.5):
    L = len(parameters) // 2
    grads = {}
    dZL = AL-Y

    A_prev, ZL, AL, D = caches[L-1]
    if useL2:
        grads['dw' +str(L)] = (1.0/m) * np.dot(dZL, A_prev.T) + (lambd/m) * parameters["w"+str(L)]
    else:
        grads['dw' +str(L)] = (1.0/m) * np.dot(dZL, A_prev.T)
    
    grads['db' + str(L)] = (1.0/m)  * np.sum(dZL, axis=1, keepdims=True)
    dZ_prev = dZL


    for l in reversed(range(1, L)):
        A_prev, Z, A, D = caches[l-1]
        dAL = np.dot(parameters["w"+str(l+1)].T, dZ_prev)
        dAL = (dAL * D[l-1])/keepProb
        
        if l != 1:
            if useTanh:
                dZ = np.multiply(dAL, 1-np.power(A, 2))
            else:
                dZ = np.where(A > 0, dAL, 0)
        else:
            dZ = dAL
        
        if useL2:
            grads["dw"+str(l)] = (1.0/m) * np.dot(dZ, A_prev.T) + (lambd/m) * parameters["w"+str(l)]
        else:
            grads["dw"+str(l)] = (1.0/m) * np.dot(dZ, A_prev.T)
        grads['db'+str(l)] = (1.0/m) * np.sum(dZ, axis=1, keepdims=True)

        dZ_prev = dZ
    return grads


def forward(X, parameters, useSoftmax=False, UseSigmoid=True, useTanh=False):
    A_prev = X
    L = len(parameters) // 2
    caches = []
    for l in range(1, L):
        Z = np.dot(parameters["w"+str(l)], A_prev) + parameters["b"+str(l)]
        if useTanh:
            A = np.tanh(Z)
        else:
            A = relu(Z)
        cache = (A_prev, Z, A)
        caches.append(cache)
        A_prev = A
    
    ZL = np.dot(parameters["w"+str(L)], A_prev) + parameters["b"+str(L)]

    if useSoftmax:
        AL = softmaxActivation(ZL)
    
    if UseSigmoid:
        AL = sigmoid(ZL)
    else:
        AL = ZL
    cache = (A_prev, ZL, AL)
    caches.append(cache)

    return AL, caches


def forward_Dropout(X, parameters, useSoftmax=False, UseSigmoid=True, useTanh=False, keepProb=0.5):
    A_prev = X
    L = len(parameters) // 2
    caches = []

    D = np.random.rand(A_prev.shape[0], A_prev.shape[1])
    D = D < keepProb
    A_prev = (A_prev * D)/keepProb
    for l in range(1, L):
        Z = np.dot(parameters["w"+str(l)], A_prev) + parameters["b"+str(l)]
        if useTanh:
            A = np.tanh(Z)
        else:
            A = relu(Z)
            D = np.random.rand(Z.shape[0], Z.shape[1])
            D = D < keepProb
            A = (A * D)/keepProb

        cache = (A_prev, Z, A, D)
        caches.append(cache)
        A_prev = A
    
    ZL = np.dot(parameters["w"+str(L)], A_prev) + parameters["b"+str(L)]

    if useSoftmax:
        AL = softmaxActivation(ZL)
    
    if UseSigmoid:
        AL = sigmoid(ZL)
    else:
        print("using linear probs")
        AL = ZL
    
    cache = (A_prev, ZL, AL, D)
    caches.append(cache)

    return AL, caches


def relu(X):
    return np.maximum(0, X)


def softmaxActivation(z):
    epsilon = 1e-8
    t = np.exp(z)
    s = np.sum(t, axis=0, keepdims=True)
    s += epsilon
    softmax = t / (s)
    return softmax


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def initializeParameters(layerDims, usingTanh=False):
    L = len(layerDims)
    parameters = {}
    for l in range(1, L):
        parameters["w"+str(l)] = np.random.randn(layerDims[l], layerDims[l-1]) * xavierInitialization(layerDims, l, usingTanh=usingTanh)
        parameters["b"+str(l)] = np.random.randn(layerDims[l], 1)

    return parameters


def xavierInitialization(layerDims, l, usingTanh=False):
    if usingTanh:
        return np.sqrt(1/layerDims[l-1])
    else:
        return np.sqrt(2/layerDims[l-1])


def zScoreNomalization(X):
    m = X.shape[1]
    mue = (1.0/m) * np.sum(X, axis=0)
    X = X - mue
    sigmaSquared = (1.0/m) * np.sum(X**2, axis=0)
    return X/np.sqrt(sigmaSquared)


def learnRateDecay(learnRate, decayRate, epoch):
    return (1.0/(1.0+decayRate+epoch)) * learnRate


def toOneHot(labels, num_classes):
    one_hot = np.zeros((labels.shape[1], num_classes))
    for i in range(labels.shape[1]):
        label = labels[:, i]
        if np.max(label) >= num_classes:
            raise ValueError("Label value exceeds number of classes")
        one_hot[i][label] = 1
    return one_hot