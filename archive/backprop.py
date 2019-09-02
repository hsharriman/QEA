import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import copy

def sigmoid(aj):
    #Calculates the sigmoid function using logistic
    return 1 / (1 + np.exp(-aj))

def sigPrime(z):
    return (z * (1 - z))

def sumOfSquareError(netOutput, target):
    #Takes 2 vectors of same dimensions, returns the sum-of-squares error
    return 0.5 * np.sum(np.power(netOutput - target, 2))

def calcAj(weights, prevLayerOut):
    #Calculates the sum of the weights and outputs from previous
    #prevLayerOut and weights must be np arrays
    #weights should be one ROW of the weights matrix
    #prevLayerOut should be a ROW vector
    return np.dot(weights, prevLayerOut)

def calcLayerOutputs(inputVec, weights):
    #define vector of output activations for all units in current layer
    #given matrix of weights, number of nodes per hidden layer,
    #and an input vector of data, propagate forward for one layer
    return sigmoid(np.matmul(weights, inputVec))

def calcDelta(netOutput, target):
    #Takes 2 np arrays, all activations from forward prop and actual target
    #returns delta value for each output neuron
    #assumes netOutput's last entry is activations of output layer
    sigOutk = []
    guesses = netOutput[-1]
    for k in range(len(guesses)):
        sigOutk.append(sigPrime(guesses[k]))
    return np.asarray(sigOutk) * (guesses - target)


def forwardprop(inputVec, weights):
    #calculates outputs for all neurons in network
    activations = [inputVec]

    #dynamically calculates activations for all nodes in network
    for layer in range(len(weights)):
        weightsInLayer = weights[layer]

        #calculates all activations for current layer
        layerActivations = calcLayerOutputs(activations[layer], weightsInLayer[:, :])

        #adds resulting np array to list
        activations.append(layerActivations)
    #should be number of layers * j (neurons per layer) activations
    return activations

def backprop(activations, weights, deltaK):
    #calculates deltas for hidden units
    allDeltas = [deltaK]

    # The index goes from n-2 to 0.
    # The deltas vectors are indexed like the list of weight matrices.
    # The deltas for n-1 (the last weight matrix) are already computed.
    for layer in range(len(weights) - 2, -1, -1):
        #fetches activations from previous layer
        zj = activations[layer + 1] #Using the activations of the next layer
        weightsInLayer = weights[layer + 1] #and the weights of the next layer

        #one matrix-vector multiplication and one element-wise vector multiplication
        deltaJs = np.matmul(np.transpose(weightsInLayer), allDeltas[0]) * sigPrime(zj)

        #builds list of deltas to have same indexing as weights matrix
        allDeltas.insert(0, deltaJs)

    return allDeltas

def calcGradient(learningRate, deltas, activations, weights):
    #for each layer, for each weight @ every node, calculate the gradient
    gradient = []
    for layer in range(len(weights) -1, -1, -1):
        deltaKs = deltas[layer]
        outJs = activations[layer]
        weightDims = weights[layer]

        #pre-allocate array to fill with gradient values. Annoyingly defined by [0] * columns * rows
        weightsInLayer = np.array([[0.] * weightDims.shape[1]] * weightDims.shape[0])

        for k in range(len(deltaKs)):
            for j in range(len(outJs)):
                #Note: [k, j] row by column. Think of matrix multiplication.
                weightsInLayer[k,j] =  learningRate * deltaKs[k] * outJs[j]

        gradient.insert(0, weightsInLayer)
    return gradient

def updateWeights(learningRate, deltas, activations, weights):
    #calculate the gradient
    gradComponents = calcGradient(learningRate, deltas, activations, weights)

    #update all of the weights based on the gradient
    newWeights = []
    for layer in range(len(weights)):
        weightsInLayer = weights[layer]
        gradInLayer = gradComponents[layer]

        #subtract the gradient components from the weights
        newWeights.append(weightsInLayer - gradInLayer)
    return newWeights

def gradientCheck(weights, inputVec, target):
    # weights is the list of weight matrices
    # inputVec and target is a single training pair

    # Compute error and gradient with current weights
    activations = forwardprop(inputVec, weights)
    deltaK = calcDelta(activations, target)
    deltas = backprop(activations, weights, deltaK)
    gradComponents = calcGradient(1, deltas, activations, weights)
    error = sumOfSquareError(activations[-1], target)

    # Check each component of the gradient via numerical differentiation
    # WARNING: Very slow
    eps = 1e-9
    for layer in range(len(weights)):
        for j in range(weights[layer].shape[0]):
            for k in range(weights[layer].shape[1]):
                # Add / subtract a small number to the current weight
                weights1 = copy.deepcopy(weights)
                weights2 = copy.deepcopy(weights)
                weights1[layer][j, k] += eps
                weights2[layer][j, k] -= eps

                # Compute the new error
                activations1 = forwardprop(inputVec, weights1)
                activations2 = forwardprop(inputVec, weights2)
                error1 = sumOfSquareError(activations1[-1], target)
                error2 = sumOfSquareError(activations2[-1], target)

                # Check gradient component vs central difference
                numGrad = (error1 - error2) / (2*eps)
                grad = gradComponents[layer][j, k]
                if (abs(grad - numGrad) > 1e-4):
                    print("[", layer, ", ", j, ", ", k, "]")
                    print("ERROR: Wrong gradient component")
                    print("Should be ", numGrad)
                    print("Is ", grad)

def trainNetwork(patterns, learningRate, neuronsPerLayer, targets):
    #Randomly initialize all weights in network
    weights = []
    errors = []
    #Note: Think of multiplying a weight matrix with an activation vector.
    for layer in range(len(neuronsPerLayer) - 1):
        weights.append(np.random.randn(neuronsPerLayer[layer + 1], neuronsPerLayer[layer]))

    #Initialize error plot
    fig, ax = plt.subplots()

    #Loop through all input vectors, train the network, plot the error
    for p in range(len(patterns)):
        #PS: Enable this to check if the forward and backward pass are still consistent.
        #    Warning, it's very slow. Use a small network.
        #gradientCheck(weights, patterns[p], targets[p])

        #forward pass through network, find all activations
        activations = forwardprop(patterns[p], weights)

        #calculate deltas at output layer
        deltaK = calcDelta(activations, targets[p])

        #calculate rest of deltas with backpropagation
        allDeltas = backprop(activations, weights, deltaK)


        #update weights
        weights = updateWeights(learningRate, allDeltas, activations, weights)

        #calculate network error
        error = sumOfSquareError(activations[-1], targets[p])

        #add to list of errors
        errors.append(error)

    print(len(allDeltas), allDeltas[1].shape)
    #plot the error
    ax.plot(errors, marker= '.')
    ax.set(xlabel = 'Iteration', ylabel= 'Sum of Squared Error', title= 'Network Error Graph')
    plt.show()

    print("last guess: ", np.argmax(activations[-1]))
    print("actual vector: ", np.argmax(targets[-1]))
    print("difference: ", (activations[-1] - targets[-1]))
    #Returns the last version of the weights matrix.
    return weights
