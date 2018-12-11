import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import copy

def sigmoid(aj):
    #Calculates the sigmoid function using logistic
    return 1 / (1 + np.exp(-aj))

def softmax(s):
    #Calculates the softmax of the output layer
    return np.exp(s) / np.sum(np.exp(s))

def sigPrime(z):
    return (z * (1 - z))

def calcAj(weights, prevLayerOut):
    #Calculates the sum of the weights and outputs from previous
    #prevLayerOut and weights must be np arrays
    #weights should be one ROW of the weights matrix
    #prevLayerOut should be a ROW vector
    return np.dot(weights, prevLayerOut)

def calcLayerOutputs(inputVec, weights):
    #given matrix of weights, number of nodes per hidden layer,
    #and an input vector of data, propagate forward for one layer
    #define vector of output activations for all units in current layer
    return sigmoid(np.matmul(weights, inputVec))

def crossEntError(netOutput, target):
    #Takes 2 vectors of same dimensions, returns the sum-of-squares error
    clip = 1e-10
    return -1 * np.sum(target * np.log(np.maximum(netOutput, clip)))

def sumOfSquareErr(output, target):
    return 0.5 * np.sum(np.power(output - target, 2))

def calcDelta(netOutput, target):
    #Takes 2 np arrays, all activations from forward prop and actual target
    #returns delta value for each output neuron
    #assumes netOutput's last entry is activations of output layer
    guesses = netOutput[-1]
    return (guesses - target)

def forwardprop(inputVec, weights):
    #calculates outputs for all neurons in network
    activations = [inputVec]

    #dynamically calculates activations for all nodes in network
    for layer in range(len(weights)):
        weightsInLayer = weights[layer]

        #Apply sigmoid activation function to hidden units
        if layer < (len(weights) - 1):
            #calculates all activations for current layer
            layerActivations = calcLayerOutputs(activations[layer], weightsInLayer[:, :])

        #Apply softmax to output units
        if layer == (len(weights) - 1):
            s = np.matmul(weightsInLayer[:,:], activations[layer])
            layerActivations = softmax(s)

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
    numBroken = 0
    weights1 = weights
    weights2 = copy.deepcopy(weights)

    # Compute error and gradient with current weights
    activations = forwardprop(inputVec, weights)
    deltaK = calcDelta(activations, target)
    deltas = backprop(activations, weights, deltaK)
    gradComponents = calcGradient(1, deltas, activations, weights)

    # Check each component of the gradient via numerical differentiation
    # WARNING: Very slow
    eps = 1e-9
    for layer in range(len(weights)):
        for j in range(weights[layer].shape[0]):
            for k in range(weights[layer].shape[1]):
                # Add / subtract a small number to the current weight
                weights1[layer][j, k] += eps
                weights2[layer][j, k] -= eps

                # Compute the new error
                activations1 = forwardprop(inputVec, weights1)
                activations2 = forwardprop(inputVec, weights2)
                error1 = crossEntError(activations1[-1], target)
                error2 = crossEntError(activations2[-1], target)

                #remove the eps value
                weights1[layer][j, k] -= eps
                weights2[layer][j, k] += eps

                # Check gradient component vs central difference
                numGrad = (error1 - error2) / (2*eps)
                grad = gradComponents[layer][j, k]
                #print(abs(grad -numGrad))
                if (abs(grad - numGrad) > 1e-4):
                    numBroken += 1
                    print("[", layer, ", ", j, ", ", k, "]")
                    print("ERROR: Wrong gradient component")
                    print("Should be: ", numGrad)
                    print("Is: ", grad)
    return numBroken

def trainNetwork(patterns, learningRate, neuronsPerLayer, targets, *weights):
    errors = []
    numCorrect = 0
    numCorrect2 = 0

    #Check to see if weights is passed, if so, index so it doesn't become a tuple
    if weights:
        weights = weights[0]

    #Else randomly initialize all weights in network
    if not weights:
        weights = []
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
        error = crossEntError(activations[-1], targets[p])

        #add to list of errors
        errors.append(error)

        #Did the network guess right?
        if np.argmax(activations[-1]) == np.argmax(targets[p]):
            numCorrect += 1

        #Other check
        if int(error) == 0:
            numCorrect2 += 1

        if (p % 10) == 0:
            loss = sumOfSquareErr(errors, np.array([0.] * np.shape(errors)[0])) / len(errors)
            acc = (numCorrect / (p + 1)) * 100
            print("[iter: ", p, " loss: ",  loss, " right: ", numCorrect, " acc: ", acc, "%]")

    #calculate accuracy
    acc = (numCorrect / len(patterns)) * 100

    #plot the error
    #ax.plot(errors, marker= '.')
    #ax.set(xlabel = 'Iteration', ylabel= 'Cross Entropy Error', title= 'Network Error Graph')
    #plt.show()

    print("last guess: ", np.argmax(activations[-1]))
    print("actual vector: ", np.argmax(targets[-1]))
    print("difference: ", (activations[-1] - targets[-1]))
    print("accuracy: ", acc, "%")
    print("number of correct guesses: ", numCorrect)
    print("# times Network Error was zero: ", numCorrect2)
    return weights

def visualize(patterns, targets, goodOnes):

    for i in range(len(goodOnes)):
        # The first column is the label
        label = np.argmax(targets[goodOnes[i][0]])
        guess = goodOnes[i][1]

        # The rest of columns are pixels
        pixels = patterns[goodOnes[i][0]] * 255

        # Make those columns into a array of 8-bits pixels
        # This array will be of 1D with length 784
        # The pixel intensity values are integers from 0 to 255
        pixels = pixels.astype(np.uint8)

        # Reshape the array into 28 x 28 array (2-dimensional array)
        pixels = pixels.reshape((28, 28))

        #plot
        plt.title('Label is {label}, guess was {guess}'.format(label=label, guess=guess))
        plt.imshow(pixels, cmap='gray')
        plt.show()

def testNetwork(patterns, targets, weights, good=True, showNums=False):
    errors = []
    numCorrect = 0
    numCorrect2 = 0
    goodOnes = []
    badOnes = []

    #Initialize error plot
    #fig, ax = plt.subplots()

    #Loop through all input vectors, train the network, plot the error
    for p in range(len(patterns)):

        #forward pass through network, find all activations
        activations = forwardprop(patterns[p], weights)

        #calculate network error
        error = crossEntError(activations[-1], targets[p])

        #add to list of errors
        errors.append(error)

        #Did the network guess right?
        if np.argmax(activations[-1]) == np.argmax(targets[p]):
            numCorrect += 1
            goodOnes.append((p, np.argmax(activations[-1])))

        if int(error) >= 1:
            badOnes.append((p, np.argmax(activations[-1])))

    #calculate accuracy
    acc = (numCorrect / len(patterns)) * 100

    #plot the error
    # ax.plot(errors, marker= '.')
    # ax.set(xlabel = 'Iteration', ylabel= 'Cross Entropy Error', title= 'Test Set Error')
    # plt.show()
    #
    # print("difference: ", (activations[-1] - targets[-1]))
    # print("accuracy: ", acc, "%")
    # print("number of correct guesses: ", numCorrect)

    #visualize the numbers that were guessed correctly
    if showNums:
        if good:
            visualize(patterns, targets, goodOnes)
        if not good:
            visualize(patterns, targets, badOnes)

    return np.average(np.asarray(errors)), acc

def make_dataset(trainfile, testfile):
    #Takes name of training set and test set, breaks into input vectors and targets
    #Initializes arrays that will contain data set
    imgtr = []
    tartr = []
    imgte = []
    tarte = []
    numOutputs = 10

    #Opens training data set
    fd = open(trainfile, 'r')
    loaded = fd.readlines()
    fd.close()

    fdtr = open(testfile, 'r')
    loadedtr = fdtr.readlines()
    fdtr.close()

    #Pre-process the training data set
    for line in loaded:
        #reset options
        opt = np.array([0.] * numOutputs)

        #split off classification from input array
        linebits = line.split(',')
        imgtr.append(np.asfarray(linebits[1:])/255)  #currently (784,1) on intensity scale 0-1reshape((28,28))

        #classify from 0-9
        opt[int(linebits[0])] = 1.
        tartr.append(opt)
    print(len(imgtr), len(imgtr[0]), len(tartr), tartr[-1], linebits[0])

    #Same thing for test data set
    for line in loadedtr:
        opt = np.array([0.] * numOutputs)

        linebits = line.split(',')
        imgte.append(np.asfarray(linebits[1:])/255)

        opt[int(linebits[0])] = 1.
        tarte.append(opt)
    print(len(imgte), len(tarte))
    print(len(imgtr), len(tartr))
    print(len(imgtr[0]), len(tartr[0]))
    return imgtr, tartr, imgte, tarte

if __name__ == '__main__':
    imgtr, tartr, imgte, tarte = make_dataset("mnist_train.csv", "mnist_test.csv")
    lastWeights = trainNetwork(imgtr, .05, [784, 625, 625, 10], tartr)
    for i in range(1):
        lastWeights = trainNetwork(imgtr, .05, [784, 625, 625, 10], tartr, lastWeights)
    numberCorrect = testNetwork(imgte, tarte, lastWeights, False, True)
