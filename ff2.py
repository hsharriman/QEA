"""Trains and tests a feed-forward neural network on the MNIST dataset of
hand-written digits using numpy.

Author: Hwei-Shin Harriman
"""
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import copy
import pickle

"""
** Network Architecture **
layer: original image --> fully connected --> fully connected -->  output
weights:      n/a            (625, 784)           (625, 625)      (10, 625)
output:     (784,1)   -->     (1, 625)    -->      (1, 625)   -->  (1, 10)
deltas:       n/a     -->     (625, 1)    -->      (625, 1)   -->  (10, 1 )

    direction of previous layer   <---  --->   direction of next layer
"""

class Data_MNIST:
    """Handles loading and pre-processing of MNIST dataset.

        img: list containing all images of MNIST numbers in set, flattened into
        (784 x 1) numpy arrays with intensity values from 0 -> 1
        tar: list containing all (10 x 1) target vectors (expected outputs) of
        MNIST numbers in set. Each target vector is filled with zeros and one 1,
        where 1's index=the value of the number
        ex) [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.] --> corresponding num is 5
        size: number of images in dataset
    """
    def __init__(self, filename):
        self.img = []
        self.tar = []
        self.size = 0

        #Open training data set
        fd = open(filename, 'r')
        loaded = fd.readlines()
        fd.close()

        numOutputs = 10     #10 possible hand-written digits
        #Pre-process the training data set
        for line in loaded:
            #reset options
            opt = np.array([0.] * numOutputs)

            #split off classification from input array
            linebits = line.split(',')
            #scale intensity values from 0->255 to 0->1
            self.img.append(np.asfarray(linebits[1:])/255)  #has shape (784,1)

            #linebits[0] = digit from 0->9, matches digit drawn in input array
            opt[int(linebits[0])] = 1.
            self.tar.append(opt)
        #set size accordingly
        self.size = len(self.img)

    def printMNIST(self):
        """Used for validation that set was initialized correctly"""

        print("Num entries in set: ", self.size)
        print("Len of input image vector (should = 784): ", len(self.img[0]))
        print("Num outputs in set (should = num entries): ", len(self.tar))
        print("Example target vector: ", self.tar[-1])

    def padSet(self, padsz):
        """Pads images in set with zeros. (Used for a convolutional neural net.
        Function not used in this file)

        padsz: number or rows/cols of 0's to surround each image with
        """
        for b in range(self.size):
            img = self.img[b]
            pads = np.zeros((np.shape(img)[0] + pad * 2, np.shape(img)[1] + pad * 2))
            pads[pad:-pad, pad:-pad] = img
            self.img[b] = pads

class DataSet(Data_MNIST):
    """Helper class to pre-process both training and test data sets

    trainfile: filename containing the training data set
    testfile: filename containing the test data set
    batchsize: number of images per batch
    batchImg: list containing the full training dataset split into lists of batches
    of length batchsize
    batchTar: same as batchImg but sublists contain corresponding target vectors
    """
    def __init__(self, trainfile, testfile, batchsize):
        self.train = Data_MNIST(trainfile)   #training dataset
        self.test = Data_MNIST(testfile)    #testing dataset
        self.batchImg = []
        self.batchTar = []
        self.batchsz = batchsize

    def printSet(self):
        """Utility function, should print:
        784 10
        784 10
        """
        print(len(self.train.img), len(self.train.tar))
        print(len(self.test.img), len(self.test.tar))

    def oneBatch(self, currBatch, shuffIndex):
        """ Utility function to make one batch and add to list of batches
        (self.batchImg and self.batchTar)

        currBatch: what batch is currently being constructed. (If -1, then start
        from end of shuffIndex and add corresponding images backwards to avoid
        uneven division of batches)
        shuffIndex: list containing all possible indices in dataset shuffled
        """
        imgs = []
        targs = []
        for i in range(self.batchsz):
            #special case where some images are left out due to batch size
            if currBatch == -1:
                index = shuffIndex[-(i+1)]
            #normally: add shuffled images to batch
            else:
                index = shuffIndex[(i + self.batchsz*currBatch)]
            imgs.append(self.train.img[index])
            targs.append(self.train.tar[index])
        #add batch to list of batches
        self.batchImg.append(imgs)
        self.batchTar.append(targs)

    def makeBatches(self):
        """Randomly shuffle entire dataset and sort into batches according to
        the chosen batchsize
        """
        #empty previous lists
        self.batchImg = []
        self.batchTar = []

        #shuffle dataset
        shuffIndex = np.random.permutation(self.train.size)

        #loop through shuffled indices to add images/targets to batches
        for batch in range(int(self.train.size // self.batchsz)):
            self.oneBatch(batch, shuffIndex)

        #if dataset not evenly split by batches
        if self.train.size % self.batchsz != 0:
            #tell oneBatch() to add remaining images + some repeats to ensure
            #batches are all the same size
            batch = -1
            self.oneBatch(batch, shuffIndex)

    def forXimgs(self, x, repeats):
        """Utility function to create a batch of duplicates of x training/test
        images. Used for testing.

        x: num images to add to batch
        repeats: num duplicates of each image
        """
        #empty previous lists
        self.batchImg = []
        self.batchTar = []
        for r in range(repeats):
            imgs, targs = [], []
            for i in range(x):
                img = self.train.img[i]
                tar = self.train.tar[i]
                imgs.append(img)
                targs.append(tar)
            self.batchImg.append(imgs)
            self.batchTar.append(targs)

    def visualize(self, patterns, targets):
        """ Plots list of input vectors as hand-written digits

        patterns: list of images to be plotted
        targets: corresponding target vectors
        """
        for i in range(len(patterns)):
            img = patterns[i]
            # Reshape the array into 28 x 28 array (2-dimensional array)
            pixels = img.reshape((28, 28))
            label = np.argmax(targets[i])
            #plot
            plt.title('Label is {label}'.format(label=label))
            plt.imshow(pixels, cmap='gray')
            plt.show()

class Layer:
    """Base class for building layers in neural networks.

    row, col: dimensions for the weights matrix
    weights: (row x col) np array initialized with random floats
    within 1 standard deviation.
    gradient: np array (row x col) to store gradient components for
    a layer
    output: output of network *after* the activation function is applied
    deltas: results of backprop stored in np array
    """
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.weights = np.random.randn(row, col)
        self.gradient = np.zeros((row, col))
        self.output = np.zeros(col)
        self.deltas = np.zeros(row)

    def makeWeights(self):
        """Initializes all vals in weights randomly within 1 stddev"""
        self.weights = np.random.randn(self.row, self.col)

    def printShapes(self):
        """Helper function, prints dimensions of weights, output and deltas"""
        print("weights: ", self.weights.shape, " out: ", self.output.shape, " deltas: ", self.deltas.shape)

    def calcGradient(self, lrate, prev_layer, isFirstLayer):
        """Calculates the gradient of a layer as a result of backpropagation,
        adds to accumulated total gradient (for stochastic gradient descent)

        lrate: small positive learning rate
        prev_layer: Previous Layer object (from forward pass)
        isFirstLayer: bool, caller sets True if prev_layer=original image vector
        """
        #reshape from (col,)-->(1,col)
        if isFirstLayer:
            reshapePrevOut = np.reshape(prev_layer, (1, prev_layer.shape[0]))
        else:
            reshapePrevOut = np.reshape(prev_layer.output, (1, prev_layer.output.shape[0]))

        #reshape from (row,)-->(row,1)
        reshapeDelta = np.reshape(self.deltas, (self.deltas.shape[0], 1))

        #calculate gradient components for single layer, add to total
        adjust = lrate * np.matmul(reshapeDelta, reshapePrevOut)    #(row,col)
        self.gradient += adjust

    def update(self):
        """Updates weights by subtracting accumulated gradient (move in
        direction of greatest descent)"""
        self.weights -= self.gradient

        #reset accumulated gradient
        self.gradient = np.zeros((self.row, self.col))

    def loadWeights(self, layerNumber, restore):
        """Loads weights from pickled file or initializes random weights

        layerNumber: layer of the neural network currently being initialized
        restore: bool, True if weights should be loaded, else initialize random
        weights
        """
        #initialize appropriate filename to load weights
        filename = 'weights' + str(layerNumber)
        try:
            if restore:
                print("restoring weights for layer ", filename)
                we = pickle.load(open(filename, "rb"))
                self.weights = we
            else:
                self.makeWeights()
        except (OSError, IOError) as e:
            print("no weights initialized for layer ", filename)
            self.makeWeights()
            pickle.dump(self.weights, open(filename, "wb"))

    def saveWeights(self, layerNumber):
        """Save weights in pickled file

        layerNumber: layer of neural network currently being saved
        """
        filename = 'weights' + str(layerNumber)
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.weights, f)
        except:
            print("error saving weights at layer ", layerNumber, " filename ",filename)

class FullyConnected(Layer):
    """subclass for fully connected layer (characterized by all neurons from
    previous layer connecting to all neurons in the fully connected layer)

    row: num of rows in weights/gradient matrix
    col: num of cols in weights/gradient matrix
    """
    def __init__(self, row, col):
        super().__init__(row, col)

    def sigmoid(self, mat):
        """Calculates the sigmoid function"""
        return 1 / (1 + np.exp(-mat))

    def sigPrime(self, mat):
        """Calculates the derivative of the sigmoid function"""
        return (mat * (1 - mat))

    def forward(self, inputMat):
        """Calculates the output of the fully connected layer from the forward
        pass

        inputMat: output from the previous layer
        """
        self.output = self.sigmoid(np.matmul(self.weights, inputMat))

    def backprop(self, nextLayer):
        """Calculates deltas as a result of backpropagation. Uses weights and
        deltas from the next layer and output of the current layer to calculate
        the deltas from the current layer.

        nextLayer: layer object
        """
        #use weights and deltas from the next layer
        nextWeights = np.transpose(nextLayer.weights)
        nextDeltas = nextLayer.deltas

        #one matrix multiplication and one elementwise vector multiplication
        #((col, row) x (row, 1)) .* (col, 1) --> (col, 1) output
        self.deltas = np.matmul(nextWeights, nextDeltas) * self.sigPrime(self.output)

class SoftmaxOut(Layer):
    """ subclass for output layer (with softmax activation and cross entropy error)

    row: num of rows in weights/gradient matrix
    col: num of cols in weights/gradient matrix
    """
    def __init__(self, row, col):
        super().__init__(row, col)

    def softmax(self, mat):
        """ Calculate softmax of mat """
        return np.exp(mat) / np.sum(np.exp(mat))

    def crossEntError(self, tar):
        """Calculate cross entropy error between output and target vector

        tar: expected output vector
        """
        clip = 1e-10    #to avoid some cases of division by zero
        return -1 * np.sum(tar * np.log(np.maximum(self.output, clip)))

    def forward(self, inputMat):
        """Calculate output of network from forward pass

        inputMat: output from the previous layer
        """
        self.output = self.softmax(np.matmul(self.weights, inputMat))

    def backprop(self, tar):
        """Calculates delta of output layer

        tar: expected output vector
        """
        self.deltas = self.output - tar

class Network:
    def __init__(self, data, lrate, epochs, neuronsPerLayer):
        """Base class to run a neural network.

        data: DataSet object containing training and test vectors
        lrate: learning rate
        epochs: number of epochs (how many times each image in data set is seen)
        npl: array with number of neurons per layer
        layers: array to be filled with Layer objects
        trainacc, trainloss: lists to be filled with accuracy/loss measurements
        on training data set
        testacc, testloss: lists to be filled with accuracy/loss measurements on
        test data set
        """
        self.data = data    #needs to be initialized outside of class
        self.lrate = lrate
        self.epochs = epochs
        self.npl = neuronsPerLayer
        self.layers = []
        self.trainacc, self.trainloss = [], []
        self.testacc, self.testloss = [], []

    def makeLayers(self):
        """ Initializes layers in network, such that all layers except for the
        last layer are fully-connected. The last layer is set as Softmax output
        layer.
        """
        for i in range(len(self.npl)-2):
            #initialize layers with row=neurons in NEXT layer and col=neurons in
            #CURR layer
            self.layers.append(FullyConnected(self.npl[i+1], self.npl[i]))
        self.layers.append(SoftmaxOut(self.npl[-1], self.npl[-2]))

    def forwardpass(self, curr_img):
        """ Passes input vector through entire network, calculates output of
        each layer

        curr_img: input vector being passed through network.
        """
        #initial case, pass input vec in
        first_layer = self.layers[0]
        first_layer.forward(curr_img)

        #use output from prev layer to calculate current layer output
        for i in range(1, len(self.layers)):
            curr_layer = self.layers[i]
            prev_layer = self.layers[i-1]
            curr_layer.forward(prev_layer.output)

    def backpass(self, curr_tar):
        """ Backpropagate error through network to calculate deltas

        curr_tar: target vector passed to output layer
        """
        #starting by passing in target vec
        last_layer = self.layers[-1]
        last_layer.backprop(curr_tar)
        #use deltas from next layer to calculate current layer delta
        for i in range(len(self.layers) - 2, -1, -1):
            curr_layer = self.layers[i]
            next_layer = self.layers[i+1]
            curr_layer.backprop(next_layer)

    def findAllGradients(self, inputVec):
        """ Calculate the gradient of every layer using outputs and deltas

        inputVec: input vector being passed through network
        """
        #loop through list of layers backwards
        for i in range(len(self.layers)-1, -1, -1):
            #set previous layer
            if i == 0:
                prev_layer = inputVec
                isFirstLayer = True
            else:
                prev_layer = self.layers[i-1]
                isFirstLayer = False
            #set current layer
            curr_layer = self.layers[i]

            #calculate gradient
            curr_layer.calcGradient(self.lrate, prev_layer, isFirstLayer)

class Utils(Network):
    """Class of utility functions for running the neural network."""
    def __init__(self, data, lrate, epochs, neuronsPerLayer):
        super().__init__(data, lrate, epochs, neuronsPerLayer)

    def loadNetWeights(self, restore):
        """loads weights for all layers in the network

        restore: bool, True=load previously saved weights, False=make new weights
        """
        for i in range(len(self.layers)):
            layer = self.layers[i]
            layer.loadWeights(i, restore)

    def saveNetWeights(self):
        """save weights for all layers in the network"""
        for i in range(len(self.layers)):
            layer = self.layers[i]
            layer.saveWeights(i)

    def save(self, itemToSave, savepath):
        """Saves training losses (used to save losses/accuracies from training/
        testing the network)

        itemToSave: an array containing either losses or accuracies
        savepath: name of pickle file to save to
        """
        temp = np.asarray(itemToSave).flatten()
        with open(savepath, 'wb') as f:
            print("Pickling. . .")
            pickle.dump(temp, f)

    def makePlots(self, results, isDebugging):
        """Plots losses and accuracies.

        results: array that should contain:
                [self.trainloss, self.trainacc, self.testloss, self.testacc]
                unless isDebugging=True, in which case it should contain:
                [self.trainloss, self.trainacc]
        isDebugging: bool, True=visualize results of dummy network->only needs
        2 plots. False=visualize results from training the network->needs 4 plots
        """
        #lists used to add information to plots
        titles = ['MNIST Training Loss', 'Average Accuracy During Training',
                'MNIST Test Set Loss', 'Average Accuracy of Testing']
        xlabels = ['Batch Number', 'Batch Number', 'Test Number', 'Test Number']
        ylabels = ['Batch Cross Entropy Error', 'Percent Accuracy Per Batch',
                'Test Cross Entropy Error', 'Percent Accuracy per Test Run']
        colors = ['xkcd:neon blue', 'xkcd:cerulean blue', 'xkcd:violet', 'xkcd:purple']

        #select number of plots to make
        numPlots = 2 if isDebugging else 4
        #create plots
        for i in range(numPlots):
            plt.figure()
            plt.plot(results[i], colors[i], marker='.')
            plt.xlabel(xlabels[i])
            plt.ylabel(ylabels[i])
            plt.title(titles[i])
        plt.show()

    def plotsave(self, savepaths, isDebugging):
        """Save and plot training loss/accuracy, and test loss/accuracy.

        savepaths: list of pickle files to save to"""
        print("Pickling network, losses and accuracies. . .")
        #save weights
        self.saveNetWeights()
        results = [self.trainloss, self.trainacc, self.testloss, self.testacc]
        for i in range(4):
            self.save(results[i], savepaths[i])
        self.makePlots(results, isDebugging=False)

    def printResults(self):
        """ Prints records for a training/testing run. """
        trloss = np.amin(np.asarray(self.trainloss))
        teloss = np.amin(np.asarray(self.testloss))
        tracc = np.amax(np.asarray(self.trainacc))
        teacc = np.amax(np.asarray(self.testacc))
        print("Lowest Recorded Training loss: ", trloss)
        print("Lowest Recorded Test loss: ", teloss)
        print("Highest training accuracy: ", tracc)
        print("Highest test accuracy: ", teacc)

class RunNetwork(Utils):
    """Class to initialize and execute training of the neural network using
    epochs, batches, checkpoints, and plotting at the conclusion.
    """
    def __init__(self, data, lrate, epochs, neuronsPerLayer):
        super().__init__(data, lrate, epochs, neuronsPerLayer)
        data.makeBatches()
        self.makeLayers()

    def feedNet(self, imgs, targs, isTraining):
        """for one batch of images and target vectors, pass one image through
        network at a time, calculate error and gradient. Returns a list containing
        all compiled errors and number of correct guesses.

        imgs: batch of image vectors
        targs: batch of corresponding target vectors
        isTraining: bool, True=add to gradient, update weights. False=don't
        update weights after each batch (used for testing)
        """
        errors = []
        correct = 0
        for b in range(len(imgs)): #0->batchsize
            curr_img = imgs[b]
            curr_tar = targs[b]

            #forward pass
            self.forwardpass(curr_img)
            if isTraining:
                #backprop, starting by passing in target vec
                self.backpass(curr_tar)

                #calculate gradients
                self.findAllGradients(curr_img)

            #calculate error
            # print(curr_tar, self.layers[-1].output)
            errors.append(self.layers[-1].crossEntError(curr_tar))

            # #Did the network guess right?
            if np.argmax(self.layers[-1].output) == np.argmax(curr_tar):
                correct += 1
        return [errors, correct]

    def train(self, batch):
        batchimg = self.data.batchImg[batch]
        batchtar = self.data.batchTar[batch]

        results = self.feedNet(batchimg, batchtar, True)
        # print(len(results[0]), results[0], results[1])
        [layer.update() for layer in self.layers]

        self.trainloss.append(np.average(np.asarray(results[0])))
        self.trainacc.append((results[1] / len(batchimg)) * 100)

    def test(self):
        testimgs = self.data.test.img
        testtargs = self.data.test.tar

        results = self.feedNet(testimgs, testtargs, False)

        self.testloss.append(np.average(np.asarray(results[0])))
        self.testacc.append((results[1] / len(testimgs)) * 100)

    def run_one_epoch(self, testInt, saveInt):
        # self.loadNetWeights(restore)
        for b in range(len(self.data.batchImg)):    #batches in epoch
            self.train(b)
            print("[Batch: ", b, " Loss: ", self.trainloss[-1], " Accuracy: ", self.trainacc[-1], "%]")

            if (b % testInt == 0) and b > 0:
                print("Testing . . .")
                self.test()

            if (b % saveInt) == 0:
                print("Checkpoint, pickling network. . .")
                self.saveNetWeights()

    def run(self, testInt, saveInt, restore, savepaths):
        plt.close('all')

        #load and pre-process the data
        print("Loading training and test data")
        self.loadNetWeights(restore)

        #how many times to pass through entire dataset
        for i in range(self.epochs):
            print("Starting new epoch: ", i)
            #Shuffle training data
            self.data.makeBatches()
            # self.data.visualize(self.data.batchImg[0][0:10], self.data.batchTar[0][0:10])
            self.run_one_epoch(testInt, saveInt)

        self.plotsave(savepaths)
        self.printResults()

class DebuggingTests(RunNetwork):
    def __init__(self, inputVec, out, lrate, epochs, neuronsPerLayer):
        data = DataSet("mnist_train_100.csv", "mnist_test_10.csv", 1)
        data.makeBatches()
        super().__init__(data, lrate, epochs, neuronsPerLayer)
        self.makeLayers()
        self.loadNetWeights(restore=False)
        self.input = inputVec
        self.out = out

    def oneImgTest(self, timesToRepeat, savepaths):
        errors = []
        for i in range(timesToRepeat):
            correct = 0
            self.forwardpass(self.input)
            self.backpass(self.out)

            self.findAllGradients(self.input)

            [layer.update() for layer in self.layers]
            errors.append(self.layers[-1].crossEntError(self.out))

            if np.argmax(self.layers[-1].output) == np.argmax(self.out):
                correct = 1
            self.trainacc.append(correct*100)
        self.trainloss = errors
        self.plotsave(savepaths)

    def gradientCheck(self):
        # weights is the list of weight matrices
        # inputVec and target is a single training pair
        numBroken = 0
        layers1 = copy.deepcopy(self.layers)
        layers2 = copy.deepcopy(self.layers)

        self.forwardpass(self.input)
        self.backpass(self.out)
        self.findAllGradients(self.input)

        [print(layer.weights.shape) for layer in self.layers]

        # Check each component of the gradient via numerical differentiation
        # WARNING: Very slow
        eps = 1e-9
        for layer in range(len(layers1)):
            for j in range(layers1[layer].row):
                for k in range(layers1[layer].col):
                    # Add / subtract a small number to the current weight
                    layers1[layer].weights[j, k] += eps
                    layers2[layer].weights[j, k] -= eps

                    # Compute the new error
                    first_layer = layers1[0]
                    first_layer.forward(self.input)
                    for i in range(1, len(layers1)):
                        curr_layer = layers1[i]
                        prev_layer = layers1[i-1]
                        # print(prev_layer.output.shape)
                        curr_layer.forward(prev_layer.output)
                    first_layer = layers2[0]
                    first_layer.forward(self.input)
                    for i in range(1, len(layers2)):
                        curr_layer = layers2[i]
                        prev_layer = layers2[i-1]
                        curr_layer.forward(prev_layer.output)
                    # activations1 = forwardprop(inputVec, weights1)
                    # activations2 = forwardprop(inputVec, weights2)
                    error1 = layers1[-1].crossEntError(self.out)
                    error2 = layers2[-1].crossEntError(self.out)

                    #remove the eps value
                    layers1[layer].weights[j, k] -= eps
                    layers2[layer].weights[j, k] += eps

                    # Check gradient component vs central difference
                    numGrad = self.lrate * (error1 - error2) / (2*eps)
                    grad = self.layers[layer].gradient[j, k]
                    #print(abs(grad -numGrad))
                    if (abs(grad - numGrad) > 1e-4):
                        numBroken += 1
                        print("[", layer, ", ", j, ", ", k, "]")
                        print("ERROR: Wrong gradient component")
                        print("Should be: ", numGrad)
                        print("Is: ", grad)
        return numBroken

if __name__ == "__main__":
    savepaths = ['trl1.pckl', 'tel1.pckl', 'tra1.pckl', 'tea1.pckl']
    data = DataSet("mnist_train.csv", "mnist_test.csv", 500)
    stoch = RunNetwork(data, .001, 1, [784, 625, 625, 10])


    stoch.run(10, 100, False, savepaths)

    #FOR TESTING
    # testimg = data.batchImg[0][0]
    # testtarg = data.batchTar[0][0]
    # debug = DebuggingTests(testimg, testtarg, .001, 1, [784, 200, 10])
    # debug.oneImgTest(150, savepaths)
    # broke = debug.gradientCheck()
    # print(broke)
