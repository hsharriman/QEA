import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import copy
import pickle

class Data_MNIST:
    def __init__(self):
        self.img = []
        self.tar = []
        self.size = 0

    def load(self, file_name):
        #Opens training data set
        fd = open(file_name, 'r')
        loaded = fd.readlines()
        fd.close()

        numOutputs = 10
        #Pre-process the training data set
        for line in loaded:
            #reset options
            opt = np.array([0.] * numOutputs)

            #split off classification from input array
            linebits = line.split(',')
            self.img.append(np.asfarray(linebits[1:])/255)  #currently (784,1) on intensity scale 0-1reshape((28,28))

            #classify from 0-9
            opt[int(linebits[0])] = 1.
            self.tar.append(opt)
        self.size = len(self.img)

    def printMNIST(self):
        print(self.size, len(self.img[0]), len(self.tar), self.tar[-1])
        print(len(self.img[0]), len(self.tar[0]))

    def padSet(self, padsz):
        for b in range(self.size):
            img = self.img[b]
            pads = np.zeros((np.shape(img)[0] + pad * 2, np.shape(img)[1] + pad * 2))
            pads[pad:-pad, pad:-pad] = img
            self.img[b] = pads

class DataSet(Data_MNIST):
    def __init__(self, trainfile, testfile, batchsize):
        self.train = Data_MNIST()   #training dataset
        self.test = Data_MNIST()    #testing dataset
        self.batchImg = []
        self.batchTar = []
        self.batchsz = batchsize
        self.train.load(trainfile)
        self.test.load(testfile)

    def printSet(self):
        print(len(self.train.img), len(self.train.tar))
        print(len(self.test.img), len(self.test.tar))

    def makeBatches(self):
        #shuffle the dataset
        self.batchImg = []
        self.batchTar = []
        shuffIndex = np.random.permutation(self.train.size)
        for bat in range(int(self.train.size // self.batchsz)):
            imgs = []
            targs = []
            for i in range(self.batchsz):
                index = shuffIndex[(i + self.batchsz*bat)]
                #print(index, (i+batchSize*bat))
                imgs.append(self.train.img[index])
                targs.append(self.train.tar[index])
            self.batchImg.append(imgs)
            self.batchTar.append(targs)

        #if the dataset not evenly split by batches
        if self.train.size % self.batchsz != 0:
            imgs = []
            targs = []
            last = self.train.size % self.batchsz
            for i in range(last - 1, -1, -1):
                index = shuffIndex[-(i + 1)]
                #print(index, -(i + 1))
                imgs.append(self.train.img[index])
                targs.append(self.train.tar[index])
            self.batchImg.append(imgs)
            self.batchTar.append(targs)
    def forXimgs(self, x, repeats):
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
        for i in range(len(patterns)):
            # The rest of columns are pixels
            # pixels = patterns[goodOnes[i][0]] * 255
            #
            # # Make those columns into a array of 8-bits pixels
            # # This array will be of 1D with length 784
            # # The pixel intensity values are integers from 0 to 255
            # pixels = pixels.astype(np.uint8)
            # print(len(patterns), len(patterns[i]))
            img = patterns[i]
            # Reshape the array into 28 x 28 array (2-dimensional array)
            pixels = img.reshape((28, 28))
            label = np.argmax(targets[i])
            #plot
            plt.title('Label is {label}'.format(label=label))
            plt.imshow(pixels, cmap='gray')
            plt.show()

class Layer:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.weights = np.random.randn(row, col)
        self.gradient = np.zeros((row, col))
        self.output = np.zeros(col)
        self.deltas = np.zeros(row)

    def makeWeights(self):
        self.weights = np.random.randn(self.row, self.col)
        print(self.weights)

    def printShapes(self):
        print("weights: ", self.weights.shape, " out: ", self.output.shape, " deltas: ", self.deltas.shape)

    def calcGradient(self, lrate, prev_layer, isFirstLayer):
        # print(self.deltas.shape)
        if isFirstLayer:
            reshapePrevOut = np.reshape(prev_layer, (1, prev_layer.shape[0]))
        else:
            reshapePrevOut = np.reshape(prev_layer.output, (1, prev_layer.output.shape[0]))
        reshapeDelta = np.reshape(self.deltas, (self.deltas.shape[0], 1))
        # reshapeOut = np.reshape(self.output, (1, len(self.output)))
        adjust = lrate * np.matmul(reshapeDelta, reshapePrevOut)
        # print("gradient size", self.gradient.shape)
        # print("Adjust ", adjust.shape)
        self.gradient += adjust

    def update(self):
        self.weights -= self.gradient
        self.gradient = np.zeros((self.row, self.col))

    def loadWeights(self, layerNumber, restore):
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
        filename = 'weights' + str(layerNumber)
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.weights, f)
        except:
            print("error saving weights at layer ", layerNumber, " filename ",filename)

class FullyConnected(Layer):
    def __init__(self, row, col):
        super().__init__(row, col)

    def sigmoid(self, mat):
        return 1 / (1 + np.exp(-mat))

    def sigPrime(self, mat):
        return (mat * (1 - mat))

    def forward(self, inputMat):
        self.output = self.sigmoid(np.matmul(self.weights, inputMat))

    def backprop(self, nextLayer):
        nextOut = nextLayer.output #Using the activations of the next layer
        nextWeights = nextLayer.weights #and the weights of the next layer
        nextDeltas = nextLayer.deltas
        # print("stuff going into backprop: ", nextOut.shape, nextWeights.shape, nextDeltas.shape)
        #one matrix-vector multiplication and one element-wise vector multiplication
        self.deltas = np.matmul(np.transpose(nextWeights), nextDeltas) * self.sigPrime(self.output)

class SoftmaxOut(Layer):
    def __init__(self, row, col):
        super().__init__(row, col)

    def softmax(self, mat):
        return np.exp(mat) / np.sum(np.exp(mat))

    def crossEntError(self, tar):
        clip = 1e-10
        return -1 * np.sum(tar * np.log(np.maximum(self.output, clip)))

    def forward(self, inputMat):
        # inMat = np.reshape(inputMat, (len(inputMat), 1))
        self.output = self.softmax(np.matmul(self.weights, inputMat))

    def backprop(self, tar):
        self.deltas = self.output - tar
        # print(self.output.shape, tar.shape, self.deltas.shape)

class Network:
    def __init__(self, data, lrate, epochs, neuronsPerLayer):
        self.data = data    #DataSet object, needs to be initialized outside of class
        self.lrate = lrate
        self.epochs = epochs
        self.npl = neuronsPerLayer  #array w num of neurons/layer
        self.layers = []    #array to be filled with layers
        self.trainacc, self.trainloss = [], []
        self.testacc, self.testloss = [], []

    def makeLayers(self):
        for i in range(len(self.npl)-2):
            self.layers.append(FullyConnected(self.npl[i+1], self.npl[i]))
        self.layers.append(SoftmaxOut(self.npl[-1], self.npl[-2]))

    def forwardpass(self, curr_img):
        #initial case, pass input vec in
        first_layer = self.layers[0]
        first_layer.forward(curr_img)
        for i in range(1, len(self.layers)):
            curr_layer = self.layers[i]
            prev_layer = self.layers[i-1]

            curr_layer.forward(prev_layer.output)

    def backpass(self, curr_tar):
        #backprop, starting by passing in target vec
        last_layer = self.layers[-1]
        last_layer.backprop(curr_tar)
        for i in range(len(self.layers) - 2, -1, -1):
            curr_layer = self.layers[i]
            next_layer = self.layers[i+1]
            #
            # print("curr layer is ")
            # curr_layer.printShapes()
            # print("next layer is")
            # next_layer.printShapes()
            # print("before backprop ", curr_layer.deltas.shape)
            curr_layer.backprop(next_layer)
            # print("after backprop ", curr_layer.deltas.shape)

    def findAllGradients(self, inputVec):
        for i in range(len(self.layers)-1, -1, -1):
            if i == 0:
                prev_layer = inputVec
                isFirstLayer = True
            else:
                prev_layer = self.layers[i-1]
                isFirstLayer = False
            # print(prev_layer.weights.shape)
            curr_layer = self.layers[i]
            # print(curr_layer.weights.shape)
            curr_layer.calcGradient(self.lrate, prev_layer, isFirstLayer)

class Utils(Network):
    def __init__(self, data, lrate, epochs, neuronsPerLayer):
        super().__init__(data, lrate, epochs, neuronsPerLayer)

    def loadNetWeights(self, restore):
        for i in range(len(self.layers)):
            layer = self.layers[i]
            layer.loadWeights(i, restore)

    def saveNetWeights(self):
        for i in range(len(self.layers)):
            layer = self.layers[i]
            layer.saveWeights(i)

    def save(self, itemToSave, savepath):
        temp = np.asarray(itemToSave).flatten()
        with open(savepath, 'wb') as f:
            print("Pickling training loss. . .")
            pickle.dump(temp, f)

    def plotsave(self, savepaths):
        print("Pickling network, losses and accuracies. . .")
        self.saveNetWeights()
        self.save(self.trainloss, savepaths[0])
        self.save(self.trainacc, savepaths[1])
        self.save(self.testloss, savepaths[2])
        self.save(self.testacc, savepaths[3])

        fig, ax = plt.subplots()
        fig2, ax2 = plt.subplots()
        fig3, ax3 = plt.subplots()
        fig4, ax4 = plt.subplots()

        ax.plot(self.trainloss, marker= '.')
        ax.set(xlabel = 'Batch Number', ylabel= 'Batch Cross Entropy Error', title= 'MNIST Training Loss')
        ax2.plot(self.testloss, 'm', marker= '.')
        ax2.set(xlabel= 'Batch Number', ylabel= 'Test Cross Entropy Error', title= 'Test Set Loss')
        ax3.plot(self.trainacc, marker='.')
        ax3.set(xlabel= 'Batch Number', ylabel= 'Percent Accuracy per Batch', title= 'Average Accuracy During Training')
        ax4.plot(self.testacc, 'm', marker='.')
        ax4.set(xlabel= 'Batch Number', ylabel= 'Percent Accuracy per Test Run', title= 'Average Accuracy of Testing')
        plt.show()

    def printResults(self):
        trloss = np.amin(np.asarray(self.trainloss))
        teloss = np.amin(np.asarray(self.testloss))
        tracc = np.amax(np.asarray(self.trainacc))
        teacc = np.amax(np.asarray(self.testacc))
        print("Lowest Recorded Training loss: ", trloss)
        print("Lowest Recorded Test loss: ", teloss)
        print("Highest training accuracy: ", tracc)
        print("Highest test accuracy: ", teacc)

class RunNetwork(Utils):
    def __init__(self, data, lrate, epochs, neuronsPerLayer):
        super().__init__(data, lrate, epochs, neuronsPerLayer)
        data.makeBatches()
        self.makeLayers()

    def feedNet(self, imgs, targs, isTraining):
        errors = []
        correct = 0
        for b in range(len(imgs)): #0->batchsize
            curr_img = imgs[b]
            curr_tar = targs[b]
            # print("target dimensions", curr_tar.shape)

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
