import numpy as np
import os

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

    print(len(imgte), len(imgte[0]), len(tarte), tarte[-1], linebits[0])
    return imgtr, tartr, imgte, tarte
