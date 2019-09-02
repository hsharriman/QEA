# Building Neural Networks Using Numpy
### Hwei-Shin Harriman
#### Originally created for Olin College of Engineering's Quantitative Engineering Analysis Final Project, Fall 2018

## About this Project
This project is meant to be a deep exploration into the math behind neural networks, specifically convolutional neural networks. As such, there is no use of high-level packages such as Keras or Tensorflow, since the goal was to understand the math that drives these powerful tools. Part of the original assignment was to create a homework assignment that other students in the class could complete to gain an understanding of some aspect of the technical material necessary for the project. The pdf of this homework assignment can be found [here](), and the source code that accompanies the assignment can be found [here]().
Additionally, the final deliverable for this project was a technical write-up detailing the necessary concepts, as well as a breakdown of the process. The writeup for this project can be found [here]().
## Running the Networks
* Before running this program, make sure that you have Python 3 installed, as well as Numpy, matplotlib, and pickle
* Also follow [this link](https://pjreddie.com/projects/mnist-in-csv/) to download the MNIST data set of handwritten digits in .csv format
* Clone this repository to your computer
#### Running the Feedforward Network
1. Go to the bottom of `ff2.py` and uncomment `stoch.run()`
2. From your terminal: `$ python ff2.py`
3. (Optional) To continue training with the most recently saved weights:
  At the bottom of the file, set `restore=True` then re-run the program
##### Guidelines for Changing Hyperparameters
- At bottom of file, can change number of epochs, batch size, and learning rate:
  - more epochs = longer training time but generally better network results
  - learning rate:
    - too small: gradient too small, likely to get stuck at local min
    - too big: gradient scaled up too much, likely to diverge
  - batch size:
    - too small: gradient updates too often, likely to converge too early
    - too big: gradient not updated enough, can't learn the details
