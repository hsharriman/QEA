# Building Neural Networks Using Numpy
### Hwei-Shin Harriman
#### Originally created for Olin College of Engineering's Quantitative Engineering Analysis Final Project, Fall 2018

## Running the Networks (TODO: flesh out)
* Before running this program, make sure that you have Python 3 installed, as well as Numpy, matplotlib, and pickle
* Also follow this link to download the MNIST data set of handwritten digits in .csv format
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
