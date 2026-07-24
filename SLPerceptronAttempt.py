#SL perceptron skeleton
#---------------------

import numpy
import matplotlib.pyplot as plt
from sklearn import datasets

# P1:
# Learning rate between 0 and 1
class SLPerceptronAttempt:
    def __init__(self):
        pass


    # P3: creates update bias/weight. Set the weights and bias at the start to default (0 and vector of zeros).
    # The perceptron is iterated an arbitrary amount of times for the entire dataset.
    # Within each loop all data is trained through another loop which 'enumerates' through the dataset (a matrice) while keeping
    # track of an object (the current vector row) and its index (the current row num). The weights are all trained simultaneously using
    # each iteration of the current enumerated row (the current object). The bias is also altered during this process.
    def train(self):
        pass

    #P4:creates predicted y (y hat)
    def predict(self):
        pass



# P2: threshold func for weighted sum - takes in float value to make a binary output
def threshold():
    pass # 1 means fire 0 otherwise

#P5: Generate testing
if __name__ == "__main__":
    pass