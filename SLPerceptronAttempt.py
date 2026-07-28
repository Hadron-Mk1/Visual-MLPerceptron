#SL perceptron skeleton
#---------------------

import numpy
import matplotlib.pyplot as plt
from sklearn import datasets

# P1:
# Learning rate between 0 and 1
class SLPerceptronAttempt:
    def __init__(self,training_loops,learning_rate,weights, bias):

        if training_loops is None:
            self.training_loops = 100
        else:
            self.training_loops = training_loops

        if learning_rate is None:
            self.learning_rate = 0.01
        else:
            self.learning_rate = learning_rate

        if weights is None:
            self.weights = []

        if bias is None:
            self.bias = 0


    # P3: creates update bias/weight. Set the weights and bias at the start to default (0 and vector of zeros).
    # The perceptron is iterated an arbitrary amount of times for the entire dataset.
    # Within each loop all data is trained through another loop which 'enumerates' through the dataset (a matrice) while keeping
    # track of an object (the current vector row) and its index (the current row num). The weights are all trained simultaneously using
    # each iteration of the current enumerated row (the current object). The bias is also altered during this process.
    def train(self,x,y):
        self.bias = 0
        x = numpy.zeros(x.shape)
        y = threshold(y)

        for iteration in range(self.training_loops):
            for index,object_in_row in enumerate(x):
                y_predicted = self.predict(object_in_row)

                bias_weight_alter = self.learning_rate * (y[index]- y_predicted)
                self.bias += bias_weight_alter
                self.weights += bias_weight_alter * object_in_row





    #P4:creates predicted y (y hat)
    def predict(self,x):
        y_predicted_no_threshold = numpy.dot(self.weights,x) + self.bias
        y_predicted = threshold(y_predicted_no_threshold)
        return y_predicted




# P2: threshold func for weighted sum - takes in float value to make a binary output
def threshold(x):
    return numpy.where(x > 0, 1, 0)

# P5: Generate testing
if __name__ == "__main__":
    pass

# Generate training data

# Use training data to make 'blobs' in which training/graphing will occur
# (i.e. utilise the dataset)

# Using the test data we must generate a scatter graph, column 1 is all x coords and column 2 is all y coords
# In training data

# Make perceptron predictions

# Generate accuracy score for perceptron using y and y-hat and show this to user

# Pick two random x coords in the training data (x coords are xo in eq for decision boundary)

# Generate the respective y coords to plot the decision boundary (use decision boundary eq)

# Plot the decision boundary

# Find the largest and smallest values in the y coordinates of the test training data
# and use this to set the limits of the graph in the y axis

# Set the limits in the y axis

# Graph is ready to be shown