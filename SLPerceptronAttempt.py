#SL perceptron skeleton
#---------------------
from random import randrange

import numpy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
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

        self.weights = weights
        self.bias = bias


    # P3: creates update bias/weight. Set the weights and bias at the start to default (0 and vector of zeros).
    # The perceptron is iterated an arbitrary amount of times for the entire dataset.
    # Within each loop all data is trained through another loop which 'enumerates' through the dataset (a matrice) while keeping
    # track of an object (the current vector row) and its index (the current row num). The weights are all trained simultaneously using
    # each iteration of the current enumerated row (the current object). The bias is also altered during this process.
    def train(self,x,y):
        self.bias = 0
        self.generate_random_weights(x)

        y = threshold(y)
        for iteration in range(self.training_loops):
            for index,object_in_row in enumerate(x):
                y_predicted = self.predict(object_in_row)

                bias_weight_alter = self.learning_rate * (y[index]- y_predicted)
                self.bias += bias_weight_alter
                self.weights += bias_weight_alter * object_in_row

    #P4:creates predicted y (y hat)
    def predict(self,x):
        y_predicted_no_threshold = numpy.dot(x,self.weights) + self.bias
        y_predicted = threshold(y_predicted_no_threshold)
        return y_predicted

    def generate_random_weights(self,x):
        self.weights = [0] * x.shape[1]
        for index in range(len(self.weights)):
            self.weights[index] = randrange(-1,1)

# P2: threshold func for weighted sum - takes in float value to make a binary output
def threshold(x):
    return numpy.where(x > 0, 1, 0)

# P5: Generate testing
if __name__ == "__main__":

    def accuracy(y_hat,y):
        accuracy = numpy.sum(y_hat == y)/len(y_hat)
        return accuracy

    X, y = datasets.make_blobs(n_samples=500, n_features=2, centers = 2, cluster_std=1.0, center_box=(-10, 10), random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)

    perceptron = SLPerceptronAttempt(learning_rate=None,training_loops=None,weights=None,bias=None)
    perceptron.train(X_train, y_train)
    predictions = perceptron.predict(X_test)

    print("Perceptron accuracy:",accuracy(predictions,y_test))

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)

    x0_1 = numpy.amin(X_train[:,0])
    x0_2 = numpy.amax(X_train[:,0])

    x1_1 = (-perceptron.weights[0] * x0_1 - perceptron.bias) / perceptron.weights[1]
    x1_2 = (-perceptron.weights[0] * x0_2 - perceptron.bias) / perceptron.weights[1]

    ax.plot([x0_1,x0_2],[x1_1,x1_2],'r')

    y_min = numpy.amin(X_train[:,1])
    y_max = numpy.amax(X_train[:,1])
    ax.set_ylim([y_min-5,y_max+5])

    plt.show()

    


    pass

# Generate training data

# Use training data to make 'blobs' in which training/graphing will occur
# (i.e. utilise the dataset)-

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