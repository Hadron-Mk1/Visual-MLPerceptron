#SL perceptron skeleton
#---------------------
from random import randrange

import numpy
import matplotlib.pyplot as plt
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

    # Generate accuracy score for perceptron using y and y-hat
    def accuracy(y_hat,y):
        accuracy = numpy.sum(y_hat == y)/len(y_hat)
        return accuracy

    # Use training data to make 'blobs' in which training/graphing will occur
    X, y = datasets.make_blobs(n_samples=250, n_features= 2, centers = 2, cluster_std= 1.0, center_box = (-10, 10), random_state = 1)
    # Split the training data into subsets that can be used for testing and training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)

    # Make perceptron predictions
    perceptron = SLPerceptronAttempt(learning_rate=None,training_loops=None,weights=None,bias=None)
    perceptron.train(X_train, y_train)
    predictions = perceptron.predict(X_test)

    # Shows accuracy to user
    print("Perceptron accuracy:",accuracy(predictions,y_test))

    # Subplot to show results of training data
    fig = plt.figure()
    ax = fig.add_subplot(2,1,1)
    plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)
    plt.title("Training Data")

    # Pick two x coords in the training data (x coords are xo in eq for decision boundary)
    train_x1 = numpy.amin(X_train[:,0])
    train_x2 = numpy.amax(X_train[:,0])

    # Generate the respective y coords to plot the decision boundary (use decision boundary eq)
    train_y1 = (-perceptron.weights[0] * train_x1 - perceptron.bias) / perceptron.weights[1]
    train_y2 = (-perceptron.weights[0] * train_x2 - perceptron.bias) / perceptron.weights[1]

    # Plot the decision boundary
    ax.plot([train_x1,train_x2],[train_y1,train_y2],'r')

    # Find the largest and smallest values in the y coordinates of the test training data
    # and use this to set the limits of the graph in the y axis
    train_y_min = numpy.amin(X_train[:,1])
    train_y_max = numpy.amax(X_train[:,1])
    ax.set_ylim([train_y_min-5,train_y_max+5])

    # Subplot to show results of test data
    ax = fig.add_subplot(2, 1, 2)
    plt.scatter(X_test[:, 0], X_test[:, 1], marker="o", c=y_test)
    plt.title("Test Data")

    test_x1 = numpy.amin(X_test[:, 0])
    test_x2 = numpy.amax(X_test[:, 0])

    test_y1 = (-perceptron.weights[0] * test_x1 - perceptron.bias) / perceptron.weights[1]
    test_y2 = (-perceptron.weights[0] * test_x2 - perceptron.bias) / perceptron.weights[1]

    ax.plot([test_x1, test_x2], [test_y1, test_y2], 'r')

    test_y_min = numpy.amin(X_test[:, 1])
    test_y_max = numpy.amax(X_test[:, 1])
    ax.set_ylim([test_y_min - 5, test_y_max + 5])

    plt.tight_layout()

    # Displays the two subplots for training and test data
    plt.show()