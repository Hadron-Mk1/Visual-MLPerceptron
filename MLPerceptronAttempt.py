import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import random

class MLPerceptronAttempt:
    """
    MLP Class, used to create a multi-layer perceptron capable of learning and recognising context and images
    """
    def __init__(self,input_size,hidden_size,output_size,training_loops):

        self.hidden_weights_one = np.random.randn(input_size,hidden_size)*0.05
        self.hidden_weights_two = np.random.randn(hidden_size,hidden_size)*0.05
        self.hidden_weights_three = np.random.randn(hidden_size, hidden_size) * 0.05
        self.output_weights = np.random.randn(hidden_size,output_size)*0.05

        self.hidden_bias_one = np.zeros(hidden_size)
        self.hidden_bias_two = np.zeros(hidden_size)
        self.hidden_bias_three = np.zeros(hidden_size)
        self.output_bias = np.zeros(output_size)

        self.total_layers = 4

        self.weights = [self.hidden_weights_one,self.hidden_weights_two,self.hidden_weights_three, self.output_weights]
        self.biases = [self.hidden_bias_one,self.hidden_bias_two, self.hidden_bias_three, self.output_bias]

        self.z_vals = []
        self.z_final_vals = []

        self.training_loops = training_loops


    def forward_propagation(self, x, layer):
        """
        Propagates an input through the network to produce a prediction
        :param x: The input to the perceptron
        :param layer: The layer to be trained
        :return: The predicted label
        """
        if layer == 0:
            self.z_vals = []
            self.z_final_vals = []

        # Z = WX + B
        z = np.dot(x, self.weights[layer]) + self.biases[layer]
        self.z_vals.append(z)

        if layer < self.total_layers - 1:
            activation_num = reLu(z)
            self.z_final_vals.append(activation_num)
            return self.forward_propagation(activation_num, layer + 1)
        else:
            activation_num = softmax(z)
            self.z_final_vals.append(activation_num)
            return activation_num


    def backward_propagation(self, x, y, learning_rate):
        """
        Used to adjust the weights and bias of the perceptron
        :param x: The input to the perceptron
        :param y: The label of the input to the perceptron
        :param learning_rate: The rate of learning
        :return:
        """
        # Works out error and how much the weight contributes to the error for later adjustment
        delta = [None] * self.total_layers
        error = self.z_final_vals[3] - y

        for layer in reversed(range(self.total_layers)):
            if layer == 3:
                delta_val = error
            else:
                error = np.dot(delta_val, self.weights[layer + 1].T)
                delta_val = error * relu_deriv(self.z_vals[layer])
            delta[layer] = delta_val

        for layer in range(self.total_layers):
            if layer == 0:
                prev_output = x
            else:
                prev_output = self.z_final_vals[layer - 1]

            # gradient is the multiple of x and delta (x is deriv z/ deriv W and delta is deriv E/ deriv a)
            # since deriv E/deriv W is split up
            gradient = np.outer(prev_output, delta[layer])
            self.weights[layer] -= learning_rate * gradient
            self.biases[layer] -= learning_rate * delta[layer]


    def train(self, dataset, learning_rate):
        """
        Used to train the perceptron
        :param dataset: The dataset to be trained on
        :param learning_rate: The rate of learning
        :return: A list of accurate guesses to draw and present to the user
        """
        accuracy_list = []
        for loop in range(self.training_loops):
            correct = 0
            for image, label in dataset:
                prediction = np.argmax(self.forward_propagation(image, 0))

                if prediction == np.argmax(label):
                    correct += 1

                self.backward_propagation(image, label, learning_rate)

            print("Training loop " + str(loop + 1)+" complete")

            accuracy = correct / len(dataset)
            accuracy_list.append(accuracy)
        return accuracy_list


def reLu(value):
    """
    Applies reLu function to mitigate vanishing gradient problem
    :param value: The value to apply the function to
    :return: A value of 0 or greater
    """
    return np.maximum(0, value)

def relu_deriv(value):
    """
    Outputs true or false depending on the value input
    :param value: the value input into the function
    :return: True if value > 0 or false if value <= 0
    """
    return value > 0

def softmax(vector):
    """
    A function that takes in a standard vector and outputs a probability vector of the input param
    :param vector: The vector you would like a probability distribution of
    :return: A probability vector
    """
    #Preventing exponential modifiers from becoming too large by enuring a range of -x to 0
    vector_adjusted = vector - np.max(vector)
    #All vector values are now in their exponential form
    probability_no_summation_division = np.exp(vector_adjusted)
    summation_of_exp_vector = np.sum(probability_no_summation_division)
    #Dividing each vector value by the sum of all vector values to create a probability distribution
    probability_vector = probability_no_summation_division / summation_of_exp_vector

    return probability_vector

if __name__ == "__main__":
    # MNIST - collection of handwritten digits (0-9)
    fashion = fetch_openml('Fashion-MNIST',version = 1, as_frame = False,parser='liac-arff')

    # Data is the actual img and digit is the number that represents
    data = np.array(fashion.data, dtype=float)
    clothes_list = np.array(fashion.target, dtype=int)

    # Normalise pixel vals as it holds 0-255
    # Wx + b can be unreliable with larger values
    x = data/255.0

    # One-hot code labels - error = self.z_final_vals[2] - y requires y to be row of 10
    y = np.zeros((clothes_list.size, 10))
    y[np.arange(clothes_list.size), clothes_list] = 1

    # Create training data
    dataset = list(zip(x, y))

    perceptron = MLPerceptronAttempt(input_size=784, hidden_size=50, output_size=10, training_loops=20)

    accuracy_scores = perceptron.train(dataset[:10000], learning_rate=0.01)
    plt.plot(range(1, len(accuracy_scores)+1), accuracy_scores)
    plt.xticks(range(0, len(accuracy_scores)+1,2))
    plt.yticks(np.arange(0, max(accuracy_scores)+0.1,0.1))
    plt.title("Accuracy score of each training loop")
    plt.xlabel('Number of training loops')
    plt.ylabel('Accuracy')
    plt.show()

    clothes = [
        "T-shirt",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Boots"
    ]

    for i in range(3):
        random_number = random.randint(0, len(x) - 1)
        image = x[random_number]

        prediction = np.argmax(perceptron.forward_propagation(image, 0))
        actual = np.argmax(y[random_number])

        plt.imshow(image.reshape(28, 28), cmap="inferno")
        plt.title(f"Pred: {clothes[prediction]}       Actual: {clothes[actual]}", pad = 10)
        plt.axis("off")
        plt.show()

