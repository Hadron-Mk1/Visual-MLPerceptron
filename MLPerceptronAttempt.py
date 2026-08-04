import numpy as np

class SLPerceptronAttempt:
    def __init__(self,input_size,hidden_size,output_size,total_layers): # Initialise weights and biases
        # Creates a random number matrices of weights for the inputs and hidden data to be processed with
        # Biases are started at 0 to be adjusted later

        self.input_weights = np.random.randn(input_size,hidden_size)
        self.hidden_weights = np.random.randn(hidden_size,output_size)
        self.input_bias = np.random.randn(hidden_size)
        self.output_bias = np.random.randn(output_size)


    def forward_propagation(self, x, layer):
        # Z = WX + B
        z = np.dot(x, self.weight[layer]) + self.bias[layer]
        if layer < self.total_layers - 1:
            # Using reLu mitigates vanishing gradient problem (prevention of lack of self learning)
            return self.forward_propagation(reLu(z), layer + 1)
        else:
            return sigmoid(z)


#Function to allow all continuous input to be mapped between 0 or 1
def sigmoid(value):
    return 1 / (1 + np.exp(-value))

def reLu(value):
    return np.maximum(0, value)


def softmax(vector):
    #Preventing exponential modifiers from becoming too large by enuring a range of -x to 0
    vector_adjusted = vector - np.max(vector)
    #All vector values are now in their exponential form
    probability_no_summation_division = np.exp(vector_adjusted)
    summation_of_exp_vector = sum(probability_no_summation_division)
    #Dividing each vector value by the sum of all vector values to create a probability distribution
    probability_vector = probability_no_summation_division / summation_of_exp_vector

    return probability_vector

def backward_propagation(z):
    pass

def train():
    pass

def predict():
    pass