import numpy as np

class MLPerceptronAttempt:
    def __init__(self,input_size,hidden_size,output_size): # Initialise weights and biases
        # Creates a random number matrices of weights for the inputs and hidden data to be processed with
        # Biases are started at 0 to be adjusted later
        # Weights and bias created for each layer

        self.hidden_weights_one = np.random.randn(input_size,hidden_size)
        self.hidden_weights_two = np.random.randn(hidden_size,hidden_size)
        self.output_weights = np.random.randn(hidden_size,output_size)

        self.hidden_bias_one = np.random.randn(hidden_size)
        self.hidden_bias_two = np.random.randn(hidden_size)
        self.output_bias = np.random.randn(output_size)

        self.total_layers = 3

        self.weights = [self.hidden_weights_one,self.hidden_weights_two, self.output_weights]
        self.biases = [self.hidden_bias_one,self.hidden_bias_two, self.output_bias]

        self.z_vals = []
        self.z_final_vals = []

    def forward_propagation(self, x, layer):
        # Z = WX + B
        z = np.dot(x, self.weights[layer]) + self.biases[layer]
        self.z_vals.append(z)

        if layer < self.total_layers - 1:
            # Using reLu mitigates vanishing gradient problem (prevention of lack of self learning)
            self.z_final_vals.append(reLu(z))
            return self.forward_propagation(reLu(z), layer + 1)
        else:
            self.z_final_vals.append(sigmoid(z))
            return sigmoid(z)


#Function to allow all continuous input to be mapped between 0 or 1
def sigmoid(value):
    return 1 / (1 + np.exp(-value))

def reLu(value):
    return np.maximum(0, value)

def sigmoid_deriv(value):
    return (sigmoid(value)*(1-sigmoid(value)))

def relu_deriv(value):
    return value > 0


def softmax(vector):
    #Preventing exponential modifiers from becoming too large by enuring a range of -x to 0
    vector_adjusted = vector - np.max(vector)
    #All vector values are now in their exponential form
    probability_no_summation_division = np.exp(vector_adjusted)
    summation_of_exp_vector = sum(probability_no_summation_division)
    #Dividing each vector value by the sum of all vector values to create a probability distribution
    probability_vector = probability_no_summation_division / summation_of_exp_vector

    return probability_vector

def backward_propagation(self, x, y, learning_rate):
    # Works out error and how much the weight contributes to the error for later adjustment

    delta = [None] * self.total_layers
    error = self.z_final_vals[2] - y

    for layer in reversed(range(self.total_layers)):
        if layer == 2:
            delta_val = error * sigmoid_deriv(self.z_vals[layer])
        else:
            error = np.dot(delta_val, self.weights[layer + 1].T)
            #
        delta[layer] = delta_val

    pass

def train():
    pass

def predict():
    pass