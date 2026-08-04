import numpy as np

class SLPerceptronAttempt:
    def __init__(self):
        pass


#Function to allow all continuous input to be mapped between 0 or 1
def sigmoid(value):
    return 1 / (1 + np.exp(-value))


def softmax(vector):
    #Preventing exponential modifiers from becoming too large by enuring a range of -x to 0
    vector_adjusted = vector - np.max(vector)
    #All vector values are now in their exponential form
    probability_no_summation_division = np.exp(vector_adjusted)
    summation_of_exp_vector = sum(probability_no_summation_division)
    #Dividing each vector value by the sum of all vector values to create a probability distribution
    probability_vector = probability_no_summation_division / summation_of_exp_vector

    return probability_vector

def forward_propagation(z):
    pass

def backward_propagation(z):
    pass

def train():
    pass

def predict():
    pass