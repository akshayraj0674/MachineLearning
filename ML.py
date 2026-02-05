x, y = [2,3,4,9], [5,8,6,15]


import numpy as np


# things required for training a NN
# 1. model
# 2. prepare data
# 3. loss function and optimizer
# 4. training loop
# 5. evaluate the model


def sigmoid_neuron(input, weight, bias):
    output = input * weight + bias
    output = 1/(1 + np.exp(-output))
    return output


sigmoid_neuron(1000, 1, 0)


