import numpy as np
import random
from feedForwardNN import FeedForwardNN

#ALL ERROR SIGNAL FUNCTIONALITY IS WORKING
#CURRENTLY TESTING BACKPROPAGATION

input = np.random.randint(1, 3, size=(3, 5))
print("INPUTS:", input)
nodes = [4]
#predicted_values = [1, 2, 0]
#actual_values = [1, 2, 2]
predicted_values = [1.98, 9.776, 0.7876]
actual_values = [2.001, 9.887, 0.6790]
#probabilities_list = [[0.2, 0.7, 0.1],[0.3, 0.2, 0.5],[0.8, 0.1, 0.1]]
probabilities_list = []
#class_names = ["0", "1", "2"]
network = FeedForwardNN(input, 1, nodes, False, 0.01, 3)
network.test_list()
MSE = network.error_signal(predicted_values, actual_values, probabilities_list)
network.backpropagate(MSE, predicted_values, actual_values, input, probabilities_list)
