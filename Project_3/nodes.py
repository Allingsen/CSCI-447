from abc import ABC, abstractmethod
import numpy as np

class Node(ABC):
    def __init__(self, no_inputs):
        super().__init__()
        self.no_inputs = no_inputs
        self.weights = None
    
    def manipulate_data(self, inputs:np.array) -> float:
        weighted_data = inputs * self.weights
        sum = 0
        for data in weighted_data:
            sum += data
        print("INPUTS: {}, WEIGHTS: {}, WEIGHTED_DATA: {}, SUM: {}".format(inputs, self.weights, weighted_data, sum))
        return np.sum(weighted_data)

    @abstractmethod
    
    def activation_function(self) -> float:
        pass

# List of Node Types
class Input(Node):
    def __init__(self, no_inputs):
        super().__init__(no_inputs)
        self.weights = np.ones(self.no_inputs)

    def activation_function(self, inputs:np.array):
        '''Implements the linear activation function'''
        return self.manipulate_data(inputs)
    
class Hidden(Node):
    def __init__(self, no_inputs):
        super().__init__(no_inputs)
        self.weights = np.random.uniform(-0.01, 0.01, self.no_inputs)

    def activation_function(self, inputs:np.array):
        '''Implements the sigmoid (logistic) activation function'''
        weighted_sum = self.manipulate_data(inputs)
        sigmoid = 1 / (1 + np.exp(-weighted_sum))
        return sigmoid

class Output_Reg(Node):
    def __init__(self, no_inputs):
        super().__init__(no_inputs)
        self.weights = np.random.uniform(-0.01, 0.01, self.no_inputs)

    def activation_function(self, inputs:np.array):
        '''Implements the linear activation function'''
        return self.manipulate_data(inputs)

class Output_Class(Node):
    def __init__(self, no_inputs):
        super().__init__(no_inputs)
        self.weights = np.random.uniform(-0.01, 0.01, self.no_inputs)

    def activation_function(self, inputs:np.array):
        '''Implements the softmax activation function'''
        # Softmax is done on the layer level, return weighted sum 
        return self.manipulate_data(inputs)
