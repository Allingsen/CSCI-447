import numpy as np
import nodes

class Layer():
    def __init__(self, version: int, num_nodes: int, batch_size: int, prev_layer, next_layer):
        self.version = version
        self.num_nodes = num_nodes
        self.nodes = list()
        self.batch_size = batch_size
        self.prev_layer = prev_layer
        self.next_layer = next_layer
        self.activation_matrix = np.zeros((batch_size, num_nodes))
        self.activation_row = 0

        self.create_nodes()

    def create_nodes(self):
        '''Creates the Nodes in the layer, depending on type of layer'''
        match self.version:
            case 0: #Input Layer
                for _ in range(self.num_nodes):
                    self.nodes.append(nodes.Input(self.num_nodes))
            case 1: # Hidden Layer
                for _ in range(self.num_nodes):
                    self.nodes.append(nodes.Hidden(self.prev_layer.num_nodes))
            case 2: # Output (Regression) Layer
                for _ in range(self.num_nodes):
                    self.nodes.append(nodes.Output_Reg(self.prev_layer.num_nodes))
            case 3: # Output (Classification) Layer
                for _ in range(self.num_nodes):
                    self.nodes.append(nodes.Output_Class(self.prev_layer.num_nodes))
            case _: raise ValueError('Incorrect value passed to Layers!')
    
    def generate_output(self, inputs:np.array) -> np.array:
        '''Generates a nump array of the output of all nodes'''
        # If it is the input layer, pass the original values
        if self.version == 0:
            out = inputs
            if(self.activation_row > self.batch_size):
                self.activation_row = 0
            self.activation_matrix[self.activation_row] = out
            self.activation_row += 1
            
        else:
            # Get the values of all output functions 
            out = np.zeros(self.num_nodes)
            for i, val in enumerate(self.nodes):
                print(f'weights: {val.weights}')
                out[i] = val.activation_function(inputs)
            if(self.activation_row > self.batch_size):
                self.activation_row = 0
            self.activation_matrix[self.activation_row] = out
            self.activation_row += 1
            
            # Implements softmax, done on the layer level for ease
            if self.version == 3:
                out = np.exp(out) / np.sum(np.exp(out))

        print(f'Layer output: {out}')
        return out
    
    def get_activation_matrix(self):
        return(self.activation_matrix)
