#inputs is a 2D array of feature vectors

#hidden_layers is an int representing the number of hidden layers to be present in the network

#nodes is a list of num of nodes for each hidden layer

#classification is a boolean indicating whether the task is classification or not

#learning rate is a float representing the learning rate (hyperparameter)

#batch size is an int representing the mini-batch size (hyperparameter)

#num_of_classes is an int representing the number of classes

#class_names is a list containing the class labels

import numpy as np
import math
from layers import Layer

probabilities_list = []

class FeedForwardNN():
    def __init__(self, inputs: np.array, hidden_layers: int, nodes: list, classification: bool, num_of_classes: int=1, class_names=None):
        self.inputs = inputs
        self.hidden_layers = hidden_layers
        self.nodes = nodes
        if len(nodes) != self.hidden_layers:
            raise ValueError('Layers and nodes do not match up!')
        self.classification = classification
        self.num_of_classes = num_of_classes
        if self.num_of_classes != 1 and not self.classification:
            raise AttributeError('Regression should only have 1 output')
        self.class_names = class_names
        if self.class_names and not self.classification:
            raise AttributeError('Regression should not have classes.')

        # Create the input layer as the length of the example, minus the class
        self.inputLayer = Layer(0, len(self.inputs[0])-1, len(self.inputs), None, None)
        self.create_layers()

    def create_layers(self):
        '''Creates the Entire Network'''
        for i in range(self.hidden_layers):
            # Initlizes the Linked list
            cur_layer = self.inputLayer
            # Gets to the end
            while cur_layer.next_layer:
                cur_layer = cur_layer.next_layer
            # Inserts the new layer at the end
            new_layer = Layer(1, self.nodes[i], len(self.inputs), cur_layer, None)
            cur_layer.next_layer = new_layer

        # Adds the final output layer depending on type of network (Classification or Regression)
        cur_layer = self.inputLayer
        while cur_layer.next_layer:
            cur_layer = cur_layer.next_layer
        if self.classification:
            cur_layer.next_layer = Layer(3, self.num_of_classes, len(self.inputs), cur_layer, None)
        else:
            cur_layer.next_layer = Layer(2, 1, len(self.inputs), cur_layer, None)

    def test_list(self):
        '''Prints Network Information, only used for manual Checking'''
        print('-----Data-----\n', self.inputs)
        print('-----List Test-----')
        cur_layer = self.inputLayer
        while cur_layer:
            print(f'First Layer Type: {cur_layer.version}, Number of Nodes: {cur_layer.num_nodes}, Number of Inputs to each Node: {len(cur_layer.nodes[0].weights)}')
            cur_layer = cur_layer.next_layer
            
    def get_prediction(self, point: np.array) -> float:
        '''Gets the predictions of the network at a specified point'''
        cur_layer = self.inputLayer
        inputs = point
        # Iterates through the network, passing the previous layers output to the next layer
        while cur_layer:
            inputs = cur_layer.generate_output(inputs)
            cur_layer = cur_layer.next_layer
        # Returns the final prediciton
        if self.classification:
            probabilities_list.append(inputs)
            index = np.where(inputs == max(inputs))
            return self.class_names[index[0][0]]
        else:
            return inputs[0]
        
    #Calculates the error of a particular mini-batch (Either Cross-Entropy Loss or MSE)
    def error_signal(self, predicted_values, actual_values, probabilities_list):
        
        #If classification, use Cross-Entropy Loss derivative
        if(self.classification):

            #Binary Cross-Entropy Loss
            if(self.num_of_classes == 2):
                error_matrix = []

                for i in range(len(predicted_values)):

                    #Negative class will be class_names[0], positive class will be class_names[1]
                    pos_or_neg = None
                    if(actual_values[i] == (self.class_names[0])):
                        pos_or_neg = [1,0]
                    else:
                        pos_or_neg = [0,1]
                    
                    to_enter = probabilities_list[i] - pos_or_neg
                    error_matrix.append(to_enter)
                return(error_matrix)

            #Categorical Cross-Entropy Loss
            else:
                error_matrix = []

                for i in range(len(predicted_values)):
                    y_true = []
                    for c in range(self.num_of_classes):                     
                        if(actual_values[i] == (self.class_names[c])):
                            y_true.append(1)                         
                        else:
                            y_true.append(0)
                    
                    #print(f"y_true: {y_true}")                  
                    error_matrix.append(probabilities_list[i] - y_true)

                return(error_matrix)

        #If regression, use Mean Squared Error derivative
        else:
            error_vector = []
            for i in range(len(predicted_values)):
                error_vector.append(predicted_values[i] - actual_values[i])
            return(error_vector)
    
    def get_weights(self):
        '''Returns the weight matrix of the final layer'''
        all_weights = []
        cur_layer = self.inputLayer
        while(cur_layer.next_layer != None):
            cur_layer = cur_layer.next_layer
            all_weights.append(cur_layer.weight_matrix.flatten())
        return all_weights
    
    def get_chromosome(self):
        '''Returns a "Chromosome" (Flattened weights)'''
        chrome = np.concatenate(self.get_weights())
        return chrome
    
    def set_weights(self, chrome: np.array) -> None:
        cur_layer = self.inputLayer
        while(cur_layer.next_layer != None):
            cur_layer = cur_layer.next_layer
            cur_layer.set_weights(chrome)
        
        cur_layer.get_weight_matrix()
            
    def get_fitness(self):
        '''Feeds data through the network, returning the fitness function'''
        cur_layer = self.inputLayer
        while(cur_layer.next_layer != None):
            cur_layer = cur_layer.next_layer
        output_layer = cur_layer
        # Counter used for mini-batches
        # Saves the predicted values and the acutal values
        all_actual = np.array([None]*len(self.inputs))
        all_pred = np.array([None]*len(self.inputs))
        # Iterates through all inputs
        for i, val in enumerate(self.inputs):
            all_actual[i] = val[-1]
            all_pred[i] = self.get_prediction(val[:-1])
                
        probabilities_list = output_layer.get_probabilities()
        error_signal = np.array(self.error_signal(all_pred, all_actual, probabilities_list))

        probabilities_list.clear()
        output_layer.probabilities.clear()

        return -1 * error_signal
    
    def test_data(self, examples: np.array) -> tuple:
        '''Tests a new input of examples'''
        actual_val = np.array([None]*len(examples))
        predicted_val = np.array([None]*len(examples))
        for i, val in enumerate(examples):
            actual_val[i] = val[-1]
            predicted_val[i] = self.get_prediction(val[:-1])
        return actual_val, predicted_val 
    
    def get_final_weights(self):
        '''Returns the weight matrix of the final layer'''
        cur_layer = self.inputLayer
        while(cur_layer.next_layer != None):
            cur_layer = cur_layer.next_layer
        return(cur_layer.weight_matrix)
