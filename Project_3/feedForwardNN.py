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
    def __init__(self, inputs: np.array, hidden_layers: int, nodes: list, classification: bool, learning_rate: float, batch_size: int, num_of_classes: int=1, class_names=None):
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
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Create the input layer as the length of the example, minus the class
        self.inputLayer = Layer(0, len(self.inputs[0])-1, self.batch_size, None, None)
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
            new_layer = Layer(1, self.nodes[i], self.batch_size, cur_layer, None)
            cur_layer.next_layer = new_layer

        # Adds the final output layer depending on type of network (Classification or Regression)
        cur_layer = self.inputLayer
        while cur_layer.next_layer:
            cur_layer = cur_layer.next_layer
        if self.classification:
            cur_layer.next_layer = Layer(3, self.num_of_classes, self.batch_size, cur_layer, None)
        else:
            cur_layer.next_layer = Layer(2, 1, self.batch_size, cur_layer, None)

    def test_list(self):
        '''Prints Network Information, only used for manual Checking'''
        print('-----Data-----\n', self.inputs)
        print('-----List Test-----')
        cur_layer = self.inputLayer
        while cur_layer:
            print(f'First Layer Type: {cur_layer.version}, Number of Nodes: {cur_layer.num_nodes}, Number of Inputs to each Node: {len(cur_layer.nodes[0].weights)}')
            cur_layer = cur_layer.next_layer

    def train_data(self):
        '''Feeds data through the network, tuning weights using minibatch backprop'''
        # Counter used for mini-batches
        counter = 0
        # Saves the predicted values and the acutal values
        actual_val = np.array([None]*self.batch_size)
        predicted_val = np.array([None]*self.batch_size)
        # Iterates through all inputs
        for i in self.inputs:
            actual_val[counter] = i[-1]
            predicted_val[counter] = self.get_prediction(i[:-1])
            # If the minibatch has been iterated through, perform backprop
            if counter == self.batch_size:
                #Clear the probabilities for the next mini-batch
                probabilities_list.clear()
                error_signal = self.error_signal(predicted_val, actual_val, probabilities_list)
                # TODO: Perform Back Prop Here!
                self.backpropagate(error_signal)
                counter = 0

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
                error_vector = []

                for i in range(len(predicted_values)):

                    #Negative class will be class_names[0], positive class will be class_names[1]
                    pos_or_neg = None
                    if(actual_values[i] == int(self.class_names[0])):
                        pos_or_neg = 0
                    else:
                        pos_or_neg = 1
                    
                    error_vector.append(probabilities_list[i][1] - actual_values[i])
                return(error_vector)

            #Categorical Cross-Entropy Loss
            else:
                error_matrix = []

                for i in range(len(predicted_values)):
                    error_vector = []
                    for c in range(self.num_of_classes):
                        y_true = None
                        if(actual_values[i] == int(self.class_names[c])):
                            y_true = 1
                        else:
                            y_true = 0

                        error_vector.append(probabilities_list[i][c] - y_true)
                    error_matrix.append(error_vector)

                return(error_matrix)

        #If regression, use Mean Squared Error derivative (The 2/n term folds into the learning rate)
        else:
            error_vector = []
            for i in range(len(predicted_values)):
                error_vector.append(predicted_values[i] - actual_values[i])
            return(error_vector)
    
    def backpropagate(self, error_signal, predicted_values, actual_values, inputs, probabilities_list):
    
        #If regression...
        if(not self.classification):
           
           #This part will go away. Forward propagates examples for testing purposes
            cur_layer = self.inputLayer
            while(cur_layer != None):
               print(cur_layer.activation_matrix)
               cur_layer = cur_layer.next_layer
            print(inputs[0, :4])
            self.get_prediction(inputs[0, :4])
            cur_layer = self.inputLayer
            while(cur_layer != None):
               print(cur_layer.activation_matrix)
               cur_layer = cur_layer.next_layer
            self.get_prediction(inputs[1, :4])
            cur_layer = self.inputLayer
            while(cur_layer != None):
               print(cur_layer.activation_matrix)
               cur_layer = cur_layer.next_layer
            self.get_prediction(inputs[2, :4])
            cur_layer = self.inputLayer
            while(cur_layer != None):
               if(cur_layer.next_layer == None):
                   output_activation = cur_layer.activation_matrix
               print(cur_layer.activation_matrix)
               cur_layer = cur_layer.next_layer
            
            error_temp = self.error_signal(output_activation.flatten().tolist(), actual_values, probabilities_list)

           #Get to the output layer
            cur_layer = self.inputLayer
            while(cur_layer.next_layer != None):
               cur_layer = cur_layer.next_layer
            
            last_hidden = cur_layer.prev_layer
            activation_matrix = last_hidden.activation_matrix
            activation_transpose = activation_matrix.T
            weight_updates_prelim = np.dot(activation_transpose, error_temp)
            print("WEIGHT UPDATES:", weight_updates_prelim)
            weight_updates = self.learning_rate * weight_updates_prelim
            #WEIGHTS ARE ALL 1. FIX!!!
            layer_weights = []
            print(last_hidden.nodes)
            for neuron in last_hidden.nodes:
                layer_weights.append(neuron.weights)
            print(layer_weights)


    


           

            

           
        
#--------------------------------------------------------------------------------------------
# Testing Data   
#data = np.random.rand(2,5)  # Creates an array of 5 random features for testing (Last would be class)
#test = FeedForwardNN(data, 0, [], False, 1, 1)
#test.test_list()
#test.train_data()
