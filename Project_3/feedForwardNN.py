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

        #If regression, use Mean Squared Error derivative
        else:
            error_vector = []
            for i in range(len(predicted_values)):
                error_vector.append((-2/self.batch_size) * (predicted_values[i] - actual_values[i]))
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

           #Update weights running into output neuron
            cur_layer = self.inputLayer
            while(cur_layer.next_layer != None):
               cur_layer = cur_layer.next_layer
            
            last_hidden = cur_layer.prev_layer
            activation_matrix = last_hidden.activation_matrix
            activation_transpose = activation_matrix.T
            print("ACTIVATION TRANSPOSE:", activation_transpose)
            #REPLACE WITH error_signal WHEN DONE TESTING
            print("ERROR:", error_temp)
            weight_updates_prelim = np.dot(activation_transpose, error_temp)
            print("WEIGHT UPDATES:", weight_updates_prelim)
            weight_updates = self.learning_rate * weight_updates_prelim
            print("WEIGHT GRADIENTS:", weight_updates)
            output_neuron = cur_layer

            #Update Weights feeding into output neuron
            output_neuron.nodes[0].weights = output_neuron.nodes[0].weights - weight_updates
            print("NEW WEIGHTS:", output_neuron.nodes[0].weights)

            #Update the rest of the weights
            while(cur_layer.prev_layer != self.inputLayer):
                
                #Obtain weight matrix for weights connecting this layer to the next layer
                weight_matrix_forward = np.zeros((cur_layer.num_nodes, cur_layer.prev_layer.num_nodes))
                i = 0
                for neuron in cur_layer.nodes:
                    weight_matrix_forward[i] = neuron.weights
                    i += 1
                print("WEIGHT MATRIX FORWARD:", weight_matrix_forward)
                prev_activation = cur_layer.prev_layer.prev_layer.activation_matrix

                #Obtain previous layer's activation
                print("PREV ACTIVATION:", prev_activation)

                #Obtain weight matrix for weights connecting the previous layer to this layer
                weight_matrix_backward = np.zeros((cur_layer.prev_layer.num_nodes, cur_layer.prev_layer.prev_layer.num_nodes))
                i = 0
                for neuron in cur_layer.prev_layer.nodes:
                    weight_matrix_backward[i] = neuron.weights
                    i += 1
                print("WEIGHT MATRIX BACKWARD:", weight_matrix_backward)

                #A^(l-1)
                pre_activation = prev_activation @ weight_matrix_backward.T
                print("PRE_ACTIVATION:", pre_activation)

                #A^(l)
                cur_activation = 1/(1 + np.exp(-pre_activation))
                print("CURRENT ACTIVATION:", cur_activation)

                #derivative of activation function...
                logistic_deriv = cur_activation * (1 - cur_activation)
                print("LOGISTIC DERIVATIVE:", logistic_deriv)

                #Get delta^l
                print("ERROR TEMP:", error_temp)
                temp_arr = np.zeros((1, 3))
                temp_arr[0] = np.array(error_temp)

                #Need to take Hadamard product but dimensions don't align!!!
                print(logistic_deriv.shape)
                cur_error = np.dot(weight_matrix_forward.T, temp_arr) * logistic_deriv
                error_temp = cur_error

                #At this point we have delta^l and the old weights, need A^(l-1) transpose
                pre_activation_transpose = pre_activation.T

                #Update the weights
                rows1, columns1 = pre_activation_transpose.shape
                rows2, columns2 = cur_error.shape
                print(f"SHAPE 1: {rows1}x{columns1}")
                print(f"SHAPE 2: {rows2}x{columns2}")
                update_to = weight_matrix_backward - (self.learning_rate * (1/self.batch_size) * (pre_activation_transpose @ cur_error))
                i = 0
                for neuron in cur_layer.prev_layer.nodes:
                    neuron.weights = update_to[i]
                    i += 1
                
                cur_layer = cur_layer.prev_layer
   
        
#--------------------------------------------------------------------------------------------
# Testing Data   
#data = np.random.rand(2,5)  # Creates an array of 5 random features for testing (Last would be class)
#test = FeedForwardNN(data, 0, [], False, 1, 1)
#test.test_list()
#test.train_data()
