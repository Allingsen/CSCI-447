import numpy as np
import math
import random
from layers import Layer

probabilities_list = []

class NeuralNetworkPSO():
    def __init__(self, inputs: np.array, hidden_layers: int, nodes: list, classification: bool, inertial_weight: float, inertia_max: float, inertia_min: float, max_epochs: int, cognitive_weight: float, social_weight: float, batch_size: int, max_velocity: float, num_of_classes: int=1, class_names=None):
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
        self.batch_size = batch_size
        self.position = None
        self.velocity = []
        self.inertial_weight = inertial_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.gbest = None
        self.pbest = None
        self.r_1 = random.random()
        self.r_2 = random.random()
        self.vect_representation = None
        self.pbest_performance = float('inf')
        self.matrix_shapes = []
        self.max_velocity = max_velocity
        self.inertia_max = inertia_max
        self.inertia_min = inertia_min
        self.iteration = 0
        self.max_epochs = max_epochs

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

        #Store weight matrix shapes into a data field for later reconstruction from 1D position vector
        cur_layer = self.inputLayer.next_layer
        while(cur_layer != None):
            self.matrix_shapes.append(cur_layer.weight_matrix.shape)
            cur_layer = cur_layer.next_layer

    def new_inputs(self, inputs:np.array):
        self.inputs = inputs

    #This function turns the weight matrices in the network into a 1D vector that represents position in the weight
    #space
    def vectorize(self):
        flat_list = []
        #Put the layer weights into the vector
        cur_layer = self.inputLayer.next_layer
        while(cur_layer != None):
            flat_list.append(cur_layer.weight_matrix.flatten())
            cur_layer = cur_layer.next_layer

        self.vect_representation = np.concatenate(flat_list)
        return(self.vect_representation)
            
    def test_list(self):
        '''Prints Network Information, only used for manual Checking'''
        print('-----Data-----\n', self.inputs)
        print('-----List Test-----')
        cur_layer = self.inputLayer
        while cur_layer:
            print(f'First Layer Type: {cur_layer.version}, Number of Nodes: {cur_layer.num_nodes}, Number of Inputs to each Node: {len(cur_layer.nodes[0].weights)}')
            cur_layer = cur_layer.next_layer

    #Feeds data through the network and obtains predicted and real values
    def test_data(self, examples: np.array) -> tuple:
        actual_val = np.array([None]*len(examples))
        predicted_val = np.array([None]*len(examples))
        for i, val in enumerate(examples):
            actual_val[i] = val[-1]
            predicted_val[i] = self.get_prediction(val[:-1])

        return actual_val, predicted_val    

    #
    def get_fitness(self):
        cur_layer = self.inputLayer
        while(cur_layer.next_layer != None):
            cur_layer = cur_layer.next_layer
        output_layer = cur_layer
        # Counter used for mini-batches
        counter = 0
        # Saves the predicted values and the actual values
        actual_val = np.array([None]*self.batch_size)
        predicted_val = np.array([None]*self.batch_size)
        all_actual = np.array([None]*len(self.inputs))
        all_pred = np.array([None]*len(self.inputs))
        # Iterates through all inputs
        for no,i in enumerate(self.inputs):
            actual_val[counter] = i[-1]
            predicted_val[counter] = self.get_prediction(i[:-1])
            all_actual[no] = i[-1]
            all_pred[no] = predicted_val[counter]
            counter += 1
            # If the minibatch has been iterated through, update pbest and obtain the error to be used for fitness
            if counter == self.batch_size:             
                probabilities_list = output_layer.get_probabilities()
                error_signal = self.error_signal(predicted_val, actual_val, probabilities_list)
                if(self.classification):
                    error_signal_perf = sum(sum(error_signal))
                else:
                    error_signal_perf = sum(error_signal)
                if(error_signal_perf < self.pbest_performance):
                    self.pbest_performance = error_signal_perf
                    self.pbest = self.vectorize()
                probabilities_list.clear()
                output_layer.probabilities.clear()
                counter = 0

        return(error_signal_perf)
    
    def update(self):
        #Get the new velocity, and then set the new position using this velocity
        wv = []
        cognitive = []
        social = []
        new_vel = []

        #Velocity update equation
        for i in range(len(self.velocity)):
            wv.append(self.inertial_weight * self.velocity[i])
        for i in range(len(self.position)):
            cognitive.append((self.cognitive_weight * self.r_1) * (self.pbest[i] - self.position[i]))
        for i in range(len(self.gbest)):
            social.append((self.social_weight * self.r_2) * (self.gbest[i] - self.position[i]))
        for i in range(len(self.velocity)):
            val = wv[i] + cognitive[i] + social[i]

            #Velocity clamping
            if(abs(val) > self.max_velocity):
                if(val < 0):
                    val = -self.max_velocity
                else:
                    val = self.max_velocity
            new_vel.append(val)

        #Velocity update
        self.velocity = new_vel

        #Position update
        for i in range(len(self.velocity)):
            self.position[i] = self.position[i] + self.velocity[i]

        #Update current epoch
        self.iteration += 1

        #Linearly decrease inertial weight to focus more on exploitation over time
        self.inertial_weight = self.inertia_max - ((self.inertia_max - self.inertia_min)/(self.max_epochs)) * self.iteration

        #Reconstruct the weight matrices from the 1D position vector
        start = 0
        reconstructed_matrices = []
        for shape in self.matrix_shapes:
            size = np.prod(shape)
            weight_matrix = self.position[start:start + size].reshape(shape)
            reconstructed_matrices.append(weight_matrix)
            start += size

        #Run through each layer in the network and update the weight matrices (set the new position in the weight space)
        cur_layer = self.inputLayer.next_layer
        counter = 0
        while(cur_layer != None):
            cur_layer.weight_matrix = reconstructed_matrices[counter]
            counter += 1
            cur_layer = cur_layer.next_layer

        #Randomly select r_1 and r_2 from a uniform distribution for the next iteration
        self.r_1 = random.random()
        self.r_2 = random.random()

    #Get the predicted value for one data point
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
                    error_matrix.append(np.abs(to_enter))
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
                    error_matrix.append(np.abs(probabilities_list[i] - y_true))

                return(error_matrix)

        #If regression, use Mean Squared Error derivative
        else:
            error_vector = []
            for i in range(len(predicted_values)):
                error_vector.append(np.abs(predicted_values[i] - actual_values[i]))
            return(error_vector)

    #Gets the weights feeding into the output layer
    def get_weights(self):
        cur_layer = self.inputLayer
        while(cur_layer.next_layer != None):
            cur_layer = cur_layer.next_layer
        return(cur_layer.weight_matrix)
    
    #Prints the current state of the network (layers, weights, activations, etc.)
    def get_state(self):
        cur_layer = self.inputLayer
        while(cur_layer != None):
            if(cur_layer.version == 0):
                print("INPUT LAYER:\n----------")
                print("Inputs:", self.inputs, "\n")
                print("Outputs:", cur_layer.activation_matrix, "\n")
                print("Weight Matrix: None\n")
            elif(cur_layer.version == 1):
                print("HIDDEN LAYER:\n---------")
                print("Inputs:", cur_layer.prev_layer.activation_matrix, "\n")
                print("Outputs:", cur_layer.activation_matrix, "\n")
                print("Weight Matrix:", cur_layer.weight_matrix)
            else:
                print("OUTPUT LAYER:\n---------")
                print("Inputs:", cur_layer.prev_layer.activation_matrix, "\n")
                print("Outputs:", cur_layer.activation_matrix, "\n")
                print("Weight Matrix:", cur_layer.weight_matrix)
            cur_layer = cur_layer.next_layer
