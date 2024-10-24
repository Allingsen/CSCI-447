import numpy as np
from layers import Layer

class FeedForwardNN():
    def __init__(self, inputs: np.array, hidden_layers: int, nodes: int, classification: bool, learning_rate: float, batch_size: int, num_of_classes: int=1):
        self.inputs = inputs
        self.hidden_layers = hidden_layers
        self.nodes = nodes
        self.classification = classification
        self.num_of_classes = num_of_classes
        if self.num_of_classes != 1 and not self.classification:
            raise AttributeError('Regression should only have 1 output')
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Create the input layer as the length of the example, minus the class
        self.inputLayer = Layer(0, len(self.inputs[0])-1, None, None)
        self.create_layers()

    def create_layers(self):
        '''Creates the Entire Network'''
        for _ in range(self.hidden_layers):
            # Initlizes the Linked list
            cur_layer = self.inputLayer
            # Gets to the end
            while cur_layer.next_layer:
                cur_layer = cur_layer.next_layer
            # Inserts the new layer at the end
            new_layer = Layer(1, self.nodes, cur_layer, None)
            cur_layer.next_layer = new_layer

        # Adds the final output layer depending on type of network (Classification or Regression)
        cur_layer = cur_layer.next_layer
        if self.classification:
            cur_layer.next_layer = Layer(3, self.num_of_classes, cur_layer, None)
        else:
            cur_layer.next_layer = Layer(2, 1, cur_layer, None)

    def test_list(self):
        '''Prints Network Information, only used for manual Checking'''
        print('-----Data-----\n', self.inputs)
        print('-----List Test-----')
        cur_layer = self.inputLayer
        while cur_layer:
            print(f'First Layer Type: {cur_layer.version}, Number of Nodes: {cur_layer.num_nodes}, Number of Inputs to each Node: {len(cur_layer.nodes[0].weights)}')
            cur_layer = cur_layer.next_layer

    def train_data(self):
        '''Feeds data through the network, tuning weights using minibatch back prop'''
        # Counter used for mini-batches
        counter = 0
        # Saves the predicted values and the acutal values
        actual_val = np.zeros(self.batch_size)
        predicted_val = np.zeros(self.batch_size)
        # Iterates through all inputs
        for i in self.inputs:
            actual_val[counter] = i[-1]
            predicted_val[counter] = self.get_prediction(i[:-1])
            # If the minibatch has been iterated through, perform backprop
            if counter == self.batch_size:
                # TODO: Perform Back Prop Here!
                counter = 0

    def get_prediction(self, point: np.array) -> float:
        '''Gets the predictions of the network at a specified point'''
        cur_layer = self.inputLayer
        inputs = point
        # Iterates through the network, passing the previous layers output to the next lauer
        while cur_layer:
            inputs = cur_layer.generate_output(inputs)
            cur_layer = cur_layer.next_layer
        # Returns the final prediciton
        return inputs[0]
    
