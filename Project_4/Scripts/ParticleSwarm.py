#Hyperparameters: cognitive weight, social weight, population size
from NeuralNetworkPSO import NeuralNetworkPSO
import numpy as np
import pandas as pd
import random
import time

convergence = 0.01

class ParticleSwarm():
    def __init__(self, inertial_weight, inertia_max, inertia_min, max_epochs, cognitive_weight, social_weight, population_size, batch_size, inputs, hidden_layers, nodes, classification, num_of_classes, class_names, max_velocity):
        self.inertial_weight = inertial_weight
        self.inertia_max = inertia_max
        self.inertia_min = inertia_min
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.batch_size = batch_size
        self.population_size = population_size
        self.population = []
        self.inputs = inputs
        self.hidden_layers = hidden_layers
        self.nodes = nodes
        self.classification = classification
        self.num_of_classes = num_of_classes
        self.class_names = class_names
        self.max_velocity = max_velocity
        self.max_epochs = max_epochs
        self.epoch = 0

    #This method initializes the population of neural networks
    def initialize_population(self):
        #Generate a population of networks with size self.population_size
        for i in range(self.population_size):
            NN = NeuralNetworkPSO(self.inputs, self.hidden_layers, self.nodes, self.classification, self.inertial_weight, self.inertia_max, self.inertia_min, self.max_epochs, self.cognitive_weight, self.social_weight, self.batch_size, self.max_velocity, self.num_of_classes, self.class_names)
            NN.position = NN.vectorize()
            velocity = []
            #Initialize velocities to small randomized vectors so that the first gbest doesn't get stuck at the
            #origin
            for i in range(len(NN.position)):
                rand = random.uniform(-0.1, 0.1)
                velocity.append(rand)
            NN.velocity = velocity
            self.population.append(NN)

    #This method performs the operations of the PSO algorithm. This includes calculating fitnesses, calculating and
    #sharing the gbest, updating positions of each network, and checking for convergence.
    def swarm(self):
        converged = False

        #While the positions of each network change by greater than 0.1%...
        while(not converged):
            print(f"EPOCH {self.epoch}")
            fitnesses = {}
            old_positions = []
            new_positions = []
            counter = 0

            #Get the fitness of each network in the population and add the current positon to a list 
            #(for convergence check)
            for NN in self.population:
                fitnesses[counter] = NN.get_fitness()
                old_positions.append(np.copy(NN.position))
                counter += 1

            #Select the global best to be the network with the lowest error
            gbest = self.population[min(fitnesses, key=fitnesses.get)].position

            #Run through each network in the population and share the global best position
            for NN in self.population:
                NN.gbest = gbest

            #Run through each network in the population and update their positions using the velocity update equation:
            #v_new = (w * v_old) + ((c_1 * r_1) * (pbest - position)) + ((c_2 * r_2) * (gbest - position))
            for NN in self.population:
                NN.update()

            #Get the new positions so we can check for convergence
            for NN in self.population:
                new_positions.append(NN.position)

            #Check to see if the weights have converged within a 0.01 threshold
            old_positions = np.array(old_positions)
            new_positions = np.array(new_positions)
            difference = new_positions - old_positions
            magnitude = np.linalg.norm(difference, axis=1)

            #If the weights change less than 0.001 or the maximum number of epochs has been surpasses, converge
            converged = np.all(magnitude < convergence)
            if(self.epoch >= self.max_epochs):
               converged = True

            self.epoch += 1

        print(f"Finished at epoch {self.epoch - 1}")

    #This method selects the best model in the population after PSO and returns it for testing
    def select_model(self):
        counter = 0
        fitnesses = {}
        for model in self.population:
            fitnesses[counter] = model.get_fitness()
            counter += 1
        best = self.population[min(fitnesses, key=fitnesses.get)]
        return(best)
