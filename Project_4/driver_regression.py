import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataProcess import DataProcess
from geneticAlgorithm import geneticAlg
from differentialEvolution import differentialEvolution
import feedForwardNN_GA
import feedForwardNN

# Forest Fires Data Set
DATASET_CALLED = 'forestfires'
DATASET = 'Project_3/datasets/forestfires.data'
DATASET_NAMES = ['x', 'y', 'month', 'day', 'ffmc', 'dmc', 'dc', 'isi', 'temp', 'rh', 'wind', 'rain', 'class']
DATASET_CLASS = False
NUM_NODES = [7,7]


def plot_loss_functions(zero_BP_layer, one_BP_layer, two_BP_layer,
                        zero_GA_layer, one_GA_layer, two_GA_layer,
                        zero_DE_layer, one_DE_layer, two_DE_layer,
                        zero_PS_layer, one_PS_layer, two_PS_layer,) -> None:
    '''Creates a figure with four subplots showing our results'''
    # Sets up plot for displaying 
    fig, ax = plt.subplots(4, 3, figsize=(12,10))
    fig.tight_layout(pad=3.0)
    cmap = plt.get_cmap('tab10')
    plt.subplots_adjust(left=0.16)

    #-------------------------------------------------------
    ax[0][0].set_ylabel('Backpropogation')
    ax[0][0].bar(x= range(10),
          height=[x for x in zero_BP_layer],
          color=cmap.colors, 
          width=0.5)
    
    ax[0][1].bar(x= range(10),
          height=[x for x in one_BP_layer],
          color=cmap.colors, 
          width=0.5)
    
    ax[0][2].bar(x= range(10),
          height=[x for x in two_BP_layer],
          color=cmap.colors, 
          width=0.5)
    
    ax[0][0].set_ylabel('Backpropogation', size=12)
    ax[0][0].set_title('No Hidden Layers')
    ax[0][1].set_title('One Hidden Layer')
    ax[0][2].set_title('Two Hidden Layers')

    #-------------------------------------------------------
    
    ax[1][0].bar(x= range(10),
        height=[x for x in zero_GA_layer],
        color=cmap.colors, 
        width=0.5)
    
    ax[1][1].bar(x= range(10),
        height=[x for x in one_GA_layer],
        color=cmap.colors, 
        width=0.5)
    
    ax[1][2].bar(x= range(10),
        height=[x for x in two_GA_layer],
        color=cmap.colors, 
        width=0.5)
    
    ax[1][0].set_ylabel('Genetic Algorithm', size=12)
    ax[1][0].set_title('No Hidden Layers')
    ax[1][1].set_title('One Hidden Layer')
    ax[1][2].set_title('Two Hidden Layers')

    #-------------------------------------------------------
    
    ax[2][0].bar(x= range(10),
        height=[x for x in zero_DE_layer],
        color=cmap.colors, 
        width=0.5)
    
    ax[2][1].bar(x= range(10),
        height=[x for x in one_DE_layer],
        color=cmap.colors, 
        width=0.5)
    
    ax[2][2].bar(x= range(10),
        height=[x for x in two_DE_layer],
        color=cmap.colors, 
        width=0.5)
    
    ax[2][0].set_ylabel('Differential Evolution', size=12)
    ax[2][0].set_title('No Hidden Layers')
    ax[2][1].set_title('One Hidden Layer')
    ax[2][2].set_title('Two Hidden Layers')

    #-------------------------------------------------------
    
    ax[3][0].bar(x= range(10),
        height=[x for x in zero_PS_layer],
        color=cmap.colors, 
        width=0.5)
    
    ax[3][1].bar(x= range(10),
        height=[x for x in one_PS_layer],
        color=cmap.colors, 
        width=0.5)
    
    ax[3][2].bar(x= range(10),
        height=[x for x in two_PS_layer],
        color=cmap.colors, 
        width=0.5)
    
    ax[3][0].set_ylabel('Particle Swarm', size=12)
    ax[3][0].set_title('No Hidden Layers')
    ax[3][1].set_title('One Hidden Layer')
    ax[3][2].set_title('Two Hidden Layers')
    

    labels = {}
    for i, col in enumerate(cmap.colors):
        labels['Fold ' + str(i+1)] = col
    handles = [plt.Rectangle((0,0),1,1, color=labels[label]) for label in labels.keys()]

    fig.legend(handles, labels.keys(), loc='center left')

    plt.savefig('Project_3/figures/' + DATASET_CALLED +'_fig.png')

def loss_functions(estimates:np.array, actual:np.array):
        '''Calculates preiciosn and recall'''
        sum = (actual - estimates)**2
        mse = np.mean(sum)
        print(mse)
        return mse

def main():
    # Creates a data process instance with accurate information from the .NAMES file
    data = DataProcess(names=DATASET_NAMES,cat_class=False,regression=True)

    # Loads the data set, creates the tuning set, then splits into ten folds
    data.loadCSV(DATASET)
    tuning_set = data.create_tuning_set()
    folds = data.reg_k_fold_split(10)
    
    #-------------------------------------------------------------------------------------
    # BACK PROPOGATION
    #-------------------------------------------------------------------------------------
    print("------------------------BP------------------------")
    # Hyperparameters
    learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]
    batch_sizes = [12, 43, 86, 129] #<- TODO: Must be changed with every dataset

    # Best Hyperparameter storage
    best_params_no = {}
    best_score_no = 100
    best_params_one = {}
    best_score_one = 100
    best_params_two = {}
    best_score_two = 100

    # Performs grid search
    for i in learning_rates:
        print('<-----------<')
        for j in batch_sizes:
            print('<<<<<<<<<<<<<<')
            no_loss = []
            one_loss = []
            two_loss = []
            for k in folds:
                # Creates the training and test fold. Training fold is all folds exept the one on the index.
                # This allows for 10 experiements to be run on different data.
                training_df = (pd.concat([x for x in folds if not (x.equals(k))], axis=0, ignore_index=True)).to_numpy()

                # Initilzes a network with no hidden layers, one hidden layer, and two hidden layers
                no_hidden = feedForwardNN.FeedForwardNN(inputs= training_df, hidden_layers= 0, nodes=[], classification=DATASET_CLASS,
                                           learning_rate=i, batch_size=j)
                one_hidden = feedForwardNN.FeedForwardNN(inputs= training_df, hidden_layers= 1, nodes=[NUM_NODES[0]], classification=DATASET_CLASS,
                                           learning_rate=i, batch_size=j)
                two_hidden = feedForwardNN.FeedForwardNN(inputs= training_df, hidden_layers= 2, nodes=NUM_NODES, classification=DATASET_CLASS,
                                           learning_rate=i, batch_size=j)
                
                # Trains the model on the given training set
                no_hidden.train_data()
                no_weights = no_hidden.get_chromosome()
                one_hidden.train_data()
                one_weights = one_hidden.get_chromosome()
                two_hidden.train_data()
                two_weights = two_hidden.get_chromosome()

                # Shuffles the data, trains again
                np.random.shuffle(training_df)
                no_hidden.new_inputs(training_df)
                no_hidden.train_data()
                new_no_weights = no_hidden.get_chromosome()
                one_hidden.new_inputs(training_df)
                one_hidden.train_data()
                new_one_weights = one_hidden.get_chromosome()
                two_hidden.new_inputs(training_df)
                two_hidden.train_data()
                new_two_weights = two_hidden.get_chromosome()

                # Continues this process until convergence for all models
                counter = 0
                while not np.all(np.abs(no_weights - new_no_weights) / no_weights <= 0.05):
                    if counter == 1000:
                        break
                    counter+=1
                    np.random.shuffle(training_df)
                    no_hidden.new_inputs(training_df)
                    no_hidden.train_data()
                    no_weights = new_no_weights
                    new_no_weights = no_hidden.get_chromosome()

                counter = 0
                while not np.all(np.abs(one_weights - new_one_weights) / one_weights <= 0.05):
                    if counter == 1000:
                        break
                    counter+=1
                    np.random.shuffle(training_df)
                    one_hidden.new_inputs(training_df)
                    one_hidden.train_data()
                    one_weights = new_one_weights
                    new_one_weights = one_hidden.get_chromosome()

                counter = 0
                while not np.all(np.abs(two_weights - new_two_weights) / two_weights <= 0.05):
                    if counter == 1000:
                        break
                    counter+=1
                    np.random.shuffle(training_df)
                    two_hidden.new_inputs(training_df)
                    two_hidden.train_data()
                    two_weights = new_two_weights
                    new_two_weights = two_hidden.get_chromosome()

                # Tests on the tuning set, gets loss function
                actual_zero, predicted_zero = no_hidden.test_data(tuning_set.to_numpy())
                actual_one, predicted_one = one_hidden.test_data(tuning_set.to_numpy())
                actual_two, predicted_two = two_hidden.test_data(tuning_set.to_numpy())

                no_loss.append(loss_functions(predicted_zero.astype(float), actual_zero))
                one_loss.append(loss_functions(predicted_one.astype(float), actual_one))
                two_loss.append(loss_functions(predicted_two.astype(float), actual_two))

            # If the loss(recall and precision) is better, save the hyperparameters
            score_no = np.mean(no_loss)
            if score_no <= best_score_no:
                best_params_no['learning_rate'] = i
                best_params_no['batch_size'] = j
                best_score_no = score_no
            score_one = np.mean(one_loss)
            if score_one <= best_score_one:
                best_params_one['learning_rate'] = i
                best_params_one['batch_size'] = j
                best_score_one = score_one
            score_two = np.mean(two_loss)
            if score_two <= best_score_two:
                best_params_two['learning_rate'] = i
                best_params_two['batch_size'] = j
                best_score_two = score_two

    # Storage for loss and recall
    no_bp_values = []
    one_bp_values = []
    two_bp_values = []
    
    # Uses backprop with optimal hyperparameters
    for i in folds:
        # Creates the training and test fold. Training fold is all folds exept the one on the index.
        # This allows for 10 experiements to be run on different data.
        training_df = (pd.concat([x for x in folds if not (x.equals(i))], axis=0, ignore_index=True)).to_numpy()

        # Initilzes a network with no hidden layers, one hidden layer, and two hidden layers
        no_hidden = feedForwardNN.FeedForwardNN(inputs= training_df, hidden_layers= 0, nodes=[], classification=DATASET_CLASS,
                                   learning_rate=best_params_no['learning_rate'], batch_size=best_params_no['batch_size'])
        one_hidden = feedForwardNN.FeedForwardNN(inputs= training_df, hidden_layers= 1, nodes=[NUM_NODES[0]], classification=DATASET_CLASS,
                                   learning_rate=best_params_one['learning_rate'], batch_size=best_params_one['batch_size'])
        two_hidden = feedForwardNN.FeedForwardNN(inputs= training_df, hidden_layers= 2, nodes=NUM_NODES, classification=DATASET_CLASS,
                                   learning_rate=best_params_two['learning_rate'], batch_size=best_params_two['batch_size'])
        
        # Trains the model on the given training set
        no_hidden.train_data()
        no_weights = no_hidden.get_chromosome()
        one_hidden.train_data()
        one_weights = one_hidden.get_chromosome()
        two_hidden.train_data()
        two_weights = two_hidden.get_chromosome()

        # Shuffles the data, trains again
        np.random.shuffle(training_df)
        no_hidden.new_inputs(training_df)
        no_hidden.train_data()
        new_no_weights = no_hidden.get_chromosome()
        one_hidden.new_inputs(training_df)
        one_hidden.train_data()
        new_one_weights = one_hidden.get_chromosome()
        two_hidden.new_inputs(training_df)
        two_hidden.train_data()
        new_two_weights = two_hidden.get_chromosome()

        # Continues this process until convergence for all models
        counter = 0
        while not np.all(np.abs(no_weights - new_no_weights) / no_weights <= 0.05):
            if counter == 1000:
                break
            counter+=1
            np.random.shuffle(training_df)
            no_hidden.new_inputs(training_df)
            no_hidden.train_data()
            no_weights = new_no_weights
            new_no_weights = no_hidden.get_chromosome()
        counter = 0
        while not np.all(np.abs(one_weights - new_one_weights) / one_weights <= 0.05):
            if counter == 1000:
                break
            counter+=1
            np.random.shuffle(training_df)
            one_hidden.new_inputs(training_df)
            one_hidden.train_data()
            one_weights = new_one_weights
            new_one_weights = one_hidden.get_chromosome()
        counter = 0
        while not np.all(np.abs(two_weights - new_two_weights) / two_weights <= 0.05):
            if counter == 1000:
                break
            counter+=1
            np.random.shuffle(training_df)
            two_hidden.new_inputs(training_df)
            two_hidden.train_data()
            two_weights = new_two_weights
            new_two_weights = two_hidden.get_chromosome()

        # Tests on the tuning set, gets loss function
        actual_zero, predicted_zero = no_hidden.test_data(i.to_numpy())
        actual_one, predicted_one = one_hidden.test_data(i.to_numpy())
        actual_two, predicted_two = two_hidden.test_data(i.to_numpy())
        no_loss = loss_functions(predicted_zero.astype(float), actual_zero)
        one_loss = loss_functions(predicted_one.astype(float), actual_one)
        two_loss = loss_functions(predicted_two.astype(float), actual_two)

        # Saves the loss functions across all folds
        no_bp_values.append(no_loss)
        one_bp_values.append(one_loss)
        two_bp_values.append(two_loss)
    
    #-------------------------------------------------------------------------------------
    # Population is used in all the following methods, so it is only defined once
    population_size = [10, 20, 50, 100]
    #-------------------------------------------------------------------------------------
    
    #-------------------------------------------------------------------------------------
    # GENETIC ALGORITHM
    #-------------------------------------------------------------------------------------
    print("------------------------GA------------------------")
    # Hyperparameters
    crossover_rate = [0.6, 0.7, 0.8, 0.9]
    mutation_rate = [0.05, 0.06, 0.07, 0.1]

    # Best Hyperparameter storage
    best_net_no = None
    best_score_no = 0
    best_net_one = None
    best_score_one = 0
    best_net_two = None
    best_score_two = 0

    # Performs grid search
    for i in population_size:
        print('<-----------<')
        for j in crossover_rate:
            print('<<<<<<<<<<<<<<')
            for k in mutation_rate:
                no_loss = []
                one_loss = []
                two_loss = []
                for l in folds:
                    # Creates the training and test fold. Training fold is all folds exept the one on the index.
                    # This allows for 10 experiements to be run on different data.
                    training_df = (pd.concat([x for x in folds if not (x.equals(l))], axis=0, ignore_index=True))
                    use_df = training_df.sample(n=i).to_numpy()

                    # Initializes the population
                    population_no_layers = []
                    population_one_layers = []
                    population_two_layers = []
                    for _ in range(i):
                        population_no_layers.append(feedForwardNN_GA.FeedForwardNN(inputs= use_df, hidden_layers= 0, nodes=[],
                                                                                    classification=DATASET_CLASS))
                        population_one_layers.append(feedForwardNN_GA.FeedForwardNN(inputs= use_df, hidden_layers= 1, nodes=[NUM_NODES[0]],
                                                                                    classification=DATASET_CLASS))
                        population_two_layers.append(feedForwardNN_GA.FeedForwardNN(inputs= use_df, hidden_layers= 2, nodes=NUM_NODES,
                                                                                    classification=DATASET_CLASS))
                    
                    # Gets the initial weights
                    best_network_no, best_fitness_no = geneticAlg(population_no_layers, j, k)
                    best_network_one, best_fitness_one = geneticAlg(population_one_layers, j, k)
                    best_network_two, best_fitness_two = geneticAlg(population_two_layers, j, k)

                    # Tests on the tuning set, gets loss functions
                    actual_zero, predicted_zero = best_network_no.test_data(tuning_set.to_numpy())
                    actual_one, predicted_one = best_network_one.test_data(tuning_set.to_numpy())
                    actual_two, predicted_two = best_network_two.test_data(tuning_set.to_numpy())
                    
                    no_loss.append(loss_functions(predicted_zero.astype(float), actual_zero))
                    one_loss.append(loss_functions(predicted_one.astype(float), actual_one))
                    two_loss.append(loss_functions(predicted_two.astype(float), actual_two))
                    
                    del population_no_layers
                    del population_one_layers
                    del population_two_layers

            # If the loss(recall and precision) is better, save the hyperparameters
            score_no = np.mean(no_loss)
            if score_no <= best_score_no:
                best_score_no = best_fitness_no
                best_net_no = best_network_no
            score_one = np.mean(one_loss)
            if score_one <= best_score_one:
                best_score_one = best_fitness_one
                best_net_one = best_network_one
            score_two = np.mean(two_loss)
            if score_two <= best_score_two:
                best_score_two = best_fitness_two
                best_net_two = best_network_two

    # Storage for loss and recall
    no_ga_values = []
    one_ga_values = []
    two_ga_values = []

    for i in folds:
        # Tests on the tuning set, gets loss function
        actual_zero, predicted_zero = best_net_no.test_data(i.to_numpy())
        actual_one, predicted_one = best_net_one.test_data(i.to_numpy())
        actual_two, predicted_two = best_net_two.test_data(i.to_numpy())
        no_loss = loss_functions(predicted_zero.astype(float), actual_zero)
        one_loss = loss_functions(predicted_one.astype(float), actual_one)
        two_loss = loss_functions(predicted_two.astype(float), actual_two)

        # Saves the loss functions across all folds
        no_ga_values.append(no_loss)
        one_ga_values.append(one_loss)
        two_ga_values.append(two_loss)
    
    #-------------------------------------------------------------------------------------
    # DIFFERENTIAL EVOLUTION
    #-------------------------------------------------------------------------------------
    print("------------------------DE------------------------")
    # Hyperparameters
    binary_crossover_rate = [0.5, 0.6, 0.7, 0.9]
    scaling_factor = [0.1, .5, 1.5, 2]

    # Best Hyperparameter storage
    best_net_no = None
    best_score_no = None
    best_net_one = None
    best_score_one = None
    best_net_two = None
    best_score_two = None

    # Performs grid search
    for i in population_size:
        for j in binary_crossover_rate:
            for k in scaling_factor:
                no_loss = []
                one_loss = []
                two_loss = []
                for l in folds:
                    # Creates the training and test fold. Training fold is all folds exept the one on the index.
                    # This allows for 10 experiements to be run on different data.
                    training_df = (pd.concat([x for x in folds if not (x.equals(l))], axis=0, ignore_index=True))
                    use_df = training_df.sample(n=i).to_numpy()

                    # Initializes the population
                    population_no_layers = []
                    population_one_layers = []
                    population_two_layers = []
                    for _ in range(i):
                        population_no_layers.append(feedForwardNN_GA.FeedForwardNN(inputs= use_df, hidden_layers= 0, nodes=[],
                                                                                    classification=DATASET_CLASS))
                        population_one_layers.append(feedForwardNN_GA.FeedForwardNN(inputs= use_df, hidden_layers= 1, nodes=[NUM_NODES[0]],
                                                                                    classification=DATASET_CLASS))
                        population_two_layers.append(feedForwardNN_GA.FeedForwardNN(inputs= use_df, hidden_layers= 2, nodes=NUM_NODES,
                                                                                    classification=DATASET_CLASS))
                    
                    # Gets the initial weights
                    best_network_no, best_fitness_no = differentialEvolution(population_no_layers, j, k)
                    best_network_one, best_fitness_one = differentialEvolution(population_one_layers, j, k)
                    best_network_two, best_fitness_two = differentialEvolution(population_two_layers, j, k)

                    # Tests on the tuning set, gets loss functions
                    actual_zero, predicted_zero = best_network_no.test_data(tuning_set.to_numpy())
                    actual_one, predicted_one = best_network_one.test_data(tuning_set.to_numpy())
                    actual_two, predicted_two = best_net_two.test_data(tuning_set.to_numpy())
                    
                    no_loss.append(loss_functions(predicted_zero.astype(float), actual_zero))
                    one_loss.append(loss_functions(predicted_one.astype(float), actual_one))
                    two_loss.append(loss_functions(predicted_two.astype(float), actual_two))
                        
                    del population_no_layers
                    del population_one_layers
                    del population_two_layers

            # If the loss(recall and precision) is better, save the hyperparameters
            score_no = np.mean(no_loss)
            if score_no <= best_score_no:
                best_score_no = best_fitness_no
                best_net_no = best_network_no
            score_one = np.mean(one_loss)
            if score_one <= best_score_one:
                best_score_one = best_fitness_one
                best_net_one = best_network_one
            score_two = np.mean(two_loss)
            if score_two <= best_score_two:
                best_score_two = best_fitness_two
                best_net_two = best_network_two                    
    
    # Storage for loss and recall
    no_de_values = []
    one_de_values = []
    two_de_values = []

    for i in folds:
        # Tests on the tuning set, gets loss function
        actual_zero, predicted_zero = best_net_no.test_data(i.to_numpy())
        actual_one, predicted_one = best_net_one.test_data(i.to_numpy())
        actual_two, predicted_two = best_net_two.test_data(i.to_numpy())
        no_loss = loss_functions(predicted_zero.astype(float), actual_zero)
        one_loss = loss_functions(predicted_one.astype(float), actual_one)
        two_loss = loss_functions(predicted_two.astype(float), actual_two)

        # Saves the loss functions across all folds
        no_de_values.append(no_loss)
        one_de_values.append(one_loss)
        two_de_values.append(two_loss)

    #-------------------------------------------------------------------------------------
    # PARTICLE SWARM
    #-------------------------------------------------------------------------------------
    print("------------------------PS------------------------")

    #-------------------------------------------------------------------------------------
    # FIGURE GENERATION
    #-------------------------------------------------------------------------------------
    plot_loss_functions(no_bp_values, one_bp_values, two_bp_values,
                        no_ga_values, one_ga_values, two_ga_values,
                        no_de_values, one_de_values, two_de_values,
                        None, None, None) # <- TODO: replace this with PSO lists of values
    
if __name__ == '__main__':
    main()