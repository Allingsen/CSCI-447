import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataProcess import DataProcess
from geneticAlgorithm import geneticAlg
import feedForwardNN_GA
import feedForwardNN

DATASET_CALLED = 'breast-cancer'
DATASET = 'datasets/breast-cancer-wisconsin.data'
DATASET_NAMES = ['id', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'class']
CLASS_NAMES = ['2', '4']
NUM_CLASSES = len(CLASS_NAMES)
NUM_NODES = [4,4]
DATASET_CLASS = True

def plot_loss_functions(zero_layer: list, one_layer: list, two_layer:list) -> None:
    '''Creates a figure with four subplots showing our results'''
    # Sets up plot for displaying 
    fig, ax = plt.subplots(1, 4, figsize=(10,4))
    fig.tight_layout(pad=5.0)
    cmap = plt.get_cmap('tab10')
    plt.subplots_adjust(left=0.2)
    # Creates the space where the bars will be placed
    recall = np.linspace(0.7, 1.3, 10)
    precision = np.linspace(1.7, 2.3, 10) 

    # Creates the tick marking
    ticks = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    ax[0].bar(x= recall,
          height=[x[0] for x in zero_layer],
          color=cmap.colors, 
          width=0.05)
    
    ax[0].bar(x= precision, 
              height=[x[1] for x in zero_layer], 
              color=cmap.colors, 
              width=0.05)
    
    ax[0].set_xticks([1,2])
    ax[0].set_xticklabels(['Recall','Precision'])
    ax[0].set_ylabel('Percentage')
    ax[0].set_xlabel('Fold')
    ax[0].set_yticks([x/100 for x in ticks])
    ax[0].set_yticklabels(ticks)
    ax[0].set_title('No Hidden Layers')
    
    ax[1].bar(x= recall,
      height=[x[0] for x in one_layer],
      color=cmap.colors, 
      width=0.05)

    ax[1].bar(x= precision, 
              height=[x[1] for x in one_layer], 
              color=cmap.colors, 
              width=0.05)
    
    ax[1].set_xticks([1,2])
    ax[1].set_xticklabels(['Recall','Precision'])
    ax[1].set_ylabel('Percentage')
    ax[1].set_xlabel('Fold')
    ax[1].set_yticks([x/100 for x in ticks])
    ax[1].set_yticklabels(ticks)
    ax[1].set_title('One Hidden Layer')

    ax[2].bar(x= recall,
          height=[x[0] for x in two_layer],
          color=cmap.colors, 
          width=0.05)
    
    ax[2].bar(x= precision, 
              height=[x[1] for x in two_layer], 
              color=cmap.colors, 
              width=0.05)
    
    ax[2].set_xticks([1,2])
    ax[2].set_xticklabels(['Recall','Precision'])
    ax[2].set_ylabel('Percentage')
    ax[2].set_xlabel('Fold')
    ax[2].set_yticks([x/100 for x in ticks])
    ax[2].set_yticklabels(ticks)
    ax[2].set_title('Two Hidden Layers')

    labels = {}
    for i, col in enumerate(cmap.colors):
        labels['Fold ' + str(i+1)] = col
    handles = [plt.Rectangle((0,0),1,1, color=labels[label]) for label in labels.keys()]

    fig.legend(handles, labels.keys(), loc='center left')

    plt.savefig('Project_3/figures/' +DATASET_CALLED+'_fig.png')

def loss_functions(estimates:np.array, actual:np.array):
        '''Calculates preiciosn and recall'''
        all_recall = []
        all_prec = []
        for Class in np.unique(actual):
            #confusion_local represents the TPs, TNs, FPs, and FNs for a specific class
            confusion_local = np.zeros(4)
            for i in range(len(estimates)):
                if(actual[i] == Class and estimates[i] == Class):
                    confusion_local[0] += 1
                elif(actual[i] != Class and estimates[i] != Class):
                    confusion_local[1] += 1
                elif(actual[i] != Class and estimates[i] == Class):
                    confusion_local[2] += 1
                else:
                    confusion_local[3] += 1
            if confusion_local[0] == 0 and confusion_local[3] == 0:
                all_recall.append(0)
            else:
                all_recall.append(confusion_local[0]/(confusion_local[0] + confusion_local[3]))
            if confusion_local[0] == 0 and confusion_local[2] == 0:
                all_prec.append(0)
            else:
                all_prec.append(confusion_local[0]/(confusion_local[0] + confusion_local[2]))

        recall = np.mean(all_recall)
        precision = np.mean(all_prec)
        print((recall, precision))
        return(recall, precision)

def main():
    # Creates a data process instance with accurate information from the .NAMES file
    data = DataProcess(names=DATASET_NAMES, cat_class=True, id_col='id', missing_val='?') #<- TODO: Must be changed with every dataset

    # Loads the data set, creates the tuning set, then splits into ten folds
    data.loadCSV(DATASET)
    tuning_set = data.create_tuning_set()
    folds = data.k_fold_split(10)

    #-------------------------------------------------------------------------------------
    # BACK PROPOGATION
    #-------------------------------------------------------------------------------------
    # Hyperparameters
    learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]
    batch_sizes = [2, 349] #<- TODO: Must be changed with every dataset

    # Best Hyperparameter storage
    best_params_no = {}
    best_score_no = 0
    best_params_one = {}
    best_score_one = 0
    best_params_two = {}
    best_score_two = 0

    # Performs grid search
    for i in learning_rates:
        for j in batch_sizes:
            for k in folds:
                # Creates the training and test fold. Training fold is all folds exept the one on the index.
                # This allows for 10 experiements to be run on different data.
                training_df = (pd.concat([x for x in folds if not (x.equals(k))], axis=0, ignore_index=True)).to_numpy()

                # Initilzes a network with no hidden layers, one hidden layer, and two hidden layers
                no_hidden = feedForwardNN.FeedForwardNN(inputs= training_df, hidden_layers= 0, nodes=[], classification=DATASET_CLASS,
                                           learning_rate=i, batch_size=j, 
                                           num_of_classes=NUM_CLASSES, class_names=CLASS_NAMES)
                one_hidden = feedForwardNN.FeedForwardNN(inputs= training_df, hidden_layers= 1, nodes=[NUM_NODES[0]], classification=DATASET_CLASS,
                                           learning_rate=i, batch_size=j,
                                           num_of_classes=NUM_CLASSES, class_names=CLASS_NAMES)
                two_hidden = feedForwardNN.FeedForwardNN(inputs= training_df, hidden_layers= 2, nodes=NUM_NODES, classification=DATASET_CLASS,
                                           learning_rate=i, batch_size=j, 
                                           num_of_classes=NUM_CLASSES,class_names=CLASS_NAMES)
                
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

                no_loss = loss_functions(predicted_zero.astype(float), actual_zero)
                one_loss = loss_functions(predicted_one.astype(float), actual_one)
                two_loss = loss_functions(predicted_two.astype(float), actual_two)

                # If the loss(recall and precision) is better, save the hyperparameters
                score_no = np.mean(no_loss)
                if score_no > best_score_no:
                    best_params_no['learning_rate'] = i
                    best_params_no['batch_size'] = j
                    best_score_no = score_no

                score_one = np.mean(one_loss)
                if score_one > best_score_one:
                    best_params_one['learning_rate'] = i
                    best_params_one['batch_size'] = j
                    best_score_one = score_one

                score_two = np.mean(two_loss)
                if score_two > best_score_two:
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
                                   learning_rate=best_params_no['learning_rate'], batch_size=best_params_no['batch_size'], 
                                   num_of_classes=NUM_CLASSES, class_names=CLASS_NAMES)
        one_hidden = feedForwardNN.FeedForwardNN(inputs= training_df, hidden_layers= 1, nodes=[NUM_NODES[0]], classification=DATASET_CLASS,
                                   learning_rate=best_params_one['learning_rate'], batch_size=best_params_one['batch_size'],
                                   num_of_classes=NUM_CLASSES, class_names=CLASS_NAMES)
        two_hidden = feedForwardNN.FeedForwardNN(inputs= training_df, hidden_layers= 2, nodes=NUM_NODES, classification=DATASET_CLASS,
                                   learning_rate=best_params_two['learning_rate'], batch_size=best_params_two['batch_size'], 
                                   num_of_classes=NUM_CLASSES,class_names=CLASS_NAMES)
        
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
    # Hyperparameters
    crossover_rate = [0.6, 0.7, 0.8, 0.9]
    mutation_rate = [0.05, 0.06, 0.07, 0.1]

    # Best Hyperparameter storage
    best_net_no = None
    best_score_no = None
    best_net_one = None
    best_score_one = None
    best_net_two = None
    best_score_two = None

    # Performs grid search
    for i in population_size:
        for j in crossover_rate:
            for k in mutation_rate:
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
                                                                                    classification=DATASET_CLASS, num_of_classes=NUM_CLASSES, class_names=CLASS_NAMES))
                        population_one_layers.append(feedForwardNN_GA.FeedForwardNN(inputs= use_df, hidden_layers= 1, nodes=[NUM_NODES[0]],
                                                                                    classification=DATASET_CLASS, num_of_classes=NUM_CLASSES, class_names=CLASS_NAMES))
                        population_two_layers.append(feedForwardNN_GA.FeedForwardNN(inputs= use_df, hidden_layers= 2, nodes=NUM_NODES,
                                                                                    classification=DATASET_CLASS, num_of_classes=NUM_CLASSES, class_names=CLASS_NAMES))
                    
                    # Gets the initial weights
                    best_network_no, best_fitness_no = geneticAlg(population_no_layers, j, k)
                    best_network_one, best_fitness_one = geneticAlg(population_one_layers, j, k)
                    best_network_two, best_fitness_two = geneticAlg(population_two_layers, j, k)

                    # Checks for the best network, by fitness score
                    if not best_score_no or best_fitness_no > best_score_no:
                        best_score_no = best_fitness_no
                        best_net_no = best_network_no

                    if not best_score_one or best_fitness_one > best_score_one:
                        best_score_one = best_fitness_one
                        best_net_one = best_network_one

                    if not best_score_two or best_fitness_two > best_score_two:
                        best_score_two = best_fitness_two
                        best_net_two = best_network_two
                        
                    del population_no_layers
                    del population_one_layers
                    del population_two_layers

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
    # Hyperparameters
    scaling_factor = []
    binary_crossover_rate = []

    # Best Hyperparameter storage

    # Performs grid search
    for i in population_size:
        for j in scaling_factor:
            for k in binary_crossover_rate:
                for l in folds:
                    pass
    
    # Storage for loss and recall
    no_de_values = []
    one_de_values = []
    two_de_values = []

    for i in folds:
        pass

    #-------------------------------------------------------------------------------------
    # PARTICLE SWARM
    #-------------------------------------------------------------------------------------
    
if __name__ == '__main__':
    main()