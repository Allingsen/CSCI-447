import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from dataProcess import DataProcess
from feedForwardNN import FeedForwardNN
 
DATASET_CALLED = 'forestfires'
DATASET = 'Project_3/datasets/forestfires.data'
DATASET_NAMES = ['x', 'y', 'month', 'day', 'ffmc', 'dmc', 'dc', 'isi', 'temp', 'rh', 'wind', 'rain', 'class']
DATASET_CLASS = False

def plot_loss_functions(knn_vals: list, enn_vals: list, kmeans_vals:list) -> None:
    '''Creates a figure with two subplots showing our results'''
    # Sets up plot for displaying 
    fig, ax = plt.subplots(1, 3, figsize=(12,4))
    fig.tight_layout(pad=5.0)
    cmap = plt.get_cmap('tab10')
    plt.subplots_adjust(left=0.2)
    ax[0].bar(x= range(10),
          height=knn_vals,
          color=cmap.colors, 
          width=0.5)

    ax[0].set_ylabel('0/1 Loss')
    ax[0].set_xlabel('Fold')
    ax[0].set_title('No Hidden Layers')
    
    ax[1].bar(x= range(10),
          height=enn_vals,
          color=cmap.colors, 
          width=0.5)
    
    ax[1].set_ylabel('0/1 Loss')
    ax[1].set_xlabel('Fold')
    ax[1].set_title('One Hidden Layer')

    ax[2].bar(x= range(10),
          height=kmeans_vals,
          color=cmap.colors, 
          width=0.5)

    ax[2].set_ylabel('0/1 Loss')
    ax[2].set_xlabel('Fold')
    ax[2].set_title('Two Hidden Layers')

    labels = {}
    for i, col in enumerate(cmap.colors):
        labels['Fold ' + str(i+1)] = col
    handles = [plt.Rectangle((0,0),1,1, color=labels[label]) for label in labels.keys()]

    fig.legend(handles, labels.keys(), loc='center left')

    plt.savefig('Project_3/figures/' +DATASET_CALLED+'_fig.png')

def loss_functions(estimates:np.array, actual:np.array, epsilon:float):
    loss = 0
    for i in range(len(estimates)):
        if (actual[i] - (epsilon * actual[i])) <= estimates[i] <= (actual[i] + (epsilon * actual[i])):
            pass
        else:
            loss += 1
    print(loss)
    return loss

def main():
    # Creates a data process instance with accurate information from the .NAMES file
    data = DataProcess(names=DATASET_NAMES,cat_class=False,regression=True)
    # Loads the data set, creates the tuning set, then splits into ten folds
    data.loadCSV(DATASET)
    tuning_set = data.create_tuning_set()
    folds = data.reg_k_fold_split(10)
    
    # Iterates through, tests on the tuning set
    learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]
    batch_sizes = [12, 43, 86, 129]
    no_nodes = [*range(1, 12)]

    best_params_no = {}
    best_score_no = 1000
    best_params_one = {}
    best_score_one = 1000
    best_params_two = {}
    best_score_two = 1000

    for i in folds:
        # Creates the training and test fold. Training fold is all folds exept the one on the index.
        # This allows for 10 experiements to be run on different data.
        learning_rate = random.choice(learning_rates)
        batch_size = random.choice(batch_sizes)
        num_nodes = [random.choice(no_nodes), random.choice(no_nodes)]
        training_df = (pd.concat([x for x in folds if not (x.equals(i))], axis=0, ignore_index=True)).to_numpy()

        no_hidden = FeedForwardNN(inputs= training_df, hidden_layers= 0, nodes=[], classification=DATASET_CLASS,
                                   learning_rate=learning_rate, batch_size=batch_size)
        one_hidden = FeedForwardNN(inputs= training_df, hidden_layers= 1, nodes=[num_nodes[0]], classification=DATASET_CLASS,
                                   learning_rate=learning_rate, batch_size=batch_size)
        two_hidden = FeedForwardNN(inputs= training_df, hidden_layers= 2, nodes=[num_nodes[0], num_nodes[1]], classification=DATASET_CLASS,
                                   learning_rate=learning_rate, batch_size=batch_size)
        predicted_zero, actual_zero = no_hidden.train_data()
        predicted_one, actual_one =one_hidden.train_data()
        predicted_two, actual_two =two_hidden.train_data()

        no_loss = loss_functions(predicted_zero, actual_zero, 0.1)
        one_loss = loss_functions(predicted_one, actual_one, 0.1)
        two_loss = loss_functions(predicted_two, actual_two, 0.1)

        np.random.shuffle(training_df)
        no_hidden.train_data()
        one_hidden.train_data()
        two_hidden.train_data()

        no_hidden.new_inputs(training_df)
        no_loss_new = loss_functions(predicted_zero, actual_zero, 0.1)
        one_hidden.new_inputs(training_df)
        one_loss_new = loss_functions(predicted_one, actual_one, 0.1)
        two_hidden.new_inputs(training_df)
        two_loss_new = loss_functions(predicted_two, actual_two, 0.1)

        counter = 0
        while no_loss > no_loss_new:
            if counter == 1000:
                break
            counter+=1
            np.random.shuffle(training_df)
            no_hidden.new_inputs(training_df)
            actual, pred = no_hidden.train_data()
            no_loss = no_loss_new
            no_loss_new = loss_functions(actual, pred, 0.1)
        
        counter = 0
        while one_loss > one_loss_new:
            if counter == 1000:
                break
            counter+=1
            np.random.shuffle(training_df)
            one_hidden.new_inputs(training_df)
            actual, pred = one_hidden.train_data()
            one_loss = one_loss_new
            one_loss_new = loss_functions(actual, pred, 0.1)

        counter = 0
        while two_loss > two_loss_new:
            if counter == 1000:
                break
            counter+=1
            np.random.shuffle(training_df)
            two_hidden.new_inputs(training_df)
            actual, pred = no_hidden.train_data()
            two_loss = two_loss_new
            two_loss_new = loss_functions(actual, pred, 0.1)

        actual_zero, predicted_zero = no_hidden.test_data(tuning_set.to_numpy())
        actual_one, predicted_one = one_hidden.test_data(tuning_set.to_numpy())
        actual_two, predicted_two = two_hidden.test_data(tuning_set.to_numpy())
        
        print('+++++++++++++')
        no_loss = loss_functions(predicted_zero, actual_zero, 0.1)
        one_loss = loss_functions(predicted_one, actual_one, 0.1)
        two_loss = loss_functions(predicted_two, actual_two, 0.1)
        print('+++++++++++++')
        if no_loss < best_score_no:
            best_params_no['learning_rate'] = learning_rate
            best_params_no['batch_size'] = batch_size
            best_score_no = no_loss

        if one_loss < best_score_one:
            best_params_one['learning_rate'] = learning_rate
            best_params_one['batch_size'] = batch_size
            best_params_one['nodes'] = [num_nodes[0]]
            best_score_one = one_loss
       
        if two_loss < best_score_two:
            best_params_two['learning_rate'] = learning_rate
            best_params_two['batch_size'] = batch_size
            best_params_two['nodes'] = num_nodes
            best_score_two = two_loss
    
    print('-------------------------------------------')
    no_values = []
    one_values = []
    two_values = []
    for i in folds:
        # Creates the training and test fold. Training fold is all folds exept the one on the index.
        # This allows for 10 experiements to be run on different data.
        training_df = (pd.concat([x for x in folds if not (x.equals(i))], axis=0, ignore_index=True)).to_numpy()

        no_hidden = FeedForwardNN(inputs= training_df, hidden_layers= 0, nodes=[], classification=DATASET_CLASS,
                                   learning_rate=best_params_no['learning_rate'], batch_size=best_params_no['batch_size'])
        one_hidden = FeedForwardNN(inputs= training_df, hidden_layers= 1, nodes=best_params_one['nodes'], classification=DATASET_CLASS,
                                   learning_rate=best_params_one['learning_rate'], batch_size=best_params_one['batch_size'])
        two_hidden = FeedForwardNN(inputs= training_df, hidden_layers= 2, nodes=best_params_two['nodes'], classification=DATASET_CLASS,
                                   learning_rate=best_params_two['learning_rate'], batch_size=best_params_two['batch_size'])
        no_hidden.train_data()
        no_weights = no_hidden.get_weights()
        one_hidden.train_data()
        one_weights = one_hidden.get_weights()
        two_hidden.train_data()
        two_weights = two_hidden.get_weights()
        
        np.random.shuffle(training_df)
        no_hidden.new_inputs(training_df)
        no_hidden.train_data()
        new_no_weights = no_hidden.get_weights()
        one_hidden.new_inputs(training_df)
        one_hidden.train_data()
        new_one_weights = one_hidden.get_weights()
        two_hidden.new_inputs(training_df)
        two_hidden.train_data()
        new_two_weights = two_hidden.get_weights()
        
        no_converge = 0
        while not np.all(np.abs(no_weights - new_no_weights) / no_weights <= 0.01):
            if no_converge == 1000:
                break
            no_converge+=1
            np.random.shuffle(training_df)
            no_hidden.new_inputs(training_df)
            no_hidden.train_data()
            no_weights = new_no_weights
            new_no_weights = no_hidden.get_weights()

        one_converge = 0
        while not np.all(np.abs(one_weights - new_one_weights) / one_weights <= 0.01):
            if one_converge == 1000:
                break
            one_converge+=1
            np.random.shuffle(training_df)
            one_hidden.new_inputs(training_df)
            one_hidden.train_data()
            one_weights = new_one_weights
            new_one_weights = one_hidden.get_weights()

        two_converge = 0
        while not np.all(np.abs(two_weights - new_two_weights) / two_weights <= 0.01):
            if two_converge == 1000:
                break
            two_converge+=1
            np.random.shuffle(training_df)
            two_hidden.new_inputs(training_df)
            two_hidden.train_data()
            two_weights = new_two_weights
            new_two_weights = two_hidden.get_weights()

        actual_zero, predicted_zero = no_hidden.test_data(i.to_numpy())
        actual_one, predicted_one = one_hidden.test_data(i.to_numpy())
        actual_two, predicted_two = two_hidden.test_data(i.to_numpy())

        no_loss = loss_functions(predicted_zero, actual_zero, 0.1)
        one_loss = loss_functions(predicted_one, actual_one, 0.1)
        two_loss = loss_functions(predicted_two, actual_two, 0.1)

        no_values.append(no_loss)
        one_values.append(one_loss)
        two_values.append(two_loss)

    print(no_values, one_values, two_values)
    plot_loss_functions(no_values, one_values, two_values)
    print(no_converge, one_converge, two_converge)
    print(best_params_no, best_params_one, best_params_two)

if __name__ == '__main__':  
    main()
