import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from dataProcess import DataProcess
from feedForwardNN import FeedForwardNN
 
DATASET_CALLED = 'soybean-two'
DATASET = 'Project_2/datasets/soybean-small.data'
DATASET_NAMES = [*range(35)] + ['class']
DATASET_CLASS = False

def plot_loss_functions(zero_layer: list, one_layer: list, two_layer:list) -> None:
    '''Creates a figure with two subplots showing our results'''
    # Sets up plot for displaying 
    fig, ax = plt.subplots(1, 3, figsize=(10,4))
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

    plt.savefig('Project_2/figures/' +DATASET_CALLED+'_fig.png')

def loss_functions(estimates:np.array, actual:np.array):
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
    data = DataProcess(names=DATASET_NAMES,cat_class=True)

    # Loads the data set, creates the tuning set, then splits into ten folds
    data.loadCSV(DATASET)
    tuning_set = data.create_tuning_set()
    folds = data.k_fold_split(10)

    # Iterates through, tests on the tuning set
    learning_rates = []
    batch_sizes = []
    no_nodes = []

    best_params_no = {}
    best_score_no = 0
    best_params_one = {}
    best_score_one = 0
    best_params_two = {}
    best_score_two = 0

    for i in folds:
        # Creates the training and test fold. Training fold is all folds exept the one on the index.
        # This allows for 10 experiements to be run on different data.
        learning_rate = random.choice(learning_rates)
        batch_size = random.choice(batch_sizes)
        num_nodes = [random.choice(no_nodes), random.choice(no_nodes)]
        training_df = (pd.concat([x for x in folds if not (x.equals(i))], axis=0, ignore_index=True)).to_numpy()

        no_hidden = FeedForwardNN(inputs= training_df, hidden_layers= 0, nodes=[], classification=DATASET_CLASS,
                                   learning_rate=learning_rate, batch_size=batch_size,
                                   class_names=DATASET_NAMES)
        one_hidden = FeedForwardNN(inputs= training_df, hidden_layers= 1, nodes=[num_nodes[0]], classification=DATASET_CLASS,
                                   learning_rate=learning_rate, batch_size=batch_size,
                                   class_names=DATASET_NAMES)
        two_hidden = FeedForwardNN(inputs= training_df, hidden_layers= 2, nodes=[num_nodes[0], num_nodes[1]], classification=DATASET_CLASS,
                                   learning_rate=learning_rate, batch_size=batch_size,
                                   class_names=DATASET_NAMES)
        no_hidden.train_data()
        one_hidden.train_data()
        two_hidden.train_data()

        actual_zero, predicted_zero = no_hidden.test_data(tuning_set)
        actual_one, predicted_one = one_hidden.test_data(tuning_set)
        actual_two, predicted_two = two_hidden.test_data(tuning_set)

        no_loss = loss_functions(predicted_zero, actual_zero)
        one_loss = loss_functions(predicted_one, actual_one)
        two_loss = loss_functions(predicted_two, actual_two)

        score_no = np.mean(no_loss)
        if score_no > best_score_no:
            best_params_no['learning_rate'] = learning_rate
            best_params_no['batch_size'] = batch_size

        score_one = np.mean(one_loss)
        if score_one > best_score_one:
            best_params_one['learning_rate'] = learning_rate
            best_params_one['batch_size'] = batch_size
            best_params_one['nodes'] = [num_nodes[0]]

        score_two = np.mean(two_loss)
        if score_two > best_score_two:
            best_params_two['learning_rate'] = learning_rate
            best_params_two['batch_size'] = batch_size
            best_params_one['nodes'] = num_nodes

    no_values = []
    one_values = []
    two_values = []
    for i in folds:
        # Creates the training and test fold. Training fold is all folds exept the one on the index.
        # This allows for 10 experiements to be run on different data.
        training_df = (pd.concat([x for x in folds if not (x.equals(i))], axis=0, ignore_index=True)).to_numpy()

        no_hidden = FeedForwardNN(inputs= training_df, hidden_layers= 0, nodes=[], classification=DATASET_CLASS,
                                   learning_rate=best_params_no['learning_rate'], batch_size=best_params_no['batch_size'],
                                   class_names=DATASET_NAMES)
        one_hidden = FeedForwardNN(inputs= training_df, hidden_layers= 1, nodes=best_params_one['nodes'], classification=DATASET_CLASS,
                                   learning_rate=best_params_one['learning_rate'], batch_size=best_params_one['batch_size'],
                                   class_names=DATASET_NAMES)
        two_hidden = FeedForwardNN(inputs= training_df, hidden_layers= 2, nodes=best_params_two['nodes'], classification=DATASET_CLASS,
                                   learning_rate=best_params_two['learning_rate'], batch_size=best_params_two['batch_size'],
                                   class_names=DATASET_NAMES)
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

        while not np.all(np.abs(no_weights - new_no_weights) / no_weights <= 0.05):
            np.random.shuffle(training_df)
            no_hidden.new_inputs(training_df)
            no_hidden.train_data()
            new_no_weights = no_hidden.get_weights()

        while not np.all(np.abs(one_weights - new_one_weights) / one_weights <= 0.05):
            np.random.shuffle(training_df)
            one_hidden.new_inputs(training_df)
            one_hidden.train_data()
            new_one_weights = one_hidden.get_weights()

        while not np.all(np.abs(two_weights - new_two_weights) / two_weights <= 0.05):
            np.random.shuffle(training_df)
            two_hidden.new_inputs(training_df)
            two_hidden.train_data()
            new_two_weights = two_hidden.get_weights()

        actual_zero, predicted_zero = no_hidden.test_data(i)
        actual_one, predicted_one = one_hidden.test_data(i)
        actual_two, predicted_two = two_hidden.test_data(i)

        no_loss = loss_functions(predicted_zero, actual_zero)
        one_loss = loss_functions(predicted_one, actual_one)
        two_loss = loss_functions(predicted_two, actual_two)

        no_values.append(no_loss)
        one_values.append(one_loss)
        two_values.append(two_loss)

    plot_loss_functions(no_values, one_values, two_values)

if __name__ == '__main__':  
    main()
