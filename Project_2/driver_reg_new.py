import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataProcess import DataProcess
from kNN import KNearestNeighbors
from K_Means_Clustering_test import KMeansClustering

#DATASET_CALLED = 'ablone'
#DATASET = 'Project_2/datasets/abalone.data'
#DATASET_NAMES = ['Sex',
#'Length',
#'Diameter',
#'Height',
#'Whole',
#'Shucked',
#'Viscera',
#'Shell',
#'class']
DATASET_CALLED = 'forestfires'
DATASET = 'Project_2/datasets/forestfires.data'
DATASET_NAMES = ['X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind',
                 'rain', 'class']
#DATASET_CALLED = 'machine'
#DATASET = 'Project_2/datasets/machine.data'
#DATASET_NAMES = ['vend', 'model', 'MCYT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP', 'class']


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
    ax[0].set_title('K-Nearest Neighbor')
    
    ax[1].bar(x= range(10),
          height=enn_vals,
          color=cmap.colors, 
          width=0.5)
    
    ax[1].set_ylabel('0/1 Loss')
    ax[1].set_xlabel('Fold')
    ax[1].set_title('Edited K-Nearest Neighbor')

    ax[2].bar(x= range(10),
          height=kmeans_vals,
          color=cmap.colors, 
          width=0.5)

    ax[2].set_ylabel('0/1 Loss')
    ax[2].set_xlabel('Fold')
    ax[2].set_title('K-Means Clustered')

    labels = {}
    for i, col in enumerate(cmap.colors):
        labels['Fold ' + str(i+1)] = col
    handles = [plt.Rectangle((0,0),1,1, color=labels[label]) for label in labels.keys()]

    fig.legend(handles, labels.keys(), loc='center left')

    plt.savefig('Project_2/figures/' +DATASET_CALLED+'_fig.png')

def loss_functions(estimates:np.array, actual:np.array, epsilon:float):
    loss = 0
    for i in range(len(estimates)):
        if (actual[i] - (epsilon * actual[i])) <= estimates[i] <= (actual[i] + (epsilon * actual[i])):
            pass
        else:
            loss += 1
    print(loss)
    return loss

def edited_kNN(df: np.array, k:int, epsilon: float) -> np.array:
    index_to_remove = []
    for i in range(len(df)):
        classless_i = df[i][:-1]
        drop_df = np.delete(df, i, axis=0)
    
        knn = KNearestNeighbors(k, drop_df)
        pred = knn.calculate_distances(classless_i)
        if (df[i][-1] - (df[i][-1] * epsilon)) <= pred <= (df[i][-1] + (df[i][-1] * epsilon)):
            index_to_remove.append(False)
        else:
            index_to_remove.append(True)
    
    mod_df = np.array([0] * len(df[0]))
    
    for i in range(len(df)):
        if not index_to_remove[i]:
            mod_df = np.vstack([mod_df, df[i]])

    return mod_df[1:]

def main():
    # Creates a data process instance with accurate information from the .NAMES file
    data = DataProcess(names=DATASET_NAMES, regression=True)

    # Loads the data set, creates the tuning set, then splits into ten folds
    names = data.loadCSV(DATASET)
    tuning_set = data.create_tuning_set()

    tuning_set_classes = list(tuning_set['class'])
    folds = data.reg_k_fold_split(10)

    # Creates dict to save performance metrics
    values = []
    k = 1
    sigmas= [.000001, .00001, .0001, .01, .1, .5, 1, 10]
    # Iterates through, tests on the tuning set
    for i in folds:
        # Creates the training and test fold. Training fold is all folds exept the one on the index.
        # This allows for 10 experiements to be run on different data.
        training_df = (pd.concat([x for x in folds if not (x.equals(i))], axis=0, ignore_index=True)).to_numpy()
        local_loss = []

        # Calculates optimal sigma
        for j in sigmas:
            # Creates the model with k nearest neighbors to be tested
            knn = KNearestNeighbors(k, training_df, reg=True, sigma=j)

            # Removes the class from comparisons, makes predicitions
            tuning_set_no_class = tuning_set.drop('class', axis=1)
            test_points = tuning_set_no_class.to_numpy()
            preds = []
            for x in test_points:
                prediciton = knn.calculate_distances(x)
                preds.append(prediciton)

            # Performs our loss functions
            local_loss.append(loss_functions(np.array(preds), np.array(tuning_set_classes), 0.1))
        values.append(local_loss)
        k+=1
    
    
    print('--------------------------------------')
    lowest_loss = -1
    best_sigma = None
    best_k = None
    for i, val in enumerate(values):
        for j, num in enumerate(val):
            if num <= lowest_loss or lowest_loss < 0:
                lowest_loss = num
                best_sigma = (j+1)/10
                best_k = (i+1)

    # Finds the best k-value for running the experiment
    print(best_k)
    print(best_sigma)
    print('--------------------------------------')

    # Runs the k-NN experiment with correct K
    knn_values = list()
    for i in folds:
        # Creates the training and test fold. Training fold is all folds exept the one on the index.
        # This allows for 10 experiements to be run on different data.
        training_df = pd.concat([x for x in folds if not (x.equals(i))], axis=0, ignore_index=True).to_numpy()
    
        # Creates the model with k nearest neighbors to be tested
        knn = KNearestNeighbors(best_k, training_df, reg=True, sigma=best_sigma)
        
        # Removes the class from comparisons, makes predicitions
        test_set_no_class = i.drop('class', axis=1)
        test_points = test_set_no_class.to_numpy()
        preds = []
        for j in test_points:
            prediciton = knn.calculate_distances(j)
            preds.append(prediciton)

        # Performs our loss functions
        test_set_classes = list(i['class'])
        knn_values.append(loss_functions(np.array(preds), np.array(test_set_classes), .1))

    print('--------------------------------------')

    # Perform 10 fold cross-validation on the enn data
    clusters = []
    enn_values = list()
    for i in folds:
        # Creates the training and test fold. Training fold is all folds exept the one on the index.
        # This allows for 10 experiements to be run on different data.
        training_df = (pd.concat([x for x in folds if not (x.equals(i))], axis=0, ignore_index=True)).to_numpy()

        # Runs edited nearest neighbor
        edited_df = edited_kNN(training_df, 1, epsilon=0.1)
        clusters.append(len(edited_df))

        # Creates the model with k nearest neighbors to be tested
        knn = KNearestNeighbors(best_k, edited_df, reg=True, sigma=best_sigma)

        # Removes the class from comparisons, makes predicitions
        test_set_no_class = i.drop('class', axis=1)
        test_points = test_set_no_class.to_numpy()

        preds = []
        for j in test_points:
            prediciton = knn.calculate_distances(j)
            preds.append(prediciton)

        # Performs our loss functions
        test_set_classes = list(i['class'])
        enn_values.append(loss_functions(np.array(preds), np.array(test_set_classes), .1))

    print('--------------------------------------')

    # Runs K-means clustering on the original dataset
    no_of_centroids = round(np.mean(clusters))

    # Performs 10 fold CV on the centroids
    kmeans_values = list()
    for i in folds:
        # Creates the training and test fold. Training fold is all folds exept the one on the index.
        # This allows for 10 experiements to be run on different data.
        training_df = pd.concat([x for x in folds if not (x.equals(i))], axis=0, ignore_index=True)
        
        # Clusters the training set
        clustering = KMeansClustering(training_df, no_of_centroids, reg=True)
        new_points = clustering.calculate_centroids()
       
        # Creates a Dataframe of the clusters
        mod_df = np.array([0] * len(new_points[0]))
        for k in new_points:
            mod_df = np.vstack([mod_df, k])
        # Creates the model with k nearest neighbors to be tested
        knn = KNearestNeighbors(best_k, mod_df[1:], reg=True, sigma=best_sigma)

        # Removes the class from comparisons, makes predicitions
        test_set_no_class = i.drop('class', axis=1)
        test_points = test_set_no_class.to_numpy()

        preds = []
        for j in test_points:
            prediciton = knn.calculate_distances(j.astype(float))
            preds.append(prediciton)
        
        # Performs our loss functions
        test_set_classes = list(i['class'])
        kmeans_values.append(loss_functions(np.array(preds, dtype=float), np.array(test_set_classes, dtype=float), .1))

    print('--------------------------------------')

    # Plots the functions
    plot_loss_functions(knn_values, enn_values, kmeans_values)

if __name__ == '__main__':
    main()