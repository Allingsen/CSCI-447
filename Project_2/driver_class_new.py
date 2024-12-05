import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataProcess import DataProcess
from kNN import KNearestNeighbors
from K_Means_Clustering_test import KMeansClustering

#DATASET_CALLED = 'glass'
#DATASET = 'Project_2/datasets/glass.data'
#DATASET_NAMES = ['id', 'Ri', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'class']
#DATASET_CALLED = 'breast-cancer-two'
#DATASET = 'Project_2/datasets/breast-cancer-wisconsin.data'
#DATASET_NAMES = ['id', 'Clump Thickness ', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion',
#                  'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'class'] 
DATASET_CALLED = 'soybean-two'
DATASET = 'Project_2/datasets/soybean-small.data'
DATASET_NAMES = [*range(35)] + ['class']

def plot_loss_functions(knn_vals: list, enn_vals: list, kmeans_vals:list) -> None:
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
          height=[x[0] for x in knn_vals],
          color=cmap.colors, 
          width=0.05)
    
    ax[0].bar(x= precision, 
              height=[x[1] for x in knn_vals], 
              color=cmap.colors, 
              width=0.05)
    
    ax[0].set_xticks([1,2])
    ax[0].set_xticklabels(['Recall','Precision'])
    ax[0].set_ylabel('Percentage')
    ax[0].set_xlabel('Fold')
    ax[0].set_yticks([x/100 for x in ticks])
    ax[0].set_yticklabels(ticks)
    ax[0].set_title('K-Nearest Neighbor')
    
    ax[1].bar(x= recall,
      height=[x[0] for x in enn_vals],
      color=cmap.colors, 
      width=0.05)

    ax[1].bar(x= precision, 
              height=[x[1] for x in enn_vals], 
              color=cmap.colors, 
              width=0.05)
    
    ax[1].set_xticks([1,2])
    ax[1].set_xticklabels(['Recall','Precision'])
    ax[1].set_ylabel('Percentage')
    ax[1].set_xlabel('Fold')
    ax[1].set_yticks([x/100 for x in ticks])
    ax[1].set_yticklabels(ticks)
    ax[1].set_title('Edited K-Nearest Neighbor')

    ax[2].bar(x= recall,
          height=[x[0] for x in kmeans_vals],
          color=cmap.colors, 
          width=0.05)
    
    ax[2].bar(x= precision, 
              height=[x[1] for x in kmeans_vals], 
              color=cmap.colors, 
              width=0.05)
    
    ax[2].set_xticks([1,2])
    ax[2].set_xticklabels(['Recall','Precision'])
    ax[2].set_ylabel('Percentage')
    ax[2].set_xlabel('Fold')
    ax[2].set_yticks([x/100 for x in ticks])
    ax[2].set_yticklabels(ticks)
    ax[2].set_title('K-Means Clustered')

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

def edited_kNN(df: np.array, k:int) -> np.array:
    index_to_remove = []
    for i in range(len(df)):
        classless_i = df[i][:-1]
        drop_df = np.delete(df, i, axis=0)
    
        knn = KNearestNeighbors(k, drop_df)
        pred = knn.calculate_distances(classless_i)
        if pred == df[i][-1]:
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
    data = DataProcess(names=DATASET_NAMES,cat_class=True)

    # Loads the data set, creates the tuning set, then splits into ten folds
    data.loadCSV(DATASET)
    tuning_set = data.create_tuning_set()
    tuning_set_classes = list(tuning_set['class'])
    folds = data.k_fold_split(10)

    # Creates dict to save performance metrics
    values = dict()
    k = 1

    # Iterates through, tests on the tuning set
    for i in folds:
        # Creates the training and test fold. Training fold is all folds exept the one on the index.
        # This allows for 10 experiements to be run on different data.
        training_df = (pd.concat([x for x in folds if not (x.equals(i))], axis=0, ignore_index=True)).to_numpy()

        # Creates the model with k nearest neighbors to be tested
        knn = KNearestNeighbors(k, training_df)

        # Removes the class from comparisons, makes predicitions
        tuning_set_no_class = tuning_set.drop('class', axis=1)
        test_points = tuning_set_no_class.to_numpy()
        preds = []
        for j in test_points:
            prediciton = knn.calculate_distances(j)
            preds.append(prediciton)

        # Performs our loss functions
        values[k] = loss_functions(np.array(preds), np.array(tuning_set_classes))

        k+=1

    print('--------------------------------------')
    
    # Finds the best k-value for running the experiment
    best_k = -1
    best_score = 0
    for key, val in values.items():
        score = np.mean(val)
        if score > best_score:
            best_score = score
            best_k = key
            
    print(best_k)
    print('--------------------------------------')

    # Runs the k-NN experiment with correct K
    knn_values = list()
    for i in folds:
        # Creates the training and test fold. Training fold is all folds exept the one on the index.
        # This allows for 10 experiements to be run on different data.
        training_df = pd.concat([x for x in folds if not (x.equals(i))], axis=0, ignore_index=True).to_numpy()
    
        # Creates the model with k nearest neighbors to be tested
        knn = KNearestNeighbors(best_k, training_df)

        # Removes the class from comparisons, makes predicitions
        test_set_no_class = i.drop('class', axis=1)
        test_points = test_set_no_class.to_numpy()
        preds = []
        for j in test_points:
            prediciton = knn.calculate_distances(j)
            preds.append(prediciton)

        # Performs our loss functions
        test_set_classes = list(i['class'])
        knn_values.append(loss_functions(np.array(preds), np.array(test_set_classes)))

    print('--------------------------------------')

    # Perform 10 fold cross-validation on the enn data
    clusters = []
    enn_values = list()
    for i in folds:
        # Creates the training and test fold. Training fold is all folds exept the one on the index.
        # This allows for 10 experiements to be run on different data.
        training_df = (pd.concat([x for x in folds if not (x.equals(i))], axis=0, ignore_index=True)).to_numpy()

        # Runs edited nearest neighbor
        edited_df = edited_kNN(training_df, 1)
        clusters.append(len(edited_df))

        # Creates the model with k nearest neighbors to be tested
        knn = KNearestNeighbors(best_k, edited_df)

        # Removes the class from comparisons, makes predicitions
        test_set_no_class = i.drop('class', axis=1)
        test_points = test_set_no_class.to_numpy()
        preds = []
        for j in test_points:
            prediciton = knn.calculate_distances(j)
            preds.append(prediciton)

        # Performs our loss functions
        test_set_classes = list(i['class'])
        enn_values.append(loss_functions(np.array(preds), np.array(test_set_classes)))

    print('--------------------------------------')

    no_of_centroids = round(np.mean(clusters))
    # Performs 10 fold CV on the centroids
    kmeans_values = list()

    for i in folds:
        # Creates the training and test fold. Training fold is all folds exept the one on the index.
        # This allows for 10 experiements to be run on different data.
        training_df = pd.concat([x for x in folds if not (x.equals(i))], axis=0, ignore_index=True)

        # Clusters the training set
        clustering = KMeansClustering(training_df, no_of_centroids)
        new_points = clustering.calculate_centroids()

        # Creates a Dataframe of the clusters
        mod_df = np.array([0] * len(new_points[0]))
        for k in new_points:
            mod_df = np.vstack([mod_df, k])

        # Creates the model with k nearest neighbors to be tested
        knn = KNearestNeighbors(best_k, mod_df[1:])

        # Removes the class from comparisons, makes predicitions
        test_set_no_class = i.drop('class', axis=1)
        test_points = test_set_no_class.to_numpy()
        preds = []
        for j in test_points:
            prediciton = knn.calculate_distances(j)
            preds.append(prediciton)

        # Performs our loss functions
        test_set_classes = list(i['class'])
        kmeans_values.append(loss_functions(np.array(preds), np.array(test_set_classes)))

    # Plots the functions
    plot_loss_functions(knn_values, enn_values, kmeans_values)

if __name__ == '__main__':
    main()