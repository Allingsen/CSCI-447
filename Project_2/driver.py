import pandas as pd
import numpy as np
from dataProcess import DataProcess
from kNN import KNearestNeighbors

DATASET_CALLED = 'breast-cancer'
DATASET = 'Project_2/datasets/breast-cancer-wisconsin.data'
DATASET_NAMES = ['id', 'Clump Thickness ', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion',
                  'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'class'] 

def loss_functions(estimates:np.array, actual:np.array):
    #confusion_total represents the TPs, TNs, FPs, and FNs summed across classes
        confusion_total = np.zeros(4)
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
                elif(actual[i] == Class and estimates[i] != Class):
                    confusion_local[3] += 1

            for i in range(4):
                confusion_total[i] += confusion_local[i]

        #Recall = TPs/(TPs + FNs)
        recall = confusion_total[0]/(confusion_total[0] + confusion_total[3])
        #Precision = TPs/ (TPs + FPs)
        precision = confusion_total[0]/(confusion_total[0] + confusion_total[2])
        print((recall, precision))
        return(recall, precision)

def main():
    # Creates a data process instance with accurate information from the .NAMES file
    data = DataProcess(names=DATASET_NAMES, missing_val='?', id_col='id', regression=True)

    # Loads the data set, creates the tuning set, then splits into ten folds
    data.loadCSV(DATASET)
    tuning_set = data.create_tuning_set()
    tuning_set_classes = list(tuning_set['class'])
    folds = data.k_fold_split(10)

    # Iterates through, tests on the tuning set
    values = dict()
    k = 1
    for i in folds:
        # Creates the training and test fold. Training fold is all folds exept the one on the index.
        # This allows for 10 experiements to be run on different data.
        training_df = pd.concat([x for x in folds if not (x.equals(i))], axis=0, ignore_index=True)
    
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
        values[k] = loss_functions(np.array(preds, dtype=float), np.array(tuning_set_classes, dtype=float))

        k+=1

    # TODO: IMPLEMENT EDITED AND K-MEANS HERE

main()