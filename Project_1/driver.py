import pandas as pd
from dataProcess import DataProcess
from modelTest import ModelTest

# CONSTANTS: 
# - DATASET is the path to the data set to be tested
# - DATASET_NAMES is a list of attribute names give in the .NAMES file
DATASET = 'Project_1/datasets/breast-cancer-wisconsin.data'
DATASET_NAMES = ['id','Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                 'Normal Nucleoli', 'Mitoses', 'class']
#DATASET = 'Project_1/datasets/glass.data'
#DATASET_NAMES = ['id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'class']

#--------------------------------------------------------------------------------------------------------------------------

def main():
    # Creates a data process instance with accurate information from the .NAMES file
    data = DataProcess(names=DATASET_NAMES, missing_val='?', id_col='id')
    # Loads the data set, splits into 10 folds
    data.loadCSV(DATASET)
    folds = data.k_fold_split(10)

    for i in folds:
        # Creates the training and test fold. Training fold is all folds exept the one on the index.
        # This allows for 10 experiements to be run on different data.
        test_df = i
        training_df = pd.concat([x for x in folds if not (x.equals(i))], axis=0, ignore_index=True)

        # Initilizes and trains the model
        test_clean = ModelTest(training_df, test_df)
        test_clean.load_data()
        test_clean.train_model()

        # Makes predictions, then tests our results
        predicted_classes = test_clean.classify_all()
        print("\nPerformance Metrics\n-------------------\n0/1-Loss = {}\nRecall = {}".format(test_clean.zero_one_loss(predicted_classes), test_clean.recall(predicted_classes)))
    
    # Repeats the process, this time with noise introduced
    data.introduce_noise(.1)
    noisy_folds = data.k_fold_split(10)
    for i in noisy_folds:
        test_df = i
        training_df = pd.concat([x for x in noisy_folds if not (x.equals(i))], axis=0, ignore_index=True)

        # Initilizes and trains the model
        test_clean = ModelTest(training_df, test_df)
        test_clean.load_data()
        test_clean.train_model()

        # Makes predictions, then tests our results
        predicted_classes = test_clean.classify_all()
        print("\nPerformance Metrics\n-------------------\n0/1-Loss = {}\nRecall = {}".format(test_clean.zero_one_loss(predicted_classes), test_clean.recall(predicted_classes)))

    # TODO: ADD A BETTER FORMAT TO DELIVER AND COMPARE RESULTS OF EACH MODEL
    
#--------------------------------------------------------------------------------------------------------------------------   
     
main()