import pandas as pd
from dataProcess import DataProcess
from modelTest import ModelTest
import numpy as np
import matplotlib.pyplot as plt

# CONSTANTS: 
# - DATASET_CALLED is a shortened version of the dataset name (For saving figures)
# - DATASET is the path to the data set to be tested
# - DATASET_NAMES is a list of attribute names give in the .NAMES file
DATASET_CALLED = 'breast-cancer'
DATASET = 'Project_1/datasets/breast-cancer-wisconsin.data'
DATASET_NAMES = ['id', 'Clump Thickness ', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion',
                  'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'class'] 

#--------------------------------------------------------------------------------------------------------------------------

def make_plots(clean_recall, noise_recall, clean_loss, noise_loss):
    '''Creates a figure with two subplots showing our results'''
    # Sets up plot for displaying 
    fig, ax = plt.subplots(1, 2, figsize=(7, 4))
    fig.tight_layout(pad=5.0)
    cmap = plt.get_cmap('tab10')

    # Creates the space where the bars will be placed
    clean_group = np.linspace(0.7, 1.3, 10)
    noise_group = np.linspace(1.7, 2.3, 10) 

    # Sets up axes and title for recall graph
    ax[0].set_xticks([1,2])
    ax[0].set_xticklabels(['Clean Data','Noisy Data'])
    ax[0].set_ylabel('Recall Percentage')
    ax[0].set_xlabel('Fold')
    ticks = ax[0].get_yticks()
    ax[0].set_yticklabels(['{:.2%}'.format(x) for x in ticks])
    ax[0].set_title('Recall')

    # Plots the folds on the clean recall graph
    ax[0].bar(x=clean_group,
              height=clean_recall,
              color=cmap.colors, 
              width=0.05)
              
    # Plots the folds on the noisy recall graph
    ax[0].bar(x=noise_group, 
              height=noise_recall, 
              color=cmap.colors, 
              width=0.05)
    
    # Sets up axes and title for loss graph
    ax[1].set_xticks([1,2])
    ax[1].set_xticklabels(['Clean Data','Noisy Data'])
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Fold')
    ax[1].set_title('0/1 Loss')

    # Plots the folds on the clean loss graph
    ax[1].bar(x=clean_group,
              height=clean_loss,
              color=cmap.colors, 
              width=0.05)

    # Plots the folds on the noisy loss graph   
    ax[1].bar(x=noise_group, 
              height=noise_loss, 
              color=cmap.colors, 
              width=0.05)
    
    plt.savefig('Project_1/figures/' +DATASET_CALLED+'_fig.png')
    plt.show()

#--------------------------------------------------------------------------------------------------------------------------

def main():
    # Creates a data process instance with accurate information from the .NAMES file
    data = DataProcess(names=DATASET_NAMES, missing_val='?', id_col='Id')
    # Loads the data set, splits into 10 folds
    data.loadCSV(DATASET)
    folds = data.k_fold_split(10)
    
    clean_loss = list()
    clean_recall = list()
    for i in folds:
        # Creates the training and test fold. Training fold is all folds exept the one on the index.
        # This allows for 10 experiements to be run on different data.
        test_df = i
        training_df = pd.concat([x for x in folds if not (x.equals(i))], axis=0, ignore_index=True)

        # Initilizes and trains the model
        test_clean = ModelTest(training_df, test_df)
        test_clean.load_data()
        test_clean.train_model()

        print(test_clean.prior_prob_of_classes)
        print(test_clean.class_probability)
        print(test_clean.number_of_examples_in_class)

        # Makes predictions, then tests our results
        predicted_classes = test_clean.classify_all()
        print("\nPerformance Metrics\n-------------------\n0/1-Loss = {}\nRecall = {}".format(test_clean.zero_one_loss(predicted_classes), test_clean.recall(predicted_classes)))
        clean_loss.append(test_clean.zero_one_loss(predicted_classes))
        clean_recall.append(test_clean.recall(predicted_classes))

    # Repeats the process, this time with noise introduced
    data.introduce_noise(.1)
    noisy_folds = data.k_fold_split(10)

    noise_loss = list()
    noise_recall = list()
    for i in noisy_folds:
        test_df = i
        training_df = pd.concat([x for x in noisy_folds if not (x.equals(i))], axis=0, ignore_index=True)

        # Initilizes and trains the model
        test_noise = ModelTest(training_df, test_df)
        test_noise.load_data()
        test_noise.train_model()

        # Makes predictions, then tests our results
        predicted_classes = test_noise.classify_all()
        print("\nPerformance Metrics\n-------------------\n0/1-Loss = {}\nRecall = {}".format(test_noise.zero_one_loss(predicted_classes), test_noise.recall(predicted_classes)))
        noise_loss.append(test_noise.zero_one_loss(predicted_classes))
        noise_recall.append(test_noise.recall(predicted_classes))
    
    make_plots(clean_recall, noise_recall, clean_loss, noise_loss)

    print('Clean loss:', clean_loss, 'Clean Recall:', clean_recall)
    print('Noise loss:', noise_loss, 'Noise Recall:', noise_recall)
    
#--------------------------------------------------------------------------------------------------------------------------   
     
if __name__ == '__main__':
    main()