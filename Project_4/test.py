import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataProcess import DataProcess
from geneticAlgorithm import geneticAlg
from differentialEvolution import differentialEvolution
from ParticleSwarm import ParticleSwarm
import feedForwardNN_GA
import feedForwardNN
import time

# Fores

# Abalone Dataset
DATASET_CALLED = 'abalone'
DATASET = 'Project_4/datasets/abalone.data'
DATASET_NAMES = ['Sex', 'length', 'diameter', 'height', 'whole', 'shucked', 'viscera', 'shell', 'class']

# Creates a data process instance with accurate information from the .NAMES file
data = DataProcess(names=DATASET_NAMES,cat_class=False,regression=True) # <- TODO: Abalone and Forest Fire Dataset
#data = DataProcess(names=DATASET_NAMES,cat_class=False,regression=True, id_col=['id', 'model'])
# Loads the data set, creates the tuning set, then splits into ten folds
data.loadCSV(DATASET)

df = data.get_df()
print(np.mean(df['class']))

def loss_functions(estimates:np.array, actual:np.array):
        '''Calculates preiciosn and recall'''
        sum = (actual - estimates)**2
        mse = np.mean(sum)
        print(mse)
        return mse


loss_functions(estimates=np.array([np.mean(df['class']) for _ in range(len(df['class']))]), actual=df['class'])