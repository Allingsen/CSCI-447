#DATA ATTRIBUTES
#----------------------------------------------------------------------------------------------------------
#df_train is a pandas dataframe holding the information from the training dataset

#df_test is a pandas dataframe holding the information from the test dataframe (could be a hyperparamter tuning set)

#distance_matrix is a 2-D numpy array that holds the distances between each pair of training examples 

#categorical is a list that holds the names of the categorical columns in the dataset

#numerical is a list that holds the names of the numerical columns in the dataset

#categorical_indices is a list that holds the indices of the categorical columns in self.df_train

#numerical_indices is a list that holds the indices of the numerical columns in self.df_train

#k is an integer representing the number of clusters

#centroids is a list holding the current centroids of the dataset. This is updated by cluster_data()

#clusters is a list that will hold dictionaries mapping cluster centroids to the examples contained in the cluster

#features is a list that will hold the feature vectors in the training set

#features_to_IDs is a dictionary that maps feature vectors to their row index in self.df_train
#----------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import random
import math

class KMeansClustering:
    def __init__(self, df_train, df_test, distance_matrix, categorical_columns, numerical_columns, k):
        self.df_train = df_train
        self.df_test = df_test
        self.distance_matrix = distance_matrix
        self.categorical = categorical_columns
        self.numerical = numerical_columns
        self.categorical_indices = []
        self.numerical_indices = []
        self.get_categorical_indices()
        self.get_numerical_indices()
        self.k = k
        self.centroids = []
        self.clusters = []
        self.features = self.fill_features()
        self.features_to_IDs = self.fill_features_to_IDs()

#----------------------------------------------------------------------------------------------------------

    #Creates a list of indices that correspond to numerical columns in the training set
    def get_numerical_indices(self):
        index = 0
        for column in self.df_train.columns:
            if(column in self.numerical):
                self.numerical_indices.append(index)
            index += 1 

#----------------------------------------------------------------------------------------------------------

    #Creates a list of indices that correspond to categorical columns in the training set
    def get_categorical_indices(self):
        index = 0
        for column in self.df_train.columns:
            if(column in self.categorical):
                self.categorical_indices.append(index)
            index += 1

#----------------------------------------------------------------------------------------------------------

    #Creates a list of feature vectors in the training set which are represented as dictionaries
    def fill_features(self):
        features = []
        for i in range(len(self.df_train)):
            feature_vector = []
            for column in self.df_train.columns:
                if(column != "class" and column != "id" and column != "value"):
                    feature_vector.append(self.df_train.loc[i, column])
            features.append(feature_vector)
        return(features)

#----------------------------------------------------------------------------------------------------------

    #Returns a dictionary mapping feature vectors to their row in self.df_train
    def fill_features_to_IDs(self):
        features_to_IDs = {}
        for i in range(len(self.features)):
            features_to_IDs[tuple(self.features[i])] = i
        return(features_to_IDs)
    
#----------------------------------------------------------------------------------------------------------

    #Getter method for the feature vectors in the training set
    def get_features(self):
        return(self.features)
    
#----------------------------------------------------------------------------------------------------------

    #Applies the clustering process until convergence has occurred
    def cluster_data(self):
        converged = False
        first_time = True
        old_centroids = []

        #Selects k random training examples to be the initial centroids
        for i in range(self.k):
            centroid = random.randint(0, len(self.features) - 1)
            while(self.features[centroid] in self.centroids):
                centroid = random.randint(0, len(self.features) - 1)
            self.centroids.append(self.features[centroid])
        print("Initial Centroids:", self.centroids)

        while(converged == False):
            #Fill keys of the dictionaries in self.clusters with the current centroids
            for i in range(len(self.centroids)):
                cluster_dict = {}
                cluster_dict[tuple(self.centroids[i])] = []
                self.clusters.append(cluster_dict)

            #If it's the first time through the loop, access distance_matrix for distances and assign vectors to
            #appropriate clusters
            if(first_time):            
                for i in range(len(self.features)):
                    centroid_distances = []
                    for centroid in self.centroids:
                        centroid_distances.append(self.distance_matrix[i][self.features.index(centroid)])
                    index = centroid_distances.index(min(centroid_distances))
                    self.clusters[index][tuple(self.centroids[index])].append(i)
                print("Clusters:", self.clusters)
                first_time = False
                
            #If it's not the first time, use calculate_distance() to get distances and assign vectors to
            #appropriate clusters
            else:
                for i in range(len(self.features)):
                    centroid_distances = []
                    for centroid in self.centroids:
                        centroid_distances.append(self.calculate_distance(self.features[i], centroid))
                    index = centroid_distances.index(min(centroid_distances))
                    self.clusters[index][tuple(self.centroids[index])].append(i)
                print("Clusters:", self.clusters)   

            #Save old centroids before updating 
            old_centroids = self.centroids.copy()

            #Update centroids 
            self.centroids.clear()

            #For each cluster, calculate the new centroid by taking the mean of all of the numerical features and
            #the mode of all of the categorical features
            for cluster in self.clusters:
                total_vector_mean = []
                mode_vector = []

                #Mean for numerical features
                for i in range(len(self.features[0])):
                    total_vector_mean.append(0)

                index = 0
                for column in self.numerical:
                    for i in range(len(list(cluster.values())[0])):
                        total_vector_mean[self.numerical_indices[index]] += self.df_train.loc[list(cluster.values())[0][i], column]
                    index += 1
                
                centroid_vector = total_vector_mean
                for i in range(len(total_vector_mean)):
                    centroid_vector[i] = total_vector_mean[i]/len(list(cluster.values())[0])

                #Mode for categorical features
                index = 0
                for column in self.categorical:
                    possible_values = {}
                    for i in range(len(list(cluster.values())[0])):
                        value = self.df_train.loc[list(cluster.values())[0][i], column]
                        if(value not in possible_values):
                            possible_values[value] = 0
                        possible_values[value] += 1
                    mode = max(possible_values, key = lambda x: possible_values[x])
                    mode_vector.append(mode)
                
                for i in range(len(mode_vector)):
                    centroid_vector[self.categorical_indices[i]] = mode_vector[i]

                self.centroids.append(centroid_vector)  
            print("New Centroids: {}".format(self.centroids)) 

            if(old_centroids == self.centroids):
                converged = True 
                print("centroids have converged, clustering complete")

            else:
                print("Centroids have not converged... recalculating centroids")
                self.clusters.clear()
        
        return(self.clusters)

#----------------------------------------------------------------------------------------------------------

    #Calculates the Euclidean distance between two feature vectors. This method is only used for numerical columns
    def euclidean_distance(self, vector1, vector2):
        sum_of_squared_differences = 0
        index = 0
        for column in self.numerical:
            val1 = vector1[self.numerical_indices[index]]
            val2 = vector2[self.numerical_indices[index]]
            squared_difference = pow((val2 - val1), 2)
            sum_of_squared_differences += squared_difference
            index += 1

        euclidean_distance = math.sqrt(sum_of_squared_differences)
        return(euclidean_distance)

#----------------------------------------------------------------------------------------------------------

    #Calculates the value difference metric between two feature vectors if the task is classification. Calculates 
    #the hamming distance if the task is regression. This method is only used for categorical columns
    def categorical_distance(self, vector1, vector2):
    
        #If the task is regression, use hamming distance
        if("value" in self.df_train.columns):
            hamming_distance = 0
            index = 0
            for column in self.categorical:
                val1 = vector1[self.categorical_indices[index]]
                val2 = vector2[self.categorical_indices[index]]
                if(val1 != val2):
                    hamming_distance += 1
            return(hamming_distance)
       
        #If the task is classification, calculate the VDM
        else:

            #Obtain list of classes
            classes = []
            for Class in self.df_train["class"]:
                if(Class not in classes):
                    classes.append(Class)

            #Intitialize variables to store steps of calculation
            value_difference_sum = 0
            value_difference_metric = 0
            index = 0

            #For each categorical column, find the value difference between vector1 and vector2
            for column in self.categorical:
                
                #Initialize variables to store steps of calculation
                val1 = vector1[self.categorical_indices[index]]
                val2 = vector2[self.categorical_indices[index]]
                C_i1 = 0
                C_i2 = 0
                C_i_a1 = 0
                C_i_a2 = 0
                sum_over_classes = 0

                #For each class, find (abs((C_i,a/C_i) - (C_j,a/C_j)))^2
                for Class in classes:  
                    calculation = 0    
                    for i in range(len(self.df_train)):
                        if(self.df_train.loc[i, column] == val1): 
                            C_i1 += 1
                        if(self.df_train.loc[i, column] == val1 and self.df_train.loc[i, "class"] == Class):
                            C_i_a1 += 1
                    for i in range(len(self.df_train)):
                        if(self.df_train.loc[i, column] == val2):
                            C_i2 += 1
                        if(self.df_train.loc[i, column] == val2 and self.df_train.loc[i, "class"] == Class):
                            C_i_a2 += 1

                    calculation = pow((abs((C_i_a1/C_i1) - (C_i_a2/C_i2))), 2)

                    #Accumulate results to obtain delta(v_i, v_j)
                    sum_over_classes += calculation
                    C_i1 = 0
                    C_i2 = 0
                    C_i_a1 = 0
                    C_i_a2 = 0

                #Accumulate the delta(v_i, v_j)s over each categorical feature
                value_difference_sum += sum_over_classes

                index += 1

            #Take sqrt(value_difference_sum) to obtain the value distance metric
            value_difference_metric = math.sqrt(value_difference_sum)
            return(value_difference_metric)
            
#----------------------------------------------------------------------------------------------------------

    #Calculates the total distance between vector1 and vector2 using euclidean_distance() and categorical_distance()
    def calculate_distance(self, vector1, vector2):
        total_distance = ((len(self.numerical)/len(self.features))*self.euclidean_distance(vector1, vector2)) + ((len(self.categorical)/len(self.features))*self.categorical_distance(vector1, vector2))
        return(total_distance)
    
#----------------------------------------------------------------------------------------------------------

#This is a trial dataset for testing this class
data = {
    #numerical columns
    'Temperature': [78.5, 97.1, 99.002, 88.5, 66.907],
    'Precipitation': [10.08, 7.99, 78.64, 33.33, 23.89],
    'wind speed': [78, 0.9, 65, 25, 7.9],
    #categorical columns
    'color': ["red", "blue", "blue", "red", "red"],
    'weather': ["sunny", "overcast", "sunny", "sunny", "overcast"],
    'class': [1, 2, 2, 1, 2]
}

distance_matrix = np.zeros((5,5))
skip = []


df_train = pd.DataFrame(data)
df_test = pd.DataFrame()
numerical = ["Temperature", "Precipitation", "wind speed"]
categorical = ["color", "weather"]
k = 2
clustering = KMeansClustering(df_train, df_test, distance_matrix, categorical, numerical, k)

for i in range(5):
    for j in range(5):
        if(i == j):
            distance_matrix[i][j] = 0

        elif((j,i) in skip):
            pass

        else:
            distance = clustering.calculate_distance(clustering.get_features()[i], clustering.get_features()[j])
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance

print(distance_matrix)
clustering.cluster_data()