#ModelTest inherits from NaiveBayesModel
import pandas as pd
import numpy as np
from NaiveBayesModel import NaiveBayesModel

#--------------------------------------------------------------------------------------------------------------------------

class ModelTest(NaiveBayesModel):
    def __init__(self, training_data, test_data):
        super().__init__(training_data)
        self.test_data = test_data

#--------------------------------------------------------------------------------------------------------------------------

    #Classifies one feature vector (helper method for classify_all())
    #feature_vector is a dictionary mapping feature names to feature values
    def classify_one(self, feature_vector):
        results_per_class = {}
        for Class in self.classes:
            repeated_product = 1
            for feature in feature_vector:
                cond_prob_tuple = (feature, feature_vector[feature], Class)
                if(cond_prob_tuple in self.conditional_prob_of_features):
                    repeated_product *= self.conditional_prob_of_features[cond_prob_tuple]
                else:
                    repeated_product *= 1/(self.number_of_examples_in_class[Class] + self.number_of_features)
            result = self.prior_prob_of_classes[Class] * repeated_product
            results_per_class[Class] = result
        
        return(max(results_per_class, key = results_per_class.get))
    
#--------------------------------------------------------------------------------------------------------------------------

    #Classifies all feature vectors in test_data and returns a list of the classifications
    def classify_all(self):
        print("\nPredicted Classes for Test Set:\n-------------------------------")
        predicted_classes = []
        for i in range(len(self.test_data)):
            print(i)    # FOR ALEX -- PROGRESS TRACKER
            feature_vector = {}
            for column in self.test_data.columns:
                if(column != "id" and column != "class"):
                    feature_vector[column] = self.test_data.loc[i, column]
            classification = self.classify_one(feature_vector)
            print("Vector {}: Class {}".format(list(feature_vector.values()), classification))
            predicted_classes.append(classification)
        return(predicted_classes)

#--------------------------------------------------------------------------------------------------------------------------

    #Returns the 0/1-Loss of the learner over the test set (Works on multi-class problems)
    def zero_one_loss(self, predicted_classes):
        loss = 0
        for i in range(len(self.test_data)):
            if(self.test_data.loc[i, "class"] != predicted_classes[i]):
                loss += 1
        return(loss)
    
#--------------------------------------------------------------------------------------------------------------------------

    #Returns the recall (Works on multi-class problems)
    def recall(self, predicted_classes):

        #confusion_total represents the TPs, TNs, FPs, and FNs summed across classes
        confusion_total = np.zeros(4)
        for Class in self.classes:

            #confusion_local represents the TPs, TNs, FPs, and FNs for a specific class
            confusion_local = np.zeros(4)
            for i in range(len(self.test_data)):
                if(self.test_data.loc[i, "class"] == Class and predicted_classes[i] == Class):
                    confusion_local[0] += 1
                elif(self.test_data.loc[i, "class"] != Class and predicted_classes[i] != Class):
                    confusion_local[1] += 1
                elif(self.test_data.loc[i, "class"] != Class and predicted_classes[i] == Class):
                    confusion_local[2] += 1
                elif(self.test_data.loc[i, "class"] == Class and predicted_classes[i] != Class):
                    confusion_local[3] += 1

            for i in range(4):
                confusion_total[i] += confusion_local[i]

        #Recall = TPs/(TPs + FNs)
        recall = confusion_total[0]/(confusion_total[0] + confusion_total[3])
        return(recall)
             
#--------------------------------------------------------------------------------------------------------------------------