#DATA ATTRIBUTES
#--------------------------------------------------------------------------------------------------------------
#prior_prob_of_classes is a dictionary with the class numbers as the keys, and their prior probabilities
#as the values

#conditional_prob_of_features is a dictionary with tuples containing the feature name, feature value, and class
#as the keys, and the corresponding conditional probabilities as values

#number_of_examples_in_class is a dictionary with the class numbers as the keys, and the number of examples in
#the classes as the values

#classes is a list containing all of the class numbers

#features is a list containing all of the feature names

#number_of_classes is an int representing the number of classes present in the training set

#number_of_features is an int representing the number of features present in the training set

#number_of_examples is an int representing the number of examples present in the training set

#dataframe is a pandas dataframe corrsponding to the training set
#--------------------------------------------------------------------------------------------------------------------------

#RUN THE ModelTest FILE TO SEE THE MODEL TESTED ON A SMALL DATASET

#--------------------------------------------------------------------------------------------------------------------------

import pandas as pd

class NaiveBayesModel:
     def __init__(self, dataframe):
        self.prior_prob_of_classes = {}
        self.conditional_prob_of_features = {}
        self.number_of_examples_in_class = {}
        self.classes = []
        self.features = []
        self.number_of_classes = 0
        self.number_of_features = 0
        self.number_of_examples = 0
        self.dataframe = dataframe
     
#--------------------------------------------------------------------------------------------------------------------------

     #Calculates the prior probability of a given class (ONLY CALL AFTER CALLING load_data())
     def class_probability(self, class_number):
        prior_probability = self.number_of_examples_in_class[class_number]/self.number_of_examples
        return(prior_probability)
     
#--------------------------------------------------------------------------------------------------------------------------
     
     #Calculates the conditional probability of a feature equaling a particular value given a class (ONLY CALL AFTER CALLING load_data())
     #Note that this method also implements the Laplace smoothing specified in step 3 of the assignment handout
     def feature_prob_given_class(self, feature_name, feature_value, class_number):
         feature_val_occurs_in_class = 0
         for i in range(len(self.dataframe)):
             if(self.dataframe.loc[i, feature_name] == feature_value and self.dataframe.loc[i, "class"] == class_number):
                 feature_val_occurs_in_class += 1
         return((feature_val_occurs_in_class + 1)/(self.number_of_examples_in_class[class_number] + self.number_of_features))
         
#--------------------------------------------------------------------------------------------------------------------------
     #Evaluates dataset for number of classes, number of examples, number of features, etc., and updates fields accordingly
     def load_data(self):
         
         #Find # of classes and add each class to the classes list, and update number_of_examples_in_class
         for Class in self.dataframe["class"]:
             if (Class not in self.classes):
                 self.classes.append(Class)
                 self.number_of_classes += 1
                 self.number_of_examples_in_class[Class] = 1
             else:
                 self.number_of_examples_in_class[Class] += 1

         #Find # of features and add each feature to the features list
         for column in self.dataframe.columns:
             if(column != "id" and column != "class"):
                 self.features.append(column)
                 self.number_of_features += 1

         #Find # of examples
         self.number_of_examples = len(self.dataframe)

#--------------------------------------------------------------------------------------------------------------------------

     #Invokes class_probability() and feature_prob_given_class() to implement the Naive Bayes training algorithm
     def train_model(self):
        
        #First, calculate and save the prior probability of each class to class_probability
        for Class in self.classes:
            self.prior_prob_of_classes[Class] = self.class_probability(Class)
        print("\nPrior Probabilities of each Class:\n----------------------------------\n{}".format(self.prior_prob_of_classes))

        #Next, calculate the conditional probabilities of the features equalling a particular value given each class and save it to feature_prob_given_class
        for column in self.dataframe:
            if(column != "id" and column != "class"):
                for i in range(len(self.dataframe)):
                    feature_value = self.dataframe.loc[i, column]

                    #Check if the feature name and feature value appear in the order (feature name, feature value, X) in the tuple keys of conditional_prob_of_features
                    if(any((column, feature_value) in zip(key, key[1:]) for key in self.conditional_prob_of_features) == False):
                        for Class in self.classes:
                            self.conditional_prob_of_features[(column, feature_value, Class)] = self.feature_prob_given_class(column, feature_value, Class)
        print("\nConditional Probabilities of Features Equalling a Value Given a Class:\n(Keep in mind that Laplace smoothing was used)\n---------------------------------------------------------------------\n{}".format(self.conditional_prob_of_features))

#--------------------------------------------------------------------------------------------------------------------------
