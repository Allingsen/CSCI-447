#ModelTest inherits from NaiveBayesModel
import pandas as pd
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

    #Classifies all feature vectors in test_data
    def classify_all(self):
        print("\nPredicted Classes for Test Set:\n-------------------------------")
        for i in range(len(self.test_data)):
            feature_vector = {}
            for column in self.test_data.columns:
                if(column != "id" and column != "class"):
                    feature_vector[column] = self.test_data.loc[i, column]
            classification = self.classify_one(feature_vector)
            print("Vector {}: Class {}".format(list(feature_vector.values()), classification))

#--------------------------------------------------------------------------------------------------------------------------

test_data = {
    "Feature_1": [0,1,1,2,0],
    "Feature_2": [1,1,0,0,3],
    "Feature_3": [1,2,2,2,1],
    "class": [1,2,1,3,1]
}

training_data = {
    "Feature_1": [0,1,0,2,0],
    "Feature_2": [1,1,0,3,3],
    "Feature_3": [1,0,0,2,1],
    "class": [1,2,3,1,1]
}

#Test the model on an example dataset (I made up the dataset so the training set and testing set are random and have no intentional correlation)
training_df = pd.DataFrame(training_data)
test_df = pd.DataFrame(test_data)
print("Test Set\n--------\n", test_df)
print("\nTraining Set\n------------\n", training_df)

test = ModelTest(training_df, test_df)
test.load_data()
test.train_model()
test.classify_all()