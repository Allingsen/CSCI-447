import pandas as pd
import numpy as np
import random


class DataProcess():
    def __init__(self, names: list, missing_val: str= None, id_col: str=None, cat_class: bool=None) -> None:
        self.cols = names
        if len(self.cols) != len(set(names)):
            raise NameError('Column names need to be unique!')
        if 'class' not in self.cols:
            raise NameError('Must have a class column!')
        self.id_col = id_col
        self.missing_val = missing_val
        self.cat_class = cat_class
        self.categorical_cols = list()

        self.df = pd.DataFrame(columns=self.cols)

    #--------------------------------------------------------------------------------------------------------------------------

    def _num_or_cat(self, row: list) -> None:
        '''Determines if the column is numeric or categorical'''
        # Attempts to map the row to ints
        try:
            new_row = list(map(lambda x: int(x), row))
            return new_row
        # If that fails, check if there are missing values
        except ValueError:
            new_row = list()
            for i in range(len(row)):
                try: new_row.append(int(row[i]))
                # If not missing val, and i is not already a categorical column, add it to categoical columns
                except:
                    if row[i] != self.missing_val and self.cols[i] not in self.categorical_cols and self.cols[i]: 
                        self.categorical_cols.append(self.cols[i])
                    new_row.append(row[i])
            return new_row
        
    #--------------------------------------------------------------------------------------------------------------------------
            
    def _get_cat_vals(self, col: pd.Series, k: int):
        '''Returns a list of specified length of randomly selected, weighted values'''
        values_weights = dict(col.value_counts())
        values = list(values_weights.keys())
        weights = [x/sum(values_weights.values()) for x in values_weights.values()]
        
        return random.choices(values, weights, k=k)

    #--------------------------------------------------------------------------------------------------------------------------

    def min_max_normalize(self):
        '''Normalizes numerical columns'''
        for i in self.cols:
            if i in self.categorical_cols or i == 'class':
                continue
            else:
                col = self.df[i].to_numpy()
                minimum = min(col)
                maximum = max(col)
                col = col - minimum
                col = col / (maximum-minimum)
                self.df[i] = col


    def loadCSV(self, path_to_data: str) -> None:
        '''Opens a .DATA file and converts it to a cleaned Pandas DataFrame'''
        # Opens file and adds to dataframe 
        with open(path_to_data) as file:
            all_data = file.readline()
            while all_data:
                list_data = all_data.strip().split(',')
                # Checks if the column is numeric or categorical
                new_row = self._num_or_cat(list_data)
                # Confirms the columns have been added via the .NAMES file
                try:
                    self.df.loc[len(self.df)] = new_row
                except ValueError as e:
                    print(f'{e}: Column names do not match. Did you use the correct names file?')
                    return
                
                all_data = file.readline()
        
        # Deals with missing values if provided
        if self.missing_val:
            self.df.replace(self.missing_val, value=None, inplace=True)
            for i in self.df.columns:
                # If there are any null values in categorical columns, randomly fill them following the value
                # distribution of that column
                if self.df[i].isnull().any() and i in self.categorical_cols:
                    missing_num = self.df[i].isnull().sum()
                    replace_vals = pd.Series(self._get_cat_vals(self.df[i], missing_num))
                    self.df[i].fillna(replace_vals, inplace=True)
                # If there are null values in numeric columns, fill with the mean
                elif self.df[i].isnull().any():
                    self.df[i].fillna(self.df[i].mean(skipna=True), inplace=True)

        # One hot encodes all categorical columns
        if self.categorical_cols:
            # If the class is categorical, do not encode
            if self.cat_class:
                columns = [x for x in self.categorical_cols if x != 'class']
            else:
                columns = self.categorical_cols
            self.df = pd.get_dummies(self.df, columns=(columns), dtype=int)

        # Normalizes Numerical Columns
        self.min_max_normalize()

        # Sends the class column to the end of the df, shuffles the data
        self.df['class'] = self.df.pop('class')
        self.df = self.df.sample(frac=1).reset_index(drop=True)
    #--------------------------------------------------------------------------------------------------------------------------

    def _num_in_split(self, folds:int) -> list:
        '''Returns the number of examples in each split using n folds'''
        group_len = list()
        examples = len(self.df)
        # Finds the percentage of the data that should be in each fold
        fold_percent = examples / folds
        for i in range(folds):
            # On data that does not perfectly fit n groups, minimize the distance so every fold is withing
            # 1 data point
            if examples/(folds-i) == np.ceil(fold_percent):
                num_in_fold = np.ceil(fold_percent)
            else:
                num_in_fold = np.floor(fold_percent)
            examples -= num_in_fold
            group_len.append(num_in_fold)
        return group_len

    #--------------------------------------------------------------------------------------------------------------------------

    def _distr_folds(self, distr, group_len) -> list:
        '''Changes the distribution of folds to be accurate to the data'''
        # Finds the real data counts
        real_vals = self.df['class'].value_counts().values
        values = np.array([0] * len(distr))
        for i in group_len:
            # Checks the number of samples in each fold by class, adds them to a matrix
            class_total = np.round(i * distr)
            values = np.vstack((values, class_total))

        # Checks the sum of each column (class) in the matrix
        col_sums = np.sum(values, axis=0)
        
        # Checks if the real values are equal to the calculated values
        test = real_vals == col_sums
        change_val = -1
        while not test.all():
            # If not, subtract or add one depending on whether the predicted number is above or below
            condlist = [col_sums < real_vals, col_sums > real_vals]
            choicelist = [values[change_val] +1, values[change_val] -1]
            values[change_val] = np.select(condlist, choicelist, values[change_val])

            col_sums = np.sum(values, axis=0)
            test = real_vals == col_sums
            # Move to the next back distribution
            change_val -=1
        # Do not include the first value as it is just a placeholder
        return(list(values[1:]))

    #--------------------------------------------------------------------------------------------------------------------------

    def k_fold_split(self, folds:int) -> list:
        '''Splits the data into n folds and returns n training and test sets''' 
        # Initilizes a DF to be modified
        mod_df = self.df.copy()
        # Creates a df for each class
        classes = list()
        for i in mod_df['class'].unique():
            classes.append(mod_df[mod_df['class'] == i])
        
        # Gets the length of n folds, keeping all within a 1-example difference
        
        group_len = self._num_in_split(folds)
        
        # Calculates the distribution of the data between classes
        class_names = mod_df['class'].value_counts().keys()
        distr = (mod_df['class'].value_counts().values)/len(mod_df)
        
        # Calculates the number of datapoints belonging to each class should be in each fold
        cross_vals = [pd.DataFrame()] * folds
        class_totals = self._distr_folds(distr, group_len)
        for i,num in enumerate(class_totals):
            for j in range(len(num)):
                # Takes a random sample without replacement of each class which number of datapoints selected
                # is the calculated number belonging to its class, then removes them from the df
                sample = mod_df[mod_df['class'] == class_names[j]].sample(n=int(num[j]), replace=False)
                mod_df.drop(sample.index, axis=0,inplace=True)
                cross_vals[i] = pd.concat([cross_vals[i], sample], axis=0, ignore_index=True)
        
        # Returns each fold in a list
        return cross_vals
    
    #--------------------------------------------------------------------------------------------------------------------------
