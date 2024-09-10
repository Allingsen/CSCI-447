import pandas as pd
import numpy as np
import random


class DataProcess():
    def __init__(self, names: list, missing_val: str= None, id_col: str=None) -> None:
        self.cols = names
        if len(self.cols) != len(set(names)):
            raise NameError('Column names need to be unique!')
        if 'class' not in self.cols:
            raise NameError('Must have a class column!')
        self.id_col = id_col
        self.missing_val = missing_val
        self.categorical_cols = list()
        self.binned_cols = list()

        self.df = pd.DataFrame(columns=self.cols)

    def _num_or_cat(self, row: list) -> None:
        '''Determines if the column is numeric or categorical'''
        try:
            new_row = list(map(lambda x: int(x), row))
            return new_row
        except ValueError:
            new_row = list()
            for i in range(len(row)):
                try: new_row.append(int(row[i]))
                except:
                    if row[i] != self.missing_val and self.cols[i] not in self.categorical_cols: 
                        self.categorical_cols.append(self.cols[i])
                    new_row.append(row[i])
            return new_row
        
    def _get_cat_vals(self, col: pd.Series, k: int):
        '''Returns a list of specified length of randomly selected, weighted values'''
        values_weights = dict(col.value_counts())
        values = list(values_weights.keys())
        weights = [x/sum(values_weights.values()) for x in values_weights.values()]
        
        return random.choices(values, weights, k=k)
    
    def _bin_num_vals(self) -> None:
        '''Bins columns if it is needed using the Freedman-Diaconis rule'''
        for i in self.cols :
            if i not in self.categorical_cols and i != self.id_col:
                bins = np.histogram_bin_edges(self.df[i], bins='fd')
                num_of_vals = len(self.df[i].value_counts().keys())
                if len(bins) < num_of_vals:
                    self.binned_cols.append(i)

    def loadCSV(self, path_to_data: str) -> None:
        '''Opens a .DATA file and converts it to a cleaned Pandas DataFrame'''
        # Opens file and adds to dataframe 
        with open(path_to_data) as file:
            all_data = file.readline()
            while all_data:
                list_data = all_data.strip().split(',')
                new_row = self._num_or_cat(list_data)
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
                if self.df[i].isnull().any() and i in self.categorical_cols:
                    missing_num = self.df[i].isnull().sum()
                    replace_vals = pd.Series(self._get_cat_vals(self.df[i], missing_num))
                    self.df[i].fillna(replace_vals, inplace=True)
                elif self.df[i].isnull().any():
                    self.df[i].fillna(self.df[i].mean(skipna=True), inplace=True)

        # Binning and one hot encoding
        self._bin_num_vals()
        if self.categorical_cols or self.binned_cols:
            self.df = pd.get_dummies(self.df, columns=(self.categorical_cols + self.binned_cols), dtype=int)

        # Sends the class column to the end of the df, shuffles the data
        self.df['class'] = self.df.pop('class')
        self.df = self.df.sample(frac=1).reset_index(drop=True)

    def _num_in_split(self, folds:int) -> list:
        group_len = list()
        examples = len(self.df)
        fold_percent = examples / folds
        for i in range(folds):
            if examples/(folds-i) == np.ceil(fold_percent):
                num_in_fold = np.ceil(fold_percent)
            else:
                num_in_fold = np.floor(fold_percent)
            examples -= num_in_fold
            group_len.append(num_in_fold)
        return group_len

    def _distr_folds(self, distr, group_len) -> list:
        real_vals = self.df['class'].value_counts().values
        values = np.array([0] * len(distr))
        for i in group_len:
            class_total = np.round(i * distr)
            values = np.vstack((values, class_total))
        col_sums = np.sum(values, axis=0)
        
        test = real_vals == col_sums
        change_val = -1
        while not test.all():
            condlist = [col_sums < real_vals, col_sums > real_vals]
            choicelist = [values[change_val] +1, values[change_val] -1]
            values[change_val] = np.select(condlist, choicelist, values[change_val])

            col_sums = np.sum(values, axis=0)
            test = real_vals == col_sums
            change_val -=1

        return(list(values[1:]))

    def k_fold_split(self, folds:int) -> list:
        '''Splits the data into n folds and returns n training and test sets'''  
        # Creates a df for each class
        classes = list()
        for i in self.df['class'].unique():
            classes.append(self.df[self.df['class'] == i])
        # Gets the length of n folds, keeping all within a 1-example difference

        group_len = self._num_in_split(folds)

        # Calculates the distribution of the data between classes
        class_names = self.df['class'].value_counts().keys()
        distr = (self.df['class'].value_counts().values)/len(self.df)
        cross_vals = [pd.DataFrame()] * folds

        class_totals = self._distr_folds(distr, group_len)
        mod_df = self.df
        for i,num in enumerate(class_totals):
            for j in range(len(num)):
                sample = mod_df[mod_df['class'] == class_names[j]].sample(n=int(num[j]), replace=False)
                mod_df.drop(sample.index, axis=0,inplace=True)
                cross_vals[i] = pd.concat([cross_vals[i], sample], axis=0)

        return cross_vals
    
    def introduce_noise(self, perc: float) -> pd.DataFrame:
        '''Introduces noise by shuffling the attributes of perc percent of features,
        while keeping the original class'''
        mod_df = self.df
        perc_samp = mod_df.sample(frac=perc, replace=False)
        mod_df.drop(perc_samp.index, axis=0,inplace=True)
        classes = perc_samp.pop('class')
        if self.id_col:
            ids = perc_samp.pop(self.id_col)
        rng = np.random.default_rng()
        shuffled_samp = pd.DataFrame()

        for i in perc_samp.columns:
            if i != self.id_col:
                new_col = pd.Series(rng.permutation(perc_samp[i].values))
                shuffled_samp.insert(len(shuffled_samp.columns), i, new_col)
    
        if self.id_col:
            shuffled_samp.insert(0, self.id_col, list(ids))
        shuffled_samp['class'] = list(classes)

        return pd.concat([mod_df, shuffled_samp], axis=0, ignore_index=True)

    
x = DataProcess(['id',2,3,4,5,6,7,8,9,10,'class'], missing_val='?', id_col='id')
x.loadCSV('Project_1/datasets/breast-cancer-wisconsin.data')
x.introduce_noise(.10)
x.k_fold_split(10)

#x = DataProcess(['id',2,3,4,5,6,7,8,9,10,'class'], missing_val='?', id_col='id')
#x.loadCSV('Project_1/datasets/glass.data')
#x.introduce_noise(.10)#.to_csv('test1.csv')
#x.k_fold_split(10)

#x = DataProcess(['test',2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,'class'], '?')
#x.loadCSV('Project_1/datasets/house-votes-84.data')

