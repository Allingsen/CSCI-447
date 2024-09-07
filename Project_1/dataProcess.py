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

        
        self.df['class'] = self.df.pop('class')
        print(self.df)

    def k_fold_split(self) -> None:
        pass
        
x = DataProcess(['id',2,3,4,5,6,7,8,9,10,'class'], missing_val='?', id_col='id')
x.loadCSV('Project_1/datasets/breast-cancer-wisconsin.data')

#x = DataProcess(['id',2,3,4,5,6,7,8,9,10,'class'], missing_val='?', id_col='id')
#x.loadCSV('Project_1/datasets/glass.data')

#x = DataProcess(['test',2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,'class'], '?')
#x.loadCSV('Project_1/datasets/house-votes-84.data')

