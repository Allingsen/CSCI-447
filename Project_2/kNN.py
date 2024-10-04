import numpy as np
import pandas as pd
from collections import Counter

class KNearestNeighbors():
    def __init__(self, k:int, df:pd.DataFrame, reg=False, sigma=1) -> None:
        self.df = df.to_numpy()
        self.k = k
        self.reg = reg
        if self.reg:   
            self.sigma = sigma
        else:
            raise ValueError('Sigma is not for classification!')

    #--------------------------------------------------------------------------------------------------------------------------

    def calculate_distances(self, point:np.array) -> None:
        '''Calculates the distances between a point and all other data, returns the estimate'''
        classes = list()
        dists = list()
        for i in self.df:
            dist, y = self._calculate_distance(point, i)
            if len(classes) < self.k:
                classes.append(y)
                dists.append(dist)
            elif dist < min(dists):
                idx = dists.index(min(dists))
                classes[idx] = y
                dists[idx] = dist
        
        return self._make_prediction(classes, dists)
    
    #--------------------------------------------------------------------------------------------------------------------------

    def _calculate_distance(self, new_point:np.array, data_point:np.array) -> tuple:
        '''Uses Educlidean or Difference value metrics to get distance'''
        point_attrs = np.delete(data_point, -1)
        if self.reg:
            calc_dist = self._euclidean_dist(new_point, point_attrs)
        else:
            calc_dist = self._diff_val(new_point, point_attrs)
        return calc_dist, data_point[-1]
    
    #--------------------------------------------------------------------------------------------------------------------------

    def _euclidean_dist(self, test:np.array, point:np.array) -> float:
        '''Calculates Euclidean Distance between two points'''
        dist_vec = test.astype('f')-point.astype('f')
        return np.linalg.norm(dist_vec)

    #--------------------------------------------------------------------------------------------------------------------------

    def _diff_val(self, test:np.array, point:np.array) -> float:
        '''Calculates Difference Value Metrics between two points'''
        # Find Probabilities in categorical columns given class found
        D_num = list()
        D_cat = list()
        for i in range(len(point)):
            try:
                # Attempts to use Euclidean distance
                D_num.append(float(test[i]) - float(point[i]))
            except ValueError:
                vals_same_test = list()
                vals_same_point = list()
                for j in self.df:
                    if test[i] == j[i]:
                        vals_same_test.append(j)
                    if point[i] == j[i]:
                        vals_same_point.append(j)
                C_i = len(vals_same_test)
                C_j = len(vals_same_point)
                delta = 0
                for k in np.unique(self.df[:, -1]):
                    C_ia = len(list(filter(lambda x:x[-1] == k, vals_same_test)))
                    C_ja = len(list(filter(lambda x:x[-1] == k, vals_same_point)))
                    inner_prod = ((C_ia/C_i) - (C_ja/C_j)) ** 2
                    delta += inner_prod
                D_cat.append(delta)
        distance = (len(D_num)/len(point))*(np.linalg.norm(D_num)) + (len(D_cat)/len(point))*np.sqrt(sum(D_cat))
        return distance
    
    #--------------------------------------------------------------------------------------------------------------------------

    def _make_prediction(self, classes:list, dists:list) -> str:
        '''Returns our estimate for the data point'''
        if self.reg:
            distances = np.array(dists)
            gaussian = np.exp(-(distances ** 2)/(2 * self.sigma ** 2))
            estimate = np.sum(gaussian * classes) / np.sum(gaussian)
        else:
            estimate = Counter(classes).most_common(1)[0][0]
        return estimate