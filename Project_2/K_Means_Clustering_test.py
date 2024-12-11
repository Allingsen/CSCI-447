import numpy as np
import pandas as pd
import random
from collections import Counter

class KMeansClustering:
    def __init__(self, df_train:pd.DataFrame, k, reg=False) -> None:
        self.df_train = df_train.to_numpy(dtype=str)
        self.k = k
        self.reg = reg
        self.centroids = []
        self.cat_cols = {len(self.df_train[0])-1}

        self._initialize_centroids()

    def _initialize_centroids(self):
        mod_df = self.df_train
        for _ in range(self.k):
            i = random.randint(0, self.k-1)
            centroid = mod_df[i]
            np.delete(mod_df, i)
            self.centroids.append(centroid)

    #--------------------------------------------------------------------------------------------------------------------------

    def calculate_distances(self, point:np.array) -> str:
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
        new_attrs = np.delete(new_point, -1)
        point_attrs = np.delete(data_point, -1)
        if self.reg:
            calc_dist = self._euclidean_dist(new_attrs, point_attrs)
        else:
            calc_dist = self._diff_val(new_point, point_attrs)
        return calc_dist, data_point
    
    #--------------------------------------------------------------------------------------------------------------------------

    def _euclidean_dist(self, test:np.array, point:np.array) -> float:
        '''Calculates Euclidean Distance between two points'''
        dist_vec = (test.astype(float))-(point.astype(float))
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
                self.cat_cols.add(i)
                vals_same_test = list()
                vals_same_point = list()
                for j in self.df_train:
                    if test[i] == j[i]:
                        vals_same_test.append(j)
                    if point[i] == j[i]:
                        vals_same_point.append(j)
                C_i = len(vals_same_test)
                C_j = len(vals_same_point)
                delta = 0
                for k in np.unique(self.df_train[:, -1]):
                    C_ia = len(list(filter(lambda x:x[-1] == k, vals_same_test)))
                    C_ja = len(list(filter(lambda x:x[-1] == k, vals_same_point)))
                    inner_prod = ((C_ia/C_i) - (C_ja/C_j)) ** 2
                    delta += inner_prod
                D_cat.append(delta)
        distance = (len(D_num)/len(point))*(np.linalg.norm(D_num)) + (len(D_cat)/len(point))*np.sqrt(sum(D_cat))
        return distance   
    

    def calculate_centroids(self):
        converged = False
        while not converged:
            new_centroids = []
            points_centroids = [0] * len(self.df_train)
            for i, val in enumerate(self.df_train):
                best_dist = -1
                for j in self.centroids:
                    dist, point = self._calculate_distance(val, j) 
                    if dist < best_dist or best_dist < 0:
                        points_centroids[i] = point
                        best_dist = dist

            for i in self.centroids:
                try:
                    vec_list = [self.df_train[x] for x in range(len(points_centroids)) if (points_centroids[x]==i).all()]
                    matrix = np.vstack(vec_list)
                    average_vec = np.array([])
                    for j in range(len(matrix[0])):
                        if j not in self.cat_cols:
                            float_val = matrix[:, j].astype(float)
                            average_vec = np.append(average_vec, np.sum(float_val)/len(matrix[:, j]))
                        else:
                            most_common = Counter(matrix[:, j]).most_common(1)[0][0]
                            average_vec = np.append(average_vec, most_common)

                    new_centroids.append(average_vec)
                except:
                    self.centroids = [troids for troids in self.centroids if (troids==i).all()]

            break_check = True
            print('---')
            for i in range(len(new_centroids)):
                classless_new = new_centroids[i][:-1]
                classless_old = self.centroids[i][:-1]
                if not (classless_new==classless_old).all():
                    break_check = False
                    break

            self.centroids = new_centroids

            if break_check:
                converged = True

        return self.centroids