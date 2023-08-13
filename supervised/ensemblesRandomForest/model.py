# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 07:55:59 2021

@author: Michal Ashkenazi
"""
import os
import math
import numpy                    as np
import pandas                   as pd
import matplotlib.pyplot        as plt
from sklearn                    import model_selection
from sklearn.model_selection    import KFold
from datasets.config            import DATA_DIR

class Leaf:
    def __init__(self, value):
        self.value = value
        
    def __str__(self):
        return f'class: {self.value}'
    
    def get_value(self):
        return self.value 
    
    def predict(self, row):
        return self.get_value()
    
class Node:
    def __init__(self, level, split_feature, split_value, left_node=None, right_node=None):
        self.level = level
        self.split_feature = split_feature
        self.split_value = split_value
        self.left_node = left_node
        self.right_node = right_node
        
    def predict(self, row):
         if row[self.split_feature] >= self.split_value:
             return self.right_node.predict(row)
         return self.left_node.predict(row)
        
class GiniDesisionTreeClassifier:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.root = None
        
    # -- Getters and setters -- 
    def get_depth(self):
        return self.depth
    
    def get_root(self):
        return self.root 
    
    def set_depth(self, depth):
        self.depth = depth
        
    def set_root(self, node):
        if self.root == None:
            self.root = node
            
    # -- Methods --
    def class_counts(self, y):
        '''
        return for each unique member in y (label) his number of appearances 
        '''
        values, counts = np.unique(y, return_counts=True)
        return counts
    
    def calc_popular_class(self, y):
        values, counts = np.unique(y, return_counts=True)
        idx = np.argmax(counts)
        popular_class = values[idx]
        return popular_class
        
    def calc_gini(self, y):
        '''
        Gini score is how good a split is
        gini = 1 - sum(p^2)
        '''
        class_probabilities = self.class_counts(y) / float(len(y))
        return 1 - np.sum(class_probabilities**2, axis=0)
            
    def features_to_check(self, num_features):
        num_features_to_check = int(math.sqrt(num_features))
        idxs = np.random.randint(0, num_features, size=num_features_to_check)
        return idxs
        
    def get_best_split(self, X, y):
        num_features = X.shape[1] 
        num_rows = len(y)
        best_split_feature = 0
        best_split_value = 0
        best_gini = 1
        
        for feature in self.features_to_check(num_features-1): 
            values = np.unique(X[:,feature])
            for val in values:
                
                # -- split data for specific value and featue --
                right_rows, right_labels, left_rows, left_labels = self.data_split(X, y, feature, val)
                
                # -- clac average gini - (check if split good) -- 
                p = float(len(right_rows)) / num_rows
                average_gini = p * self.calc_gini(right_labels)/num_rows + (1-p) * self.calc_gini(left_labels)/num_rows
                
                if average_gini < best_gini:
                    best_gini = average_gini
                    best_split_feature, best_split_value = feature, val
                    
        return best_split_feature, best_split_value, best_gini
    
    def data_split(self, X, y, split_feature, split_value):
        # -- right --
        idx_right_subtree = X[:,split_feature] >= split_value
        right_subtree = X[idx_right_subtree]
        right_subtree_labels = y[idx_right_subtree]
        
        # -- left --
        idx_left_subtree = X[:,split_feature] < split_value
        left_subtree = X[idx_left_subtree]
        left_subtree_labels = y[idx_left_subtree]
        
        return right_subtree, right_subtree_labels, left_subtree, left_subtree_labels
    
    # -- building the tree --
    def fit(self, X, y):
        self.set_root(self.split_node(X, y))
        
    def split_node(self, X, y, node_level=0):
        node_level += 1
        
        if len(y) == 1:
            return Leaf(y[0])
        
        split_feature, split_value, gini = self.get_best_split(X, y)
        
        if gini == 0.0 or self.max_depth < node_level:
            popular_class = self.calc_popular_class(y)
            return Leaf(popular_class)
        
        right_subtree, right_subtree_labels, left_subtree, left_subtree_labels = self.data_split(X, y, split_feature, split_value)
        
        if len(right_subtree_labels) == 1:
            return Leaf(right_subtree_labels[0])
        if len(left_subtree_labels) == 1:
            return Leaf(left_subtree_labels[0])
        
        right_node = self.split_node(right_subtree, right_subtree_labels, node_level)
        left_node = self.split_node(left_subtree, left_subtree_labels, node_level)
        
        return Node(node_level, split_feature, split_value, left_node, right_node)
    
    def predict_labels(self, X_test):
        y_probs = []
        
        for row in X_test:
            y_probs.append(self.root.predict(row))
        
        return np.asarray(y_probs)
    
    def get_accuracy(self, y, y_probs):
        correct = y == y_probs
        acc = ( np.sum(correct) / float(len(y)) ) * 100.0
        return acc


class RandomForest:
    def __init__(self):
        self.forest = []
    
    def create_subsample(self, X, y, a=0.25):
        '''
        return sub sample of size n' of the dataset.
        n' = a*n
        '''
        n = len(y)
        number_of_samples = int(a*n)
        idx = np.random.randint(0, n, size=number_of_samples)
        X_subsample = X[idx]
        y_subsample = y[idx]
        return X_subsample, y_subsample
    
    def fit(self, X, y, T = 300, max_depth=4):
        '''
        T : number of trees in the forest. The default is 300.
        '''
        for i in range(0,T):
            X_subsample, y_subsample = self.create_subsample(X, y)
            tree = GiniDesisionTreeClassifier(max_depth)
            tree.fit(X_subsample, y_subsample) 
            self.forest.append(tree)
    
    def calc_popular_class(self, y):
        values, counts = np.unique(y, return_counts=True)
        idx = np.argmax(counts)
        popular_class = values[idx]
        return popular_class
    
    def bagging_predict(self, X_test):
        predictions = []
        
        for row in X_test:
            all_trees_preds = np.asarray([tree.root.predict(row) for tree in self.forest])
            predictions.append(self.calc_popular_class(all_trees_preds))
        
        return np.asarray(predictions)
             
    def get_accuracy(self, y, y_probs):
        correct = y == y_probs
        acc = ( np.sum(correct) / float(len(y)) ) * 100.0
        return acc

if __name__ == '__main__':
    DATA_PATH   = os.path.join(DATA_DIR, 'wdbc.data')
    KFOLD_PARAMETER = 4
    
    # -- Data --
    breast_cancer_data = pd.read_csv(DATA_PATH, header=None)
    X = np.asarray(breast_cancer_data.iloc[:, 2:])
    y = np.asarray(breast_cancer_data.iloc[:, 1]).astype('str')
    
    # -- Split to train and test --
    #X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25)
    
    # -- cross validation model --
    kfold = KFold(KFOLD_PARAMETER)
    
    # -- define and train the model --
    for train_idx, val_idx in kfold.split(X):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        forest = RandomForest()
        forest.fit(X_train, y_train)
        preds = forest.bagging_predict(X_val)
        acc = forest.get_accuracy(y_val, preds)
        print("accuracy is: ",  acc)
        
#accuracy is:  88.81118881118881
#accuracy is:  96.47887323943662
#accuracy is:  95.77464788732394
#accuracy is:  97.1830985915493
