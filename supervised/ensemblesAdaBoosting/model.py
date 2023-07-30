"""
@author: Michal Ashkenazi
"""
import os
import numpy             as np
import pandas            as pd
from sklearn             import model_selection
from sklearn.tree        import DecisionTreeClassifier
from datasets.config     import DATA_DIR

class AdaBoostClassifier:
    """
    Adaptive Boosting:
    Machine learning mata algorithem boosting technique, that uses 
    weak learners in order to form a strong learner.
    It is called Adaptive Boosting as the weights are re-assigned 
    to each instance, with higher weights assigned to incorrectly 
    classified instances.
    """
    
    def __init__(self, base_learner, T=100):
        self.T = T
        self.base_learner = base_learner
        
    def fit(self, X_train, y_train):
        
        N        = X_train.shape[0]
        self.W   = np.full(N, fill_value=1/N)
        
        self.h_t = [] #base learners
        self.a_t = [] #base learners weights (one weight for each one of them)
        
        for i in range(self.T-1):
            
            # (a) train base learner:
            ht = self.base_learner.fit(X_train,y_train)
            self.h_t.append(ht)
            
            # (b) alc ht err:
            incorrect = 1 - (ht.predict(X_train) == y_train)
            errt = np.sum(self.W*incorrect) / np.sum(self.W, axis=0)
            
            # (c) calc ht weight:
            at = np.log((1 - errt) / errt)
            self.a_t.append(at)
            
            # (d) update weights:
            self.W = self.W * np.exp(at*incorrect)

    def predict(self, X):
        
        # get predictions from all base leraner
        y_preds = []
        for ht in self.h_t:
            y_preds.append(ht.predict(X))
        
        # orgenizing arrays from matrix multiplication
        y_preds = np.asarray(y_preds)
        self.a_t = np.array(self.a_t).reshape(len(self.a_t), 1)
        
        # getting prediction based on model weights
        preds = np.sign(y_preds.T@self.a_t).flatten()
       
        return preds 
            
    def get_accuracy(self, X, y):
        acc = sum(self.predict(X) == y) / len(y)
        return acc
    
if __name__ == "__main__":
    # data preparation:
    DATA_PATH   = os.path.join(DATA_DIR, 'pima-indians-diabetes.csv')
    data        = pd.read_csv(DATA_PATH, header=None)
    
    X = np.asarray(data.iloc[:, :-1])
    y = np.asarray(data.iloc[:, -1])
    y[y==0] = -1
    
    # split data to train and test:
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,
                                                                        test_size=0.20,
                                                                        random_state=42)
    
    # define a base model:
    base_model = DecisionTreeClassifier(max_depth=3)


    # define and train adaboost:
    model = AdaBoostClassifier(base_learner=base_model,
                            T=100)
    model.fit(X_train,y_train)
    
    # model evaluation:
    acc = model.get_accuracy(X_test, y_test) 
    print(acc)
 