"""
@author: Michal Ashkenazi
"""

import numpy                   as np
import pandas                  as pd
import matplotlib.pyplot       as plt
import sklearn.model_selection as model_selection
from   sklearn                 import datasets

# activation function
def sigmoid(x):
    """
    sigmoid function (logistic function): s(x) = 1 / (1 + e^-x)
    """
    s = 1 / (1 + np.exp(-x))
    return x

class LogisticRegression:
    def __init__(self, lr=1e-6, max_iter=100, precision=0):
        self.lr        = lr
        self.max_iter  = max_iter
        self.precision = precision
        self.loss      = []
        self.acc       = []
        self.loss_val  = []
        self.acc_val   = []
        
    def get_hypothesis(self, x):
        return sigmoid(np.dot(x, self.W) + self.b) 
    
    def nagative_log_likelihood(self, X, y, epsilon = 1e-7):
        """
        Negative log likelihood (the loss function):
        
        J = 1/m * (- y * log(h) - (1 - y) * log(1 - h))
        """
        h = self.get_hypothesis(X)
        a = - y * np.log(h + epsilon) - (1 - y) * np.log(1 - h + epsilon)
        a[np.isnan(a)] = 0
        j = np.mean(a)
        return j
    
    def calc_gradient(self, x_batch, y_batch):
        """
        The derivative of the loss function:
        
        J' = |y_predict - y| * x
        """
        h        = self.get_hypothesis(x_batch)
        gradient = np.dot(x_batch.T, np.abs(h - y_batch))
        return gradient
        
    def split_data(self, X, y, batch_size):
        shuffle   = np.random.permutation(y.shape[0])
        idx       = np.arange(start=batch_size, stop=X.shape[0], step=batch_size) 
        X_batches = np.split(X[shuffle], idx)
        y_batches = np.split(y[shuffle], idx)
        return X_batches, y_batches
        
        
    def fit(self, X, y, X_val, y_val, batch_size):
        
        # initilizing weights and bias
        self.W = np.zeros((X.shape[1], 1))
        self.b = np.zeros(1,)
        
        # split data into batches
        X_batches, y_batches = self.split_data(X, y, batch_size)
        
        
        for i in range(self.max_iter):
            for x_batch, y_batch in zip(X_batches, y_batches):
                y_batch = np.reshape(y_batch, (y_batch.shape[0], 1))
                gradient = self.calc_gradient(x_batch, y_batch)
                
                # update weights
                self.W = self.W + self.lr * gradient
                self.b = self.b + self.lr * np.mean(gradient, axis=0)
                
            self.loss.append(self.nagative_log_likelihood(X, y))
            self.loss_val.append(self.nagative_log_likelihood(X_val, y_val))
            self.acc.append(self.get_accuracy(X, y))
            self.acc_val.append(self.get_accuracy(X_val, y_val))
            
            # early stopping
            if self.loss[i] < self.precision:
                break
            
        return self.loss, self.loss_val, self.acc, self.acc_val
                
                
    def predict(self, x):
        y         = self.get_hypothesis(x)
        y[y>=0.5] = 1
        y[y<0.5]  = 0
        return y
    
    def get_accuracy(self, X, y):
        y          = np.reshape(y, (y.shape[0], 1))
        correct    = self.predict(X) == y
        acc        = np.sum(correct) / len(y) * 100
        return acc


if __name__ == "__main__":
    # load data:
    X, y = datasets.load_iris(return_X_y=True)
    
    print(np.unique(y, return_counts=True))
    y[y != 0] = 1
    y[y == 0] = 0
    
    # -- polt all data -- 
    plt.title("iris - all data")
    plt.scatter(X[:,0], X[:,1], c= np.asarray(y))
    plt.show()

    # -- split data to test, train --
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, 
                                                                        test_size=0.2, 
                                                                        random_state=42, 
                                                                        stratify=y)

    # -- define the model --
    classifier = LogisticRegression()

    # -- Train -- 
    loss, loss_val, acc, acc_val = classifier.fit(X_train, y_train, 
                                                X_test, y_test, 
                                                batch_size = 16)

    # -- plot loss --
    plt.title('loss')
    plt.plot(loss, label='train loss')
    plt.plot(loss_val, label='test loss')
    plt.legend()
    plt.show()

    # -- plot acc --
    plt.title('accuracy')
    plt.plot(acc, label='train acc')
    plt.plot(acc_val, label='test acc')
    plt.legend()
    plt.show()

