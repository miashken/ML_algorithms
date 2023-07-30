"""
@author: Michal Ashkenazi
"""
import os
import numpy                   as np
import pandas                  as pd
import matplotlib.pyplot       as plt
from sklearn.model_selection   import train_test_split
from datasets.config           import DATA_DIR

class SVMClassifier:
    """
    Support Vector Machine with Stochastic Gradient Descent.
    """
    def __init__(self, learning_rate=1e-4, lambda_param=1e-4, max_iter=10000):
        self.learning_rate = learning_rate
        self.lambda_param  = lambda_param
        self.max_iter      = max_iter
        self.w             = None    # weights
        self.b             = None    # bias
        self.train_loss    = []
        self.val_loss      = []
        self.train_acc     = []
        self.val_acc       = []
        
    def hinge_loss(self, X, y):
        '''
        hinge_loss = (1/n) * max(0, 1 - y * (<W,X> + b)) + lambda * ||W||^2
        '''
        classification_term = 1 - y * (np.dot(X, self.w) + self.b)
        regularization_term = self.lambda_param * (np.linalg.norm(self.w)**2)
        hinge_loss          = np.mean(np.maximum(0, classification_term)) + regularization_term
        return hinge_loss
    
    
    def derivative_hinge_loss(self, err, xi, yi):
        """
        the derivative for the hinge loss with respect to W is:
        
            2*lambda*W              if   y * (<W,X> + b) > 1 
            2*lambda*W - y*X        otherwise
            
        the derivative for the hinge loss with respect to b is:
        
            0                       if   y * (<W,X> + b) > 1 
            -y                      otherwise
        """
        if err > 1:
            d_w = 2*self.lambda_param*self.w
            d_b = 0
        else:
            d_w = 2*self.lambda_param*self.w - yi*xi
            d_b = -yi

        return d_w, d_b
    
        
    def fit(self, X, y, X_val, y_val):
        """
        Train the model using gradient decsent. 
        """
        # initilizing weights and bias
        self.w = np.ones(X.shape[1])
        self.b = 1
        
        for j in range(self.max_iter):
            for i, xi in enumerate(X):
                
                # calc y * (<W,X> + b)
                err = y[i]*(np.dot(X[i], self.w) + self.b)
                
                # update weights
                d_w, d_b = self.derivative_hinge_loss(err, X[i], y[i])
                self.w -= self.learning_rate * d_w
                self.b -= self.learning_rate * d_b
                
            if j % 100 == 0:
                # update loss and accuracy lists
                self.train_loss.append(self.hinge_loss(X, y))
                self.train_acc.append(self.get_accuracy(X, y))
                
                self.val_loss.append(self.hinge_loss(X_val, y_val))
                self.val_acc.append(self.get_accuracy(X_val, y_val))
    
        return self.w, self.b
    
    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)
    
    def get_accuracy(self, X, y):
        correct = self.predict(X) == y
        acc     = sum(correct) / len(y) * 100
        return acc
    
if __name__ == "__main__":
    # load dataset
    data_path   = os.path.join(DATA_DIR, 'pima-indians-diabetes.csv')
    df          = pd.read_csv(data_path, header=None)

    # change lables to be {1, -1}
    df = df.values
    X = df[:,:-1]
    y = df[:,-1]
    y = 2*y-1

    # plot data
    index_1 = 2
    index_2 = 1

    vals_minus1 = X[y==-1]
    vals_plus1  = X[y==1]

    plt.scatter(vals_minus1[:,index_1], vals_minus1[:,index_2], s=7, c="r", label="YES")
    plt.scatter(vals_plus1[:,index_1], vals_plus1[:,index_2], s=7, c="c", label="NO")
    plt.legend()

    # split data to train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=42)

    # train the model
    svm_model = SVMClassifier()
    svm_model.fit(X_train, y_train, 
                X_val=X_test, 
                y_val=y_test)

    # plot curvs
    plt.title("loss")
    plt.plot(svm_model.train_loss, label="train loss")
    plt.plot(svm_model.val_loss, label="val loss")
    plt.legend()
    plt.show()
    plt.title("accuracy")
    plt.plot(svm_model.train_acc, label="train acc")
    plt.plot(svm_model.val_acc, label="val acc")
    plt.legend()
    plt.show()

    print(svm_model.get_accuracy(X_test, y_test))
    print(svm_model.hinge_loss(X_test, y_test))
