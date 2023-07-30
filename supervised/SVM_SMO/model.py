"""
@author: Michal Ashkenazi
"""
import os
import math
import random
import numpy                   as np
import pandas                  as pd
from sklearn.model_selection   import train_test_split
from datasets.config           import DATA_DIR


class SVMClassifier:
    """
    Support Vector Machine with Sequential Minimal Optimization (SMO)
    """
    def __init__(self, max_iter=1000, kernel_type="linear", C=1.0, epsilon=0.0001):
        self.kernels  = {"linear":    self.kernel_linear,
                         "quadratic": self.kernel_quadratic,
                         "rbf":       self.kernel_rbf}
        self.kernel   = self.kernels[kernel_type]
        self.max_iter = max_iter
        self.C        = C
        self.epsilon  = epsilon
        self.W        = None    # weights
        self.b        = None    # bias
        
    def kernel_linear(self, x1, x2):
        """
        Linear kernel: k(x1,x2) = x1*x2.T
        """
        return np.dot(x1, x2.T)
    
    def kernel_quadratic(self, x1, x2):
        """
        Quadratic kernel: k(x1, x2) = (x1*x2.T)^2
        """
        return self.kernel_linear(x1, x2)**2
    
    def kernel_rbf(self, x1, x2):
        """
        RBF kernel: k(x,z)=e^((x−z)^2/σ^2)
        """
        return math.exp((np.linalg.norm(x1-x2)**2) / -0.1**2)
    
    def get_random_int(self, a, b, z):
        """
        return random int in the range of [a,b] excluding z.
        """
        r = list(range(a,z)) + list(range(z+1, b))
        return random.choice(r)
    
    def calc_w(self, X, y, alpha):
        """
        W = α*y*X
        """
        return np.dot(X.T, np.multiply(alpha, y))
        
    def calc_b(self, X, y, W):
        """
        b = 1/n * (y - X*W)
        """
        return np.mean(y - np.dot(X, W))
    
    def E(self, X_k, y_k, W, b):
        """
        calculate prediction error.
        err = sign(Xk*W + b) - yk
        """
        return np.sign(np.dot(X_k, W) + b) - y_k
    
    def compute_L_H(self, C, alpha_prime_j, alpha_prime_i, y_j, y_i):
        """
        calculate bounds for αj
        """
        if y_i != y_j:
            return max(0, alpha_prime_j - alpha_prime_i), min(C, C - alpha_prime_i + alpha_prime_j)
        else:
            return max(0, alpha_prime_i + alpha_prime_j - C), min(C, alpha_prime_i + alpha_prime_j)
        
    def fit(self, X, y):
        n, d  = X.shape
        alpha = np.zeros((n))
        
        for itr in range(self.max_iter):
            
            alpha_prev = np.copy(alpha)
            
            for j in range(0, n):
                
                # get random int i, where i!=j
                i = self.get_random_int(0, n-1, j)
                
                # calculating the second derivative of the objective function along the diagonal line:
                kij = self.kernel(X[i], X[i]) + self.kernel(X[j], X[j]) - 2*self.kernel(X[i], X[j])
                
                if kij <= 0:
                    continue
                
                # storing the current values of αi and αj
                alpha_prime_j, alpha_prime_i = alpha[j], alpha[i]
                
                # calculate bounds for αj
                L, H = self.compute_L_H(self.C, alpha_prime_j, alpha_prime_i, y[j], y[i])
                
                # compute model parameters
                self.W = self.calc_w(X, y, alpha)
                self.b = self.calc_b(X, y, self.W)
                
                # compute Ei, Ej
                E_i = self.E(X[i], y[i], self.W, self.b)
                E_j = self.E(X[j], y[j], self.W, self.b)
                

                # ---------------------
                # set new alpha values:
                # ---------------------
                # αj new = αj + yj(Ei-Ej)/kij
                alpha[j] = alpha_prime_j + float(y[j] * (E_i - E_j)) / kij
                
                # αj clipped:
                alpha[j] = max(alpha[j], L)
                alpha[j] = min(alpha[j], H)
                
                # αi:
                alpha[i] = alpha_prime_i + y[i]*y[j]*(alpha_prime_j - alpha[j])
                
            # check convergence
            # check if there is a difference between previous alpha and the current, if the change is smaller
            # than epsilon, we stop, cuase we found owr suport vectors. otherwise keep going utill convergence.
            diff = np.linalg.norm(alpha - alpha_prev) 
            if diff < self.epsilon:
                break
                    
        # compute final model parameters
        self.W = self.calc_w(X, y, alpha)
        self.b = self.calc_b(X, y, self.W)
                
        # get support vectors
        idx    = np.where(alpha>0)[0]
        SV     = X[idx]
    
        return SV
    
    def predict(self, X):
        return np.sign(np.dot(X, self.W) + self.b)
    
    
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

    # split data to train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=42)

    # train the model
    svm_model = SVMClassifier(kernel_type="linear")
    sv = svm_model.fit(X_train, y_train)

    # get accuracy for test set
    print(svm_model.get_accuracy(X_test, y_test))