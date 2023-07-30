"""
@author: Michal Ashkenazi
"""
import os
import math
import numpy            as np
import pandas           as pd
from sklearn            import model_selection
from datasets.config    import DATA_DIR

class NaiveBayesClassifier:
    
    def __init__(self):
        self.prior = None
        self.means = None
        self.stds  = None
        
    def fit(self, X_train, y_train):
        """
        Summarize the Data: calculate priors and likelihoods (gaussian distribution)
        """
        # get classes unique values
        self.unique_labels = y_train.unique()
        
        self.means = np.zeros((len(self.unique_labels), X_train.shape[1]))
        self.stds  = np.zeros_like(self.means)
        self.prior = np.zeros(len(self.unique_labels))
        
        
        for i, label in enumerate(self.unique_labels):
            X_with_label = X_train[y_train == label]
            
            # probability of belonging to class
            self.prior[i] = len(X_with_label) / len(X_train)
            
            # for each feature and class calc mean and std
            self.means[i, :] = X_with_label.mean(axis = 0)
            self.stds[i, :]  = X_with_label.std(axis = 0)

    def gaussian(self, x, mean, std):
        '''
        gaussian naive bayes - When dealing with continuous data, the assumption is that the continuous 
        values associated with each class are distributed according to a normal (or Gaussian) distribution. 
        
        f(x) = (1 / sqrt(2 * PI * sigma**2) * exp(-((x-mean)^2 / (2 * sigma^2)))
        '''
        return np.prod((1 / (np.sqrt(2 * math.pi) * std)) * np.exp( -(x-mean)**2 / (2 * std**2)), axis=1)
    
    def predict(self, X_test):
        """
        For all possible values y, check for what value of y, 𝑃(𝑌|𝑋) is the maximum.
        """
        probs = np.zeros((len(X_test), len(self.unique_labels)))
        
        for i, label in enumerate(self.unique_labels):
            prob_i = self.gaussian(X_test, self.means[i, :], self.stds[i, :]) * self.prior[i]
            probs[:, i] = prob_i
            
        predicted_y = self.unique_labels[np.argmax(probs, axis=1)]
        return predicted_y
        
    def get_accuracy(self, y, y_preds):
        correct = y == y_preds
        acc     = correct.sum() / len(y) * 100.0
        acc     = float("{:.2f}".format(acc))
        return acc
    
if __name__ == "__main__":
    # load dataset:
    DATA_PATH   = os.path.join(DATA_DIR, 'pima-indians-diabetes.csv')
    data      = pd.read_csv(DATA_PATH, header=None)
    X         = data.iloc[:, :-1]
    y         = data.iloc[:, -1]

    # split data into train and test:
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.20, random_state=42)

    # define and train model: 
    naive_bayes_model = NaiveBayesClassifier()
    naive_bayes_model.fit(X_train, y_train)

    # evaluate the model:
    preds = naive_bayes_model.predict(X_test)
    acc   = naive_bayes_model.get_accuracy(y_test, preds)
    print(acc, "%")
