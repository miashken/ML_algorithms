"""
@author: Michal Ashkenazi
"""
import os
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
from sklearn             import model_selection
from sklearn.tree        import DecisionTreeRegressor
from sklearn.metrics     import explained_variance_score
from datasets.config     import DATA_DIR

class gradientBoostingRegressor:
    """
    Gradient boosting is regarded as iterative functional gradient descent algorithms, 
    so while gradient descent algorithm search in the parameter space, gradient boosting 
    search in the function space.
    """
    
    def __init__(self, lr=0.1, n_estimators=25, base_learner=DecisionTreeRegressor):
        self.lr           = lr
        self.n_estimators = n_estimators
        self.base_learner = base_learner

    def fit(self, X, y, **params):
        self.base_models = []
        
        # initial the first base model with a constant value
        f0 = np.full(shape=y.shape, fill_value=0.0)

        # update the model
        Fm = f0
        
        # create a subplot for each step prediction 
        _, axs = plt.subplots(5, 5, figsize=(10, 10))
        axs = axs.flatten()

        for i in range(0, self.n_estimators):
            # compute pseudo-residuals (gradient of MSE-loss)
            r_i = y - Fm
            
            # base learner
            h_i = self.base_learner(**params)
            h_i.fit(X,r_i)
            self.base_models.append(h_i)

            # update the model
            Fm = Fm + self.lr*h_i.predict(X)

            # plotting after prediction
            axs[i].plot(y, ".")
            axs[i].plot(Fm, ".")
            axs[i].set_title(str(i))
            axs[i].axis("off")

        plt.show()
        
        return Fm
    
    def predict(self, X):
        y_pred = np.array([])
        
        for h_i in self.base_models:
            update = self.lr*h_i.predict(X)
            y_pred = update if not y_pred.any() else y_pred + update  #y_pred.any() is False at the beginning

        return y_pred

if __name__ == "__main__":
    # data preparation:
    DATA_PATH   = os.path.join(DATA_DIR, "./Fish.csv")
    data        = pd.read_csv(DATA_PATH)
    
    y = np.asarray(data.Weight)
    
    species = pd.get_dummies(data.Species)
    X       = data.drop(columns=["Weight", "Species"])
    X       = np.asarray(pd.concat([X, species], axis=1))
    
    # split data into train and test:
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=42) 

    # define and train the model:
    model = gradientBoostingRegressor()
    r = model.fit(X_train, y_train, max_depth=2)
    
    # get predictions:
    preds = model.predict(X_test)

    score = explained_variance_score(y_test, preds)
    print("model score: ", score)
    
    # plot predictions vs. the ground truth:
    _, ax = plt.subplots(1, 1)
    plt.title("Test")
    ax.plot(y_test, "o", label = "y_test")
    ax.plot(preds, "o", label="preds")
    ax.legend()
    plt.show()
