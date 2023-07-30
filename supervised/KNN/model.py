"""
@author: Michal Ashkenazi
"""

import numpy                    as np
import matplotlib.pyplot        as plt
import sklearn.datasets         as datasets
import sklearn.model_selection  as model_selection

def plt_iris_dataset(iris_dataset):
    plt.title('iris')
    plt.scatter(iris_dataset.data[:,0],iris_dataset.data[:,1], c=iris_dataset.target)
    formatter=plt.FuncFormatter(lambda i, *args: iris_dataset.target_names[int(i)])
    plt.colorbar(ticks=[0,1,2], format=formatter)
    plt.tight_layout()
    plt.show()

class KNN:
    """
    K Nearest Neighbors Classifier.
    """
    def __init__(self, X, y, k, classification=True):
        self.X_train = X
        self.y_train = y
        self.k = k
        self.classification=classification

    def _euclidean_distance(self, inst1, inst2):
        ''' calculate the euclidean distance between 2 rows in the dataset '''
        distance= np.linalg.norm(inst1-inst2)
        return distance

    def _get_k_neighbors(self, inst1):
        ''' This function return k most close neighbors of inst1 '''
        distances=[]
        for inst2 in self.X_train:
            if not(np.array_equal(inst2,inst1)):
                distances.append(self._euclidean_distance(inst1, inst2))

        distances=np.asarray(distances)

        # argpartition will sort the array and return their indices in the original array,
        # (it will place the smallest values in the first k places, and will not continue to sort the rest of the values)
        indices=np.argpartition(distances, self.k)

        # return the first k neighbors indices
        k_first_indices = indices[:self.k]
        return k_first_indices

    def predict_class(self, inst1):
        knn_indices=self._get_k_neighbors(inst1)
        knn_labels = self.y_train[knn_indices]

        if self.classification:
            occurrences=np.bincount(knn_labels)
            mode=np.argmax(occurrences)
            return mode
        else: #regression (mean/median)
            pass

    def get_accuracy(self, y_test, predictions):
        correct= y_test == predictions
        acc = ((np.sum(correct) / y_test.shape[0]) * 100.0).round(2)
        return acc

if __name__ == '__main__':
    # -- load dataset --
    iris_dataset = datasets.load_iris()
    X = iris_dataset.data
    y = iris_dataset.target

    # -- plot data --
    plt_iris_dataset(iris_dataset)

    # -- train test split --
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,
                                                                        test_size=0.25,
                                                                        shuffle=True)
    # -- model --
    #try for different values of k:
    K = [1, 5, 100]

    for k in K:
        classifier = KNN(X_train,y_train,k)

        preds=[]
        for inst in X_test:
            preds.append(classifier.predict_class(inst))
        acc = classifier.get_accuracy(y_test, preds)
        print('accuracy for k=', k, ': ', acc, '%')
