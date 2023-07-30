"""
@author: Michal Ashkenazi
"""
import random
import numpy             as np
import matplotlib.pyplot as plt
from   sklearn           import datasets
from   sklearn.metrics   import silhouette_score

class KMeansClustering:
    def __init__(self, k, epsilon=1e-9, max_iter=2000):
        self.k         = k
        self.epsilon   = epsilon
        self.max_iter  = max_iter
        self.centroids = []
        self.clusters  = []
    
    def cluster_points(self, X):
        """
        Assign each data point to a cluster by measuring the distance to each centroid.
        """
        # create empty list for the new clusters
        self.clusters = []
        for i in range(self.k):
            self.clusters.append([])
        
        # assisn each instance to its closest cluster
        for instance in X:
            
            # calculate euclidean distance from every cluster
            distances = [np.linalg.norm(instance - self.centroids[i], axis=0) for i in range(self.k)]
            
            # take the closest cluster
            closest_cluster_idx = np.argmin(distances)
            self.clusters[closest_cluster_idx].append(instance)
            
    def recalc_centroids(self):
        """
        recalculate centroids to be the mean of all its attached data
        """
        for i in range(self.k):
            self.centroids[i] = np.mean(self.clusters[i], axis=0).tolist()
            
    def is_converged(self, previous_centroids):
        """
        check model convergence.
        """
        all_distances = np.linalg.norm(np.array(previous_centroids) - np.array(self.centroids), axis=1)
        max_diss      = all_distances.max()
        return max_diss < self.epsilon
            
    def fit(self, X):
        """
        Train k-means model
        """
        # randomly choose k centroids from the data:
        self.centroids = random.sample(list(X), self.k)
        
        # repeat until convergance:
        for i in range(self.max_iter):
            
            # store current centroids:
            previous_centroids = self.centroids.copy()
            
            # cluster samples:
            self.cluster_points(X)
            
            # recalculate centroids:
            self.recalc_centroids()
            
            # check model convergence:
            if self.is_converged(previous_centroids):
                return self.clusters
            
        return self.clusters
            

if __name__ == "__main__":     
    # load data:
    X, y = datasets.load_iris(return_X_y=True)

    # plot original data with labels:
    fig  = plt.figure()
    ax   = fig.add_subplot(111, projection="3d")
    img  = ax.scatter(X[:, 2], X[:, 1], X[:, 0], c=y)
    fig.colorbar(img)
    plt.show()

    # train:
    k             = 3
    kmeans_clustr = KMeansClustering(k=k)
    clusters      = kmeans_clustr.fit(X)


    # plot clusters
    fig          = plt.figure()
    labeled_data = []
            
    for i in range(k):
        row = clusters[i]
                
        for j in range(len(row)):
            row[j] = np.append(row[j], i).tolist() # giving each instance a lable according to its cluster
            labeled_data.append(row[j])
                
    labeled_data = np.asarray(labeled_data)
            
    ax = fig.add_subplot(111, projection='3d')
            
    X_tmp = labeled_data[:, 0:3]
    y_predicted = labeled_data[:, -1]
            
    img = ax.scatter(X_tmp[:, 2], X_tmp[:, 1], X_tmp[:, 0], c=y_predicted)
    fig.colorbar(img)
    plt.show()


    # checking accuracy:

    # The best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters.
    # Negative values generally indicate that a sample has been assigned to the wrong cluster, 
    # as a different cluster is more similar.
    silhouette_score(X, y_predicted, metric='euclidean')
