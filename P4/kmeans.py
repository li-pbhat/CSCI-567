import numpy as np


class KMeans():
    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x):
        '''
             Finds n_cluster in the data x
             params:
                 x - N X D numpy array
             returns:
                 A tuple
                 (centroids a n_cluster X D numpy array, y a size (N,) numpy array where cell i is the ith sample's assigned cluster, number_of_updates an Int)
             Note: Number of iterations is the number of time you update the assignment
         '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        np.random.seed(42)
        N, D = x.shape

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership until convergence or until you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE
        mean_indices = np.random.choice(N, self.n_cluster)
        centroids = np.take(x, mean_indices, axis=0)
        y = np.zeros(N)
        J = np.power(10, 10)
        number_of_updates = 0
        while number_of_updates < self.max_iter:
            l2_dist = np.linalg.norm(x - np.expand_dims(centroids, axis=1), axis=2)
            y = np.argmin(l2_dist, axis=0)
            J_new = np.sum([np.sum(np.linalg.norm(x[y==k] - centroids[k]) for k in range(self.n_cluster))]) / N
            if np.absolute(J - J_new) <= self.e:
                break
            J = J_new
            centroids = np.array([np.mean(x[y==k], axis=0) for k in range(self.n_cluster)])
            number_of_updates += 1
        return (centroids, y, number_of_updates)
        # DONOT CHANGE CODE BELOW THIS LINE


class KMeansClassifier():
    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x, y):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting ((N,) numpy array)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        kmeans = KMeans(self.n_cluster, self.max_iter, self.e)
        centroids, membership, number_of_updates = kmeans.fit(x)
        votes = {}
        for i in range(N):
            centroid = membership[i]
            label = y[i]
            if centroid not in votes.keys():
                votes[centroid] = {}
            if label not in votes[centroid].keys():
                votes[centroid][label] = 0
            votes[centroid][label] += 1
        centroid_labels = np.asarray([max(votes[i], key=votes[i].get) for i in range(self.n_cluster)])
        #centroid_labels = centroids[np.argmax([np.sum(membership[])]
        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
        self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(
            self.n_cluster)

        assert self.centroids.shape == (self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(
            self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function

            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        l2_dist = np.linalg.norm(x - np.expand_dims(self.centroids, axis=1), axis=2)
        labels = self.centroid_labels[np.argmin(l2_dist, axis=0)]
        # DONOT CHANGE CODE BELOW THIS LINE
        return labels
