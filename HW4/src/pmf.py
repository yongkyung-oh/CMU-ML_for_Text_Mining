import numpy as np


class PMF(object):
    """PMF

    :param object:
    """

    def __init__(self, num_factors, num_users, num_movies):
        """__init__

        :param num_factors:
        :param num_users:
        :param num_movies:
        """
        # note that you should not modify this function
        np.random.seed(11)
        self.U = np.random.normal(size=(num_factors, num_users))
        self.V = np.random.normal(size=(num_factors, num_movies))
        self.num_users = num_users
        self.num_movies = num_movies

    def predict(self, user, movie):
        """predict

        :param user:
        :param movie:
        """
        # note that you should not modify this function
        return (self.U[:, user] * self.V[:, movie]).sum()

    def train(self, users, movies, ratings, alpha, lambda_u, lambda_v,
              batch_size, num_iterations):
        """train

        :param users: np.array of shape [N], type = np.int64
        :param movies: np.array of shape [N], type = np.int64
        :param ratings: np.array of shape [N], type = np.float32
        :param alpha: learning rate
        :param lambda_u:
        :param lambda_v:
        :param batch_size:
        :param num_iterations: how many SGD iterations to run
        """
        # modify this function to implement mini-batch SGD
        # for the i-th training instance,
        # user `users[i]` rates the movie `movies[i]`
        # with a rating `ratings[i]`.

        total_training_cases = users.shape[0]
        for i in range(num_iterations):
            start_idx = (i * batch_size) % total_training_cases
            users_batch = users[start_idx:start_idx + batch_size]
            movies_batch = movies[start_idx:start_idx + batch_size]
            ratings_batch = ratings[start_idx:start_idx + batch_size]
            curr_size = ratings_batch.shape[0]

            # TODO: implement your SGD here!!
            dUV = 2*(np.sum(self.U[:,users_batch] * self.V[:,movies_batch], axis=0) - ratings_batch)
            dUV = dUV.reshape([-1,1]) 
            dU = self.V[:,movies_batch].T * dUV 
            dV = self.U[:,users_batch].T * dUV 
            
            for i, user in enumerate(users_batch): 
                self.U[:,user] = self.U[:,user] - alpha * (dU[i] + lambda_u*self.U[:,user])
            for i, movie in enumerate(movies_batch): 
                self.V[:,movie] = self.V[:,movie] - alpha * (dV[i] + lambda_v*self.V[:,movie])
        return self.U, self.V            
            
            
