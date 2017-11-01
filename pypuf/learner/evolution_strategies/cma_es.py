import numpy as np
from scipy import special as sp

class CMA_ES():

    def __init__(self, pop_size, fitness_function, n, limit_step_size, limit_iteration, prng=np.random.RandomState(),
                 abortion_function=None):
        self.prng = prng                                        # pseudo random number generator for random mutations
        self.iterations = 0                                     # number of iterations within the search method
        self.stagnation_cnt = 0                                 # counter for stagnations
        self.limit_step_size = limit_step_size                  # intended scale of step-size to achieve
        self.limit_iteration = limit_iteration                  # maximum number of iterations
        self.check_unwanted = abortion_function                 # comparing function to abort search method
                                                                    # for comparing with previously learned LTFs
        self.termination_cause = None                           # string of the cause of termination
        self.evaluate_fitness = fitness_function                # fitness function for evaluating solution candidates
        self.n = n                                              # number of parameters to learn
        # mean, step_size,  pop_size,   parent_size,    individuals,    weights,    cov_matrix,   path_cm,    path_ss
        # m,    sigma,      lambda,     mu,             x_i,            w_i,        C,            p_c,        p_sigma
        self.mean = np.zeros(self.n)                            # mean vector of distribution
        self.step_size = 1                                      # distance to subsequent mean
        self.pop_size = pop_size                                # number of individuals per generation
        self.parent_size = np.int32(np.floor(self.pop_size/2))  # number of considered individuals
        self.individuals = np.zeros((self.pop_size, n))         # solution candidates
        self.weights = self.get_weights(self.parent_size)       # array of consideration proportions
        self.cov_matrix = np.identity(self.n)                   # shape of distribution ellipsoid
        self.path_cm = np.zeros(self.n)                         # cumulated evolution path of covariance matrix
        self.path_ss = np.zeros(self.n)                         # cumulated evolution path of step size
        # auxiliary constants
        self.mu_w = 1 / np.sum(np.square(self.weights))
        self.c_sigma = (self.mu_w + 2) / (n + self.mu_w + 3)
        self.d_sigma = 1 + 2 * max(0, np.sqrt((self.mu_w - 1) / (n + 1))) + self.c_sigma
        self.c_d_sigma = self.c_sigma / self.d_sigma
        self.c_c = 4 / (n + 4)
        self.c_mu = (1 / self.mu_w) * (2 / (n + np.sqrt(2)) ** 2)\
                    + (1 - 1 / self.mu_w) * min(1, (2 * self.mu_w - 1) / ((n + 2) ** 2 + self.mu_w))
        self.c_1 = self.c_mu / self.mu_w


    def evolutionary_search(self):
        # this is the main CMA-ES algorithm like that from Hansen
        terminate = False
        while not terminate:
            self.iterations += 1
            # abort, if the current mean approximates an unwanted solution; check every 50 iterations
            if self.iterations % 50 == 0 and self.check_unwanted!=None:
                if self.check_unwanted(self.mean):
                    return None
            # terminate, if any termination condition is fulfilled
            if self.step_size <= self.limit_step_size or self.iterations >= self.limit_iteration\
                    or self.stagnation_cnt >= 3:
                if self.step_size <= self.limit_step_size:
                    self.termination_cause = 'step_size'
                elif self.iterations >= self.limit_iteration:
                    self.termination_cause = 'iteration'
                elif self.stagnation_cnt >= 3:
                    self.termination_cause = 'stagnation'
                terminate = True
                break
            self.update_parameters()
        return self.mean

    def update_parameters(self):
        estimation_multinormal = np.sqrt(2) * sp.gamma((self.n + 1) / 2) / sp.gamma((self.n) / 2)
        zero_mean = np.zeros(np.shape(self.mean))
        desc_range = list(range(self.pop_size, 0, -1))
        # updating parameters
        mutations = self.sample_mutations(zero_mean, self.cov_matrix, self.pop_size, self.prng)
        self.individuals = self.reproduce(self.mean, self.pop_size, self.step_size, mutations)
        fitness_values = self.evaluate_fitness(self.individuals)
        count_nan = np.count_nonzero(np.isnan(fitness_values))
        sorting_indices = np.argsort(fitness_values)
        if np.array_equal(desc_range, sorting_indices):  # check, if order of individuals remains unchanged
            self.stagnation_cnt += 1
        else:
            self.stagnation_cnt = 0
        sorted_mutations = np.roll(mutations[sorting_indices[::-1]], -count_nan)
        favorite_mutations = self.get_favorite_mutations(sorted_mutations, self.parent_size, self.weights)
        self.mean = self.update_mean(self.mean, self.step_size, favorite_mutations)
        self.path_cm = self.cumulation_for_cm(self.path_cm, self.c_c, self.path_ss, self.n, self.mu_w,
                                              favorite_mutations)
        self.path_ss = self.cumulation_for_ss(self.path_ss, self.c_sigma, self.mu_w, self.cov_matrix,
                                              favorite_mutations)
        cm_mu = self.get_mutations_outer_product(sorted_mutations, self.parent_size, self.weights)
        self.cov_matrix = self.update_cm(self.cov_matrix, self.c_1, self.c_mu, self.path_cm, cm_mu)
        self.step_size = self.update_ss(self.step_size, self.c_d_sigma, self.path_ss, estimation_multinormal)

    # updating methods of evolution strategies
    @staticmethod
    def sample_mutations(zero_mean, cov_matrix, pop_size, prng):
        # returns mutations for a new generation of individuals as 2D array (corresponds to y_i)
        return prng.multivariate_normal(zero_mean, cov_matrix, pop_size)

    @staticmethod
    def reproduce(mean, pop_size, step_size, mutations):
        # returns a new generation of individuals out of mutations as 2D array (corresponds to x_i)
        duplicated_mean = np.tile(mean, (pop_size, 1))
        return duplicated_mean + (step_size * mutations)

    @staticmethod
    def update_mean(mean, step_size, favorite_mutation):
        # returns mean of a new population as array (corresponds to m)
        return mean + step_size * favorite_mutation

    @staticmethod
    def cumulation_for_cm(path_cm, c_c, path_ss, n, mu_w, favorite_mutation):
        # returns cumulated evolution path of covariance matrix (corresponds to p_c)
        path_cm = path_cm * (1-c_c)
        if(np.linalg.norm(path_ss) < 1.5 * np.sqrt(n)):
            path_cm += (np.sqrt(1 - (1-c_c)**2) * np.sqrt(mu_w) * favorite_mutation)
        return path_cm

    @staticmethod
    def cumulation_for_ss(path_ss, c_sigma, mu_w, cov_matrix, favorite):
        # returns cumulated evolution path of step-size (corresponds to p_sigma)
        cm_eigen_dec = __class__.modify_eigen_decomposition(cov_matrix)
        return (1-c_sigma) * path_ss + np.sqrt(1 - (1-c_sigma)**2) * np.sqrt(mu_w) * cm_eigen_dec @ favorite

    @staticmethod
    def update_cm(cov_matrix, c_1, c_mu, path_cm, cm_mu):
        # returns covariance matrix of a new population (corresponds to C)
        matrix = (1 - c_1 - c_mu) * cov_matrix + c_1 * path_cm[:, np.newaxis] @ path_cm[np.newaxis, :] + c_mu * cm_mu
        matrix = np.triu(matrix) + np.triu(matrix,1).T      # enforce matrix to be symmetric
        return matrix / max(np.diag(matrix))                # normalize matrix: highest diagonal element := 1

    @staticmethod
    def update_ss(step_size, c_d_sigma, path_ss, estimation_multinormal):
        # returns step-size of a new population (corresponds to sigma)
        factor = np.exp((c_d_sigma) * ((np.linalg.norm(path_ss) / estimation_multinormal) - 1))
        return step_size * factor

    # helping methods
    @staticmethod
    def get_favorite_mutations(sorted_mutations, parent_size, priorities):
        # returns the weighted sum of the fittest individuals mutations (corresponds to y_w)
        parent_mutations = np.zeros(np.shape(sorted_mutations)[1])
        for i in range(parent_size):
            parent_mutations = parent_mutations + priorities[i] * sorted_mutations[i, :]
        return parent_mutations

    @staticmethod
    def get_mutations_outer_product(sorted_mutations, parent_size, priorities):
        # returns the weighted sum of the product of the fittest individuals mutations (corresponds to C_mu)
        outer_product = np.zeros((np.shape(sorted_mutations)[1], np.shape(sorted_mutations)[1]))
        for i in range(parent_size):
            outer_product += priorities[i] * sorted_mutations[i, :, np.newaxis] @ sorted_mutations[i, np.newaxis, :]
        return outer_product

    @staticmethod
    def modify_eigen_decomposition(matrix):
        # returns modified eigen-decomposition (B * D^(-1) * B^T) of matrix A=(B * D^2 * B^T) (corresponds to C^(-1/2))
        eigen_values, eigen_vectors = np.linalg.eigh(matrix)
        if not (eigen_values > 0).all():
            indices = eigen_values <= 0
            eigen_values[indices] = 0.001
        diagonal = 1 / np.sqrt(eigen_values)
        return eigen_vectors @ np.diag(diagonal) @ eigen_vectors.T

    @staticmethod
    def is_symmetric(matrix, tol=1e-8):
        # returns true, if the matrix is symmetric
        return np.allclose(matrix, matrix.T, atol=tol)

    @staticmethod
    def get_weights(mu):
        # returns array of consideration proportions with length parent_size (corresponds to w_i)
        weights = np.empty(np.int32(mu))
        for i in range(mu):
            sum = 0
            for j in range(mu):
                sum += (np.log(mu + 1) - np.log(j + 1))
            weights[i] = (np.log(mu + 1) - np.log(i + 1)) / sum
        return weights
