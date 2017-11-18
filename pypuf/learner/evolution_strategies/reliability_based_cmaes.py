"""This module provides a learner exploiting different reliabilities of challenges evaluated several times on an
Arbiter PUF. It is based on the work from G. T. Becker in "The Gap Between Promise and Reality: On the Insecurity
of XOR Arbiter PUFs". The learning algorithm applies Covariance Matrix Adaptation Evolution Strategies from
N. Hansen in "The CMA Evolution Strategy: A Comparing Review".
"""
import numpy as np
from scipy.special import gamma
from scipy.linalg import norm
import cma

from pypuf.learner.base import Learner
from pypuf.simulation.arbiter_based.ltfarray import LTFArray


class ReliabilityBasedCMAES(Learner):

    def __init__(self, training_set, k, n, pop_size, step_size_limit, iteration_limit, seed_model):
        """Initialize a Reliability based CMAES Learner for the specified LTF array

        :param training_set:    Training set, a data structure containing repeated challenge response pairs
        :param k:               Width, the number of parallel LTFs in the LTF array
        :param n:               Length, the number stages within the LTF array
        :param pop_size:        Population size, the number of sampled points of every CMAES iteration
        :param step_size_limit: Step size limit, the minimal size of step size within the CMAES
        :param iteration_limit: Iteration limit, the maximal number of iterations within the CMAES
        :param seed_model:      PRNG seed used by the CMAES algorithm for sampling solution points
        """
        self.__training_set = training_set
        self.k = k
        self.n = n
        self.transform = LTFArray.transform_id
        self.combiner = LTFArray.combiner_xor
        self.challenges = training_set.challenges
        self.responses_rep = training_set.responses
        self.reps = training_set.reps
        self.different_LTFs = np.zeros((self.k, self.n))
        self.num_of_LTFs = 0
        self.pop_size = pop_size
        self.limit_s = step_size_limit
        self.limit_i = iteration_limit
        self.seed_model = seed_model

    @property
    def training_set(self):
        """Return the training set which is used to learn a PUF instance
        :return: pypuf.tools.TrainingSet
        """
        return self.__training_set

    @training_set.setter
    def training_set(self, val):
        """Set the training set which is used to learn a PUF instance
        :param val: pypuf.tools.TrainingSet
        """
        self.__training_set = val

    def learn(self):
        """Compute a model according to the given LTF Array parameters and training set
        Note that this function can take long to return
        :return: pypuf.simulation.arbiter_based.LTFArray
                 The computed model.
        """

        # Preparation
        epsilon = np.sqrt(self.n) * 0.1
        measured_rels = self.measure_rels(self.responses_rep)
        fitness = self.create_fitness_function(self.challenges, measured_rels, epsilon, self.transform, self.combiner)
        normalize = np.sqrt(2) * gamma(self.n / 2) / gamma((self.n - 1) / 2)

        # Learn new particular LTF
        options = {'seed': self.seed_model, 'pop': self.pop_size}
        res = cma.fmin(fitness, np.zeros(self.n), 1, options)
        solution = res[0]

        # Include normalized new LTF, if it is different from previous ones
        if self.is_different_ltf(solution, self.different_LTFs, self.num_of_LTFs, self.challenges,
                                 self.transform, self.combiner):
            self.different_LTFs[self.num_of_LTFs] = solution * normalize / norm(solution)  # normalize weights
            self.num_of_LTFs += 1

        # Polarize the learned combined LTF
        common_responses = self.common_responses(self.responses_rep)
        self.different_LTFs = self.polarize_ltfs(self.different_LTFs, self.challenges, common_responses,
                                                 self.transform, self.combiner)
        return LTFArray(self.different_LTFs, self.transform, self.combiner)

    @staticmethod
    def create_fitness_function(challenges, measured_rels, epsilon, transform, combiner):
        """Return a fitness function on a fixed set of challenges and corresponding reliabilities"""
        this = __class__

        def fitness(individual):
            """Return individuals sorted by their correlation coefficient as fitness"""
            built_ltf_array = LTFArray(individual[np.newaxis, :], transform, combiner)
            delay_diffs = built_ltf_array.val(challenges)
            reliabilities = np.zeros(np.shape(delay_diffs))
            indices_of_reliable = np.abs(delay_diffs[:]) > epsilon
            reliabilities[indices_of_reliable] = 1
            correlation = this.calc_corr(reliabilities, measured_rels)
            obj_vals = 1 - (1 + correlation)/2
            return obj_vals

        return fitness

    @staticmethod
    def calc_corr(reliabilities, measured_rels):
        """Return pearson correlation coefficient between reliability arrays of individual and instance"""
        if np.var(reliabilities[:]) == 0:  # Avoid divide by zero
            return -1
        else:
            return np.corrcoef(reliabilities[:], measured_rels)[0, 1]

    @staticmethod
    def create_abortion_function(different_ltfs, num_of_ltfs, challenges, transform, combiner):
        """Return an abortion function on a fixed set of challenges and LTFs
        ###Currently not used###
        """
        this = __class__
        weight_arrays = different_ltfs[:num_of_ltfs, :]
        different_ltf_arrays = this.build_ltf_arrays(weight_arrays, transform, combiner)
        responses_diff_ltfs = np.zeros((num_of_ltfs, np.shape(challenges)[0]))
        for i, current_ltf in enumerate(different_ltf_arrays):
            responses_diff_ltfs[i, :] = current_ltf.eval(challenges)

        def abortion(new_ltf):
            """Return True, if the current solution mean within CMAES is similar to a previously learned LTF array"""
            if num_of_ltfs == 0:
                return False
            new_ltf_array = LTFArray(new_ltf[np.newaxis, :], transform, combiner)
            responses_new_ltf = new_ltf_array.eval(challenges)
            return this.is_correlated(responses_new_ltf, responses_diff_ltfs)

        return abortion

    @staticmethod
    def is_different_ltf(new_ltf, different_ltfs, num_of_ltfs, challenges, transform, combiner):
        """Return True, if new LTF is different from previously learned LTFs"""
        if num_of_ltfs == 0:
            return True
        weight_arrays = different_ltfs[:num_of_ltfs, :]
        new_ltf_array = LTFArray(new_ltf[np.newaxis, :], transform, combiner)
        different_ltf_arrays = __class__.build_ltf_arrays(weight_arrays, transform, combiner)
        responses_new_ltf = new_ltf_array.eval(challenges)
        responses_diff_ltfs = np.zeros((num_of_ltfs, np.shape(challenges)[0]))
        for i, current_ltf in enumerate(different_ltf_arrays):
            responses_diff_ltfs[i, :] = current_ltf.eval(challenges)
        return not __class__.is_correlated(responses_new_ltf, responses_diff_ltfs)

    @staticmethod
    def polarize_ltfs(different_ltfs, challenges, common_responses, transform, combiner):
        """Return the correctly polarized XOR LTF array"""
        model = LTFArray(different_ltfs, transform, combiner)
        responses_model = model.eval(challenges)
        challenge_num = np.shape(challenges)[0]
        accuracy = np.count_nonzero(responses_model == common_responses) / challenge_num
        polarized_ltfs = different_ltfs
        if accuracy < 0.5:
            polarized_ltfs[0, :] *= -1
        return polarized_ltfs

    @staticmethod
    def build_ltf_arrays(weight_arrays, transform, combiner):
        """Return iterator over LTF arrays created out of every individual"""
        pop_size = np.shape(weight_arrays)[0]
        for i in range(pop_size):
            yield LTFArray(weight_arrays[i, np.newaxis, :], transform, combiner)

    @staticmethod
    def is_correlated(responses_new_ltf, responses_diff_ltfs):
        """Return True, if 2 response arrays are more than 75% equal (Hamming distance)"""
        num_of_ltfs, challenge_num = np.shape(responses_diff_ltfs)
        for i in range(num_of_ltfs):
            differences = np.sum(np.abs(responses_new_ltf[:] - responses_diff_ltfs[i, :])) / 2
            if differences < 0.25*challenge_num or differences > 0.75*challenge_num:
                return True
        return False

    @staticmethod
    def common_responses(responses):
        """Return the common responses out of repeated responses"""
        return np.sign(np.sum(responses, axis=0))

    @staticmethod
    def measure_rels(responses):
        """Return array of measured reliabilities of instance"""
        measured_rels = np.abs(np.sum(responses, axis=0))
        if np.var(measured_rels) == 0:
            raise Exception('The challenges\' reliabilities evaluated on the instance to learn are to high!')
        return measured_rels
