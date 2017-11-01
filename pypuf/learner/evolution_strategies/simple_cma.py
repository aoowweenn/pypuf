import numpy as np
from scipy.special import gamma
from scipy.linalg import norm
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
import cma

class Simple_CMA_ES():
    """
        This class provides a learner based on Evolution Strategies, which automatically generates a model with similar
        behavior as an LTFArray, whose behavior was evaluated by a set of repeated Challenge-Response-Pairs.
        Thus, this class corresponds to the side-channel modeling attack of Becker.
        The blueprint of the LTFArray and the CRPs are defined in the constructor, as well as termination criteria for
        the utilized evolution strategies algorithm using covariance matrix adaptation (see Hansen et. al.) and a
        pseudo random number generator.
    """

    def __init__(self, k, n, transform, combiner, challenges, responses_repeated, repetitions, step_size_limit,
                 iteration_limit, prng=np.random.RandomState()):
        self.k = k                                          # number of XORed LTFs
        self.n = n                                          # length of LTFs
        self.transform = transform                          # function for modifying challenges
        self.combiner = combiner                            # function for combining the particular LTFArrays
        self.challenges = challenges                        # set of challenges applied on instance to learn
        self.responses_repeated = responses_repeated        # responses of repeatedly evaluated challenges on instance
        self.repetitions = repetitions                      # number of repetitions of all challenges
        self.different_LTFs = np.zeros((self.k, self.n))    # all currently learned LTFs
        self.num_of_LTFs = 0                                # number of different learned LTFs
        # cmaes
        self.pop_size = 24

    def learn(self):
        # this is the general learning method
        # returns an XOR-LTFArray with nearly the same behavior as learned instance
        epsilon = np.sqrt(self.n) * 0.1
        measured_rels = self.get_measured_rels(self.responses_repeated)
        if np.var(measured_rels) == 0:
            raise Exception('The reliabilities of the responses from the instance to learn are to high!')
        fitness_function = self.get_fitness_function(self.challenges, measured_rels, epsilon, self.transform,
                                                     self.combiner)
        normalize = np.sqrt(2) * gamma((self.n) / 2) / gamma((self.n - 1) / 2)
        # learn new particular LTF
        options = {'CMA_diagonal': 100, 'seed': 1234, 'verb_time': 0, 'pop': self.pop_size}
        res = cma.fmin(fitness_function, np.zeros(self.n), 1, options)
        solution = res[0]
        # include normalized new_LTF, if it is different from previous ones
        if self.is_different_LTF(solution, self.different_LTFs, self.num_of_LTFs, self.challenges,
                                 self.transform, self.combiner):
            self.different_LTFs[self.num_of_LTFs] = solution * normalize / norm(solution)  # normalize weights
            self.num_of_LTFs += 1
        # polarize the learned combined LTF
        common_responses = self.get_common_responses(self.responses_repeated)
        self.different_LTFs = self.set_pole_of_LTFs(self.different_LTFs, self.challenges, common_responses,
                                                    self.transform, self.combiner)
        return LTFArray(self.different_LTFs, self.transform, self.combiner, bias=False)

    @staticmethod
    def get_fitness_function(challenges, measured_rels, epsilon, transform, combiner):
        # returns a fitness function on a fixed set of challenges and corresponding reliabilities
        becker = __class__

        def fitness(individual):
            # returns individuals sorted by their correlation coefficient as fitness
            built_LTFArray = LTFArray(individual[np.newaxis, :], transform, combiner, bias=False)
            delay_diffs = built_LTFArray.val(challenges)
            reliabilities = np.zeros(np.shape(delay_diffs))
            indices_of_reliable = np.abs(delay_diffs[:]) > epsilon
            reliabilities[indices_of_reliable] = 1
            correlation = becker.get_correlation(reliabilities, measured_rels)
            obj_vals = 1 - (1 + correlation)/2
            return obj_vals

        return fitness

    @staticmethod
    def get_correlation(reliabilities, measured_rels):
        if np.var(reliabilities[:]) == 0:  # avoid divide by zero
            return -1
        else:
            return np.corrcoef(reliabilities[:], measured_rels)[0, 1]

    @staticmethod
    def get_abortion_function(different_LTFs, num_of_LTFs, challenges, transform, combiner):
        # returns an abortion function on a fixed set of challenges and LTFs
        becker = __class__
        weight_arrays = different_LTFs[:num_of_LTFs, :]
        different_LTFArrays = becker.build_LTFArrays(weight_arrays, transform, combiner)
        responses_diff_LTFs = np.zeros((num_of_LTFs, np.shape(challenges)[0]))
        for i, current_LTF in enumerate(different_LTFArrays):
            responses_diff_LTFs[i, :] = current_LTF.eval(challenges)

        def abortion_function(new_LTF):
            if num_of_LTFs == 0:
                return False
            new_LTFArray = LTFArray(new_LTF[np.newaxis, :], transform, combiner)
            responses_new_LTF = new_LTFArray.eval(challenges)
            return becker.is_correlated(responses_new_LTF, responses_diff_LTFs)

        return abortion_function

    @staticmethod
    def is_different_LTF(new_LTF, different_LTFs, num_of_LTFs, challenges, transform, combiner):
        # returns True, if new_LTF is different from previously learned LTFs
        if num_of_LTFs == 0:
            return True
        weight_arrays = different_LTFs[:num_of_LTFs, :]
        new_LTFArray = LTFArray(new_LTF[np.newaxis, :], transform, combiner)
        different_LTFArrays = __class__.build_LTFArrays(weight_arrays, transform, combiner)
        responses_new_LTF = new_LTFArray.eval(challenges)
        responses_diff_LTFs = np.zeros((num_of_LTFs, np.shape(challenges)[0]))
        for i, current_LTF in enumerate(different_LTFArrays):
            responses_diff_LTFs[i, :] = current_LTF.eval(challenges)
        return not __class__.is_correlated(responses_new_LTF, responses_diff_LTFs)

    @staticmethod
    def set_pole_of_LTFs(different_LTFs, challenges, common_responses, transform, combiner):
        # returns the correctly polarized XOR-LTFArray
        model = LTFArray(different_LTFs, transform, combiner)
        responses_model = model.eval(challenges)
        challenge_num = np.shape(challenges)[0]
        accuracy = np.count_nonzero(responses_model == common_responses) / challenge_num
        polarized_LTFs = different_LTFs
        if accuracy < 0.5:
            polarized_LTFs[0, :] *= -1
        return polarized_LTFs


    # methods for calculating fitness
    @staticmethod
    def build_LTFArrays(weight_arrays, transform, combiner):
        # returns iterator over ltf_arrays created out of every individual
        pop_size = np.shape(weight_arrays)[0]
        for i in range(pop_size):
            yield LTFArray(weight_arrays[i, np.newaxis, :], transform, combiner, bias=False)

    @staticmethod
    def get_delay_differences(built_LTFArrays, pop_size, challenges):
        # returns 2D array of delay differences for all challenges on every individual
        delay_diffs = np.empty((pop_size, np.shape(challenges)[0]))
        for i, built_LTFArray in enumerate(built_LTFArrays):
            delay_diffs[i, :] = built_LTFArray.val(challenges)
        return delay_diffs

    @staticmethod
    def get_reliabilities(delay_diffs, epsilon):
        # returns 2D array of reliabilities for all challenges on every individual
        reliabilities = np.zeros(np.shape(delay_diffs))
        for i in range(np.shape(reliabilities)[0]):
            indices_of_reliable = np.abs(delay_diffs[i, :]) > epsilon
            reliabilities[i, indices_of_reliable] = 1
        return reliabilities

    @staticmethod
    def get_correlations(reliabilities, measured_rels):
        # returns array of pearson correlation coefficients between reliability array of individual
        #   and instance for all individuals
        pop_size = np.shape(reliabilities)[0]
        correlations = np.zeros(pop_size)
        for i in range(pop_size):
            if np.var(reliabilities[i, :]) == 0:    # avoid divide by zero
                correlations[i] = -1
            else:
                correlations[i] = np.corrcoef(reliabilities[i, :], measured_rels)[0, 1]
        return correlations


    # helping methods
    @staticmethod
    def is_correlated(responses_new_LTF, responses_diff_LTFs):
        # returns True, if 2 response arrays are more than 75% equal (Hamming distance)
        num_of_LTFs, challenge_num = np.shape(responses_diff_LTFs)
        for i in range(num_of_LTFs):
            differences = np.sum(np.abs(responses_new_LTF[:] - responses_diff_LTFs[i, :])) / 2
            if differences < 0.25*challenge_num or differences > 0.75*challenge_num:
                return True
        return False

    @staticmethod
    def get_common_responses(responses):
        # returns the common responses out of repeated responses
        return np.sign(np.sum(responses, axis=0))

    @staticmethod
    def get_measured_rels(responses):
        # returns array of measured reliabilities of instance
        return np.abs(np.sum(responses, axis=0))
