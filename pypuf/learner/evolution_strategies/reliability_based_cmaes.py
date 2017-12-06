"""This module provides a learner exploiting different reliabilities of challenges evaluated several times on an
XOR Arbiter PUF. It is based on the work from G. T. Becker in "The Gap Between Promise and Reality: On the Insecurity
of XOR Arbiter PUFs". The learning algorithm applies Covariance Matrix Adaptation Evolution Strategies from
N. Hansen in "The CMA Evolution Strategy: A Comparing Review".
"""
import sys
import contextlib
import numpy as np
from scipy.special import gamma
from scipy.linalg import norm
import cma

from pypuf.learner.base import Learner
from pypuf.simulation.arbiter_based.ltfarray import LTFArray


class ReliabilityBasedCMAES(Learner):

    def __init__(self, training_set, k, n, transform, combiner,
                 pop_size, limit_stag, limit_iter, random_seed, logger):
        """Initialize a Reliability based CMAES Learner for the specified LTF array

        :param training_set:    Training set, a data structure containing repeated challenge response pairs
        :param k:               Width, the number of parallel LTFs in the LTF array
        :param n:               Length, the number stages within the LTF array
        :param transform:       Transformation function, the function that modifies the input within the LTF array
        :param combiner:        Combiner, the function that combines particular chains' outputs within the LTF array
        :param pop_size:        Population size, the number of sampled points of every CMAES iteration
        :param limit_stag:      Stagnation limit, the maximal number of stagnating iterations within the CMAES
        :param limit_iter:      Iteration limit, the maximal number of iterations within the CMAES
        :param random_seed:     PRNG seed used by the CMAES algorithm for sampling solution points
        :param logger:          Logger, the instance that logs detailed information every learning iteration
        """
        self.training_set = training_set
        self.k = k
        self.n = n
        self.transform = transform
        self.combiner = combiner
        self.challenges = training_set.challenges
        self.responses_rep = training_set.responses
        self.reps = training_set.reps
        self.pop_size = pop_size
        self.limit_s = limit_stag
        self.limit_i = limit_iter
        self.prng = np.random.RandomState(random_seed)
        self.learned_chains = np.zeros((self.k, self.n))
        self.num_iterations = 0
        self.stops = ''
        self.num_abortions = 0
        self.num_learned = 0
        self.logger = logger

    def learn(self):
        """Compute a model according to the given LTF Array parameters and training set
        Note that this function can take long to return
        :return: pypuf.simulation.arbiter_based.LTFArray
                 The computed model.
        """

        def log_state(es):
            """Log a snapshot of learning variables while running"""
            if self.logger is None:
                return
            self.logger.debug(
                '%i\t%f\t%f\t%s\n',
                self.num_iterations,
                es.sigma,
                fitness(es.mean),
                ','.join(map(str, list(es.mean))),
            )

        # Preparation
        epsilon = np.sqrt(self.n) * 0.1
        measured_rels = self.measure_rels(self.responses_rep)
        fitness = self.create_fitness_function(self.challenges, measured_rels, epsilon, self.transform, self.combiner)
        normalize = np.sqrt(2) * gamma(self.n / 2) / gamma((self.n - 1) / 2)
        mean_start = np.zeros(self.n)
        step_size_start = 1
        options = {
            'seed': 0,
            'pop': self.pop_size,
            'maxiter': self.limit_i,
            'tolstagnation': self.limit_s,
        }

        # Learn all individual LTF arrays (chains)
        with self.avoid_printing():
            while self.num_learned < self.k:
                options['seed'] = self.prng.randint(2 ** 32)
                is_same_solution = self.create_abortion_function(
                    self.learned_chains, self.num_learned, self.challenges, self.transform, self.combiner
                )
                es = cma.CMAEvolutionStrategy(x0=mean_start, sigma0=step_size_start, inopts=options)
                counter = 0
                # Learn particular LTF array using abortion if evolutionary search approximates previous solution
                while not es.stop():
                    if counter % 50 == 0 and is_same_solution is not None:
                        if is_same_solution(es.mean):
                            self.num_abortions += 1
                            break
                    curr_points = es.ask()
                    es.tell(curr_points, [fitness(point) for point in curr_points])
                    self.num_iterations += 1
                    log_state(es)
                    counter += 1
                solution = es.result.xbest
                self.num_iterations += es.result.iterations

                # Include normalized new LTF, if it is different from previous ones
                if not is_same_solution(solution):
                    self.learned_chains[self.num_learned] = normalize * solution / norm(solution)
                    self.num_learned += 1
                    if self.stops != '':
                        self.stops += ','
                    self.stops += '_'.join(list(es.stop()))

        # Polarize the learned combined LTF
        majority_responses = self.majority_responses(self.responses_rep)
        self.learned_chains = self.polarize_ltfs(self.learned_chains, self.challenges, majority_responses,
                                                 self.transform, self.combiner)
        return LTFArray(self.learned_chains, self.transform, self.combiner)

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

    @staticmethod
    @contextlib.contextmanager
    def avoid_printing():
        save_stdout = sys.stdout
        sys.stdout = open('/dev/null', 'w')
        yield
        sys.stdout = save_stdout

    @staticmethod
    def create_fitness_function(challenges, measured_rels, epsilon, transform, combiner):
        """Return a fitness function on a fixed set of challenges and corresponding reliabilities"""
        this = __class__

        def fitness(individual):
            """Return individuals sorted by their correlation coefficient as fitness"""
            ltf_array = LTFArray(individual[np.newaxis, :], transform, combiner)
            delay_diffs = ltf_array.val(challenges)
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
    def create_abortion_function(learned_ltfs, num_learned, challenges, transform, combiner):
        """Return an abortion function on a fixed set of challenges and LTFs"""
        this = __class__
        weight_arrays = learned_ltfs[:num_learned, :]
        learned_ltf_arrays = this.build_ltf_arrays(weight_arrays, transform, combiner)
        responses_learned_ltfs = np.zeros((num_learned, np.shape(challenges)[0]))
        for i, current_ltf in enumerate(learned_ltf_arrays):
            responses_learned_ltfs[i, :] = current_ltf.eval(challenges)

        def same_solution(solution):
            """Return True, if the current solution mean within CMAES is similar to a previously learned LTF array"""
            if num_learned == 0:
                return False
            new_ltf_array = LTFArray(solution[np.newaxis, :], transform, combiner)
            responses_new_ltf = new_ltf_array.eval(challenges)
            return this.is_correlated(responses_new_ltf, responses_learned_ltfs)

        return same_solution

    @staticmethod
    def polarize_ltfs(learned_ltfs, challenges, majority_responses, transform, combiner):
        """Return the correctly polarized combined LTF array"""
        model = LTFArray(learned_ltfs, transform, combiner)
        responses_model = model.eval(challenges)
        challenge_num = np.shape(challenges)[0]
        accuracy = np.count_nonzero(responses_model == majority_responses) / challenge_num
        polarized_ltfs = learned_ltfs
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
    def is_correlated(responses_new_ltf, responses_learned_ltfs):
        """Return True, if 2 response arrays are more than 75% equal (Hamming distance)"""
        num_of_ltfs, challenge_num = np.shape(responses_learned_ltfs)
        for i in range(num_of_ltfs):
            differences = np.sum(np.abs(responses_new_ltf[:] - responses_learned_ltfs[i, :])) / 2
            if differences < 0.25*challenge_num or differences > 0.75*challenge_num:
                return True
        return False

    @staticmethod
    def majority_responses(responses):
        """Return the common responses out of repeated responses"""
        return np.sign(np.sum(responses, axis=0))

    @staticmethod
    def measure_rels(responses):
        """Return array of measured reliabilities of instance"""
        measured_rels = np.abs(np.sum(responses, axis=0))
        if np.var(measured_rels) == 0:
            raise Exception('The challenges\' reliabilities evaluated on the instance to learn are to high!')
        return measured_rels