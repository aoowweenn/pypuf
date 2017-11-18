"""This module provides an experiment class which learns an instance
of LTFArray with reliability based CMAES learner.
"""
from numpy.random import RandomState
from numpy.linalg import norm

from pypuf.experiments.experiment.base import Experiment
from pypuf.learner.evolution_strategies.reliability_based_cmaes import ReliabilityBasedCMAES
from pypuf.simulation.arbiter_based.ltfarray import LTFArray, NoisyLTFArray
from pypuf import tools


class ExperimentReliabilityBasedCMAES(Experiment):

    def __init__(self, log_name,
                 seed_instance, k, n, transform, combiner,
                 seed_challenges, challenge_num, reps,
                 seed_model, pop_size, step_size_limit, iteration_limit,
                 ):
        """Initialize an Experiment using the Reliability based CMAES Learner for modeling LTF arrays

        :param log_name:        Log name, Prefix of the path or name of the experiment log file
        :param seed_instance:   PRNG seed used to create LTF array instances
        :param k:               Width, the number of parallel LTFs in the LTF array
        :param n:               Length, the number stages within the LTF array
        :param transform:       Transform, the function for modifying challenges
        :param combiner:        Combiner, the function for combining particular LTF arrays
        :param seed_challenges: PRNG seed used to sample challenges
        :param challenge_num:   Challenge number, the number of binary inputs for the LTF array
        :param reps:            Repetitions, the frequency of evaluations of every challenge (part of training_set)
        :param seed_model:      PRNG seed used by the CMAES algorithm for sampling solution points
        :param pop_size:        Population size, the number of sampled points of every CMAES iteration
        :param step_size_limit: Step size limit, the minimal size of step size within the CMAES
        :param iteration_limit: Iteration limit, the maximal number of iterations within the CMAES
        """
        super().__init__(
            log_name='%s.0x%x_%i_%i_%s_%s_0x%x_%i_%i_0x%x_%i_%f_%f' % (
                log_name,
                seed_instance,
                k,
                n,
                transform.__name__,
                combiner.__name__,
                seed_challenges,
                challenge_num,
                reps,
                seed_model,
                pop_size,
                step_size_limit,
                iteration_limit,
            ),
        )
        self.seed_instance = seed_instance
        self.k = k
        self.n = n
        self.transform = transform
        self.combiner = combiner
        self.seed_challenges = seed_challenges
        self.challenge_num = challenge_num
        self.reps = reps
        self.training_set = tools.TrainingSet(instance=self.instance, N=self.challenge_num)
        self.seed_model = seed_model
        self.pop_size = pop_size
        self.limit_s = step_size_limit
        self.limit_i = iteration_limit
        self.prng_instance = RandomState(seed=self.seed_instance)
        self.prng_challenges = RandomState(seed=self.seed_instance)
        self.prng_model = RandomState(seed=self.seed_model)
        self.instance = None
        self.learner = None
        self.model = None

    def run(self):
        """Initialize the instance, the training set and the learner to then run the Reliability based CMAES
        with the given parameters
        """
        self.instance = NoisyLTFArray(
            weight_array=LTFArray.normal_weights(self.n, self.k, random_instance=self.prng_instance),
            transform=self.transform,
            combiner=self.combiner,
            sigma_noise=1,
            random_instance=RandomState(seed=self.seed_instance),
        )
        self.learner = ReliabilityBasedCMAES(
            self.training_set,
            self.k,
            self.n,
            self.pop_size,
            self.limit_s,
            self.limit_i,
            self.seed_model,
        )
        self.model = self.learner.learn()

    def analyze(self):
        """Analyze the learned result"""
        assert self.model is not None

        self.result_logger.info(
            # seed_instance  seed_model i      n      k      N      trans  comb   iter   time   accuracy  model values
            '0x%x\t'        '0x%x\t'   '%i\t' '%i\t' '%i\t' '%i\t' '%s\t' '%s\t' '%i\t' '%f\t' '%f\t'    '%s',
            self.seed_instance,
            self.seed_model,
            0,  # restart count, kept for compatibility to old log files
            self.n,
            self.k,
            self.challenge_num,
            self.transform.__name__,
            self.combiner.__name__,
            self.learner.iteration_count,
            self.measured_time,
            1.0 - tools.approx_dist(self.instance, self.model, min(10000, 2 ** self.n)),
            ','.join(map(str, self.model.weight_array.flatten() / norm(self.model.weight_array.flatten())))

        )
