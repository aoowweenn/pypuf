"""This module provides an experiment class which learns an instance
of LTFArray with reliability based CMAES learner.
"""
from numpy.random import RandomState
from numpy.linalg import norm
import logging

from pypuf.experiments.experiment.base import Experiment
from pypuf.learner.evolution_strategies.reliability_based_cmaes import ReliabilityBasedCMAES
from pypuf.simulation.arbiter_based.ltfarray import LTFArray, NoisyLTFArray
from pypuf import tools


class ExperimentReliabilityBasedCMAES(Experiment):

    def __init__(self, log_name,
                 seed_i, k, n, noisiness,
                 seed_c, num, reps,
                 seed_m, pop_size, step_size_limit, iteration_limit,
                 ):
        """Initialize an Experiment using the Reliability based CMAES Learner for modeling LTF Arrays
        :param log_name:        Log name, Prefix of the path or name of the experiment log file
        :param seed_i:          PRNG seed used to create LTF array instances
        :param k:               Width, the number of parallel LTFs in the LTF array
        :param n:               Length, the number stages within the LTF array
        :param noisiness:       Noisiness, the relative scale of noise of instance compared to the scale of weights
        :param seed_c:          PRNG seed used to sample challenges
        :param num:             Challenge number, the number of binary inputs for the LTF array
        :param reps:            Repetitions, the frequency of evaluations of every challenge (part of training_set)
        :param seed_m:          PRNG seed used by the CMAES algorithm for sampling solution points
        :param pop_size:        Population size, the number of sampled points of every CMAES iteration
        :param step_size_limit: Step size limit, the minimal size of step size within the CMAES
        :param iteration_limit: Iteration limit, the maximal number of iterations within the CMAES
        """
        super().__init__(
            log_name='%s.0x%x_%i_%i_%i_%i_%i' % (
                log_name,
                seed_i,
                k,
                n,
                num,
                reps,
                pop_size,
            ),
        )
        # PUF instance to learn
        self.seed_instance = seed_i
        self.prng_instance = RandomState(seed=self.seed_instance)
        self.k = k
        self.n = n
        self.noisiness = noisiness
        # Training set
        self.seed_challenges = seed_c
        self.prng_challenges = RandomState(seed=self.seed_instance)
        self.num = num
        self.reps = reps
        # Parameters for CMAES
        self.seed_model = seed_m
        self.prng_model = RandomState(seed=self.seed_model)
        self.pop_size = pop_size
        self.limit_s = step_size_limit
        self.limit_i = iteration_limit
        # Basic objects
        self.instance = None
        self.learner = None
        self.model = None

    def run(self):
        """Initialize the instance, the training set and the learner to then run the Reliability based CMAES
        with the given parameters
        """
        self.instance = NoisyLTFArray(
            weight_array=LTFArray.normal_weights(self.n, self.k, random_instance=self.prng_instance),
            transform=LTFArray.transform_id,
            combiner=LTFArray.combiner_xor,
            sigma_noise=NoisyLTFArray.sigma_noise_from_random_weights(self.n, 1, self.noisiness),
            random_instance=self.prng_instance,
        )
        self.learner = ReliabilityBasedCMAES(
            tools.TrainingSet(self.instance, self.num, self.prng_challenges, self.reps),
            self.k,
            self.n,
            self.pop_size,
            self.limit_s,
            self.limit_i,
            self.seed_model,
        )
        self.model = self.learner.learn()

    def analyze(self):
        """Analyze the learned model"""
        assert self.model is not None
        self.result_logger.info(
            # seed_i    seed_m      n       k       N       reps    noisiness   time    abortions   accuracy    model
            '0x%x\t'    '0x%x\t'    '%i\t'  '%i\t'  '%i\t'  '%i\t'  '%f\t'      '%f\t'  '%i\t'      '%f\t'      '%s',
            self.seed_instance,
            self.seed_model,
            self.n,
            self.k,
            self.num,
            self.reps,
            self.noisiness,
            self.measured_time,
            self.model.abortions,
            1.0 - tools.approx_dist(self.instance, self.model, min(10000, 2 ** self.n), self.prng_challenges),
            ','.join(map(str, self.model.weight_array.flatten() / norm(self.model.weight_array.flatten())))
        )
