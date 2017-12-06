"""This module tests the reliability based CMAES learner."""
import unittest
import numpy as np

from pypuf.simulation.arbiter_based.ltfarray import LTFArray, NoisyLTFArray
from pypuf.learner.evolution_strategies.reliability_based_cmaes import ReliabilityBasedCMAES
from pypuf import tools


class TestReliabilityBasedCMAES(unittest.TestCase):
    """This class contains tests for the methods of the reliability based CMAES learner."""
    n = 16
    k = 2
    N = 2**12
    reps = 4
    mu_weight = 0
    sigma_weight = 1
    transform = LTFArray.transform_id
    combiner = LTFArray.combiner_xor
    seed_instance = 1234
    prng_i = np.random.RandomState(seed_instance)
    seed_model = 1234
    prng_m = np.random.RandomState(seed_model)
    seed_challenges = 1234
    prng_c = np.random.RandomState(seed_challenges)

    weight_array = LTFArray.normal_weights(n, k, mu_weight, sigma_weight, prng_i)

    @unittest.skip
    def test_learn(self):
        sigma_noise = NoisyLTFArray.sigma_noise_from_random_weights(self.n, self.sigma_weight, noisiness=0.05)
        instance = NoisyLTFArray(self.weight_array, self.transform, self.combiner, sigma_noise, self.prng_i)
        training_set = tools.TrainingSet(instance, self.N, self.prng_c, self.reps)
        pop_size = 16
        limit_stag = 100
        limit_iter = 1000
        logger = None
        learner = ReliabilityBasedCMAES(training_set, self.k, self.n, self.transform, self.combiner,
                                        pop_size, limit_stag, limit_iter, self.seed_model, logger)
        model = learner.learn()
        distance = tools.approx_dist(instance, model, 100000)
        assert distance < 0.4

    def test_calc_corr(self):
        rels_1 = np.array([0, 1, 2, 1])
        rels_2 = np.array([0, 0, 0, 1])
        rels_3 = np.array([0, 1, 2, 5])
        rels_4 = np.array([1, 1, 1, 1])
        corr_1_2 = ReliabilityBasedCMAES.calc_corr(rels_1, rels_2)
        corr_1_3 = ReliabilityBasedCMAES.calc_corr(rels_1, rels_3)
        corr_2_3 = ReliabilityBasedCMAES.calc_corr(rels_2, rels_3)
        corr_4_1 = ReliabilityBasedCMAES.calc_corr(rels_4, rels_1)
        assert corr_1_2 < corr_1_3 < corr_2_3
        self.assertEqual(corr_4_1, -1)

    @unittest.skip
    def test_polarize_ltfs(self):
        learned_ltfs = np.array([
            [.5, -1, -.5, 1],
            [-1, -1, 1, 1],
        ])
        challenges = tools.sample_inputs(n=4, num=8, random_instance=self.prng_c)
        majority_responses = np.array([1, 1, 1, 1, -1, -1, -1, -1])
        polarized_ltf_array = ReliabilityBasedCMAES.polarize_ltfs(
            learned_ltfs, challenges, majority_responses, self.transform, self.combiner
        )
        self.assertIsNotNone(polarized_ltf_array)

    @unittest.skip
    def test_build_ltf_arrays(self):
        challenges = tools.sample_inputs(self.n, self.N)
        ltf_array_original = LTFArray(self.weight_array, self.transform, self.combiner)
        res_original = ltf_array_original.eval(challenges)
        weight_arrays = self.weight_array[np.newaxis, :].repeat(2, axis=0)
        ltf_arrays = ReliabilityBasedCMAES.build_ltf_arrays(weight_arrays, self.transform, self.combiner)
        for ltf_array in ltf_arrays:
            res = ltf_array.eval(challenges)
            np.testing.assert_array_equal(res, res_original)

    def test_is_correlated(self):
        res_1 = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        res_2 = np.array([[1, 1, 1, 1, 1, 1, 1, -1],
                          [1, 1, 1, 1, -1, -1, -1, -1]])
        res_3 = np.array([[1, 1, 1, 1, 1, 1, -1, -1],
                          [1, 1, 1, 1, -1, -1, -1, -1]])
        corr_1_2 = ReliabilityBasedCMAES.is_correlated(res_1, res_2)
        corr_1_3 = ReliabilityBasedCMAES.is_correlated(res_1, res_3)
        self.assertTrue(corr_1_2)
        self.assertFalse(corr_1_3)

    responses = np.array([
        [1, 1, 1, 1],
        [1, 1, 1, -1],
        [1, 1, -1, -1],
        [1, -1, -1, -1]
    ])

    def test_common_responses(self):
        common_res = ReliabilityBasedCMAES.majority_responses(self.responses)
        self.assertEqual(common_res.all(), np.array([1, 1, 0, -1]).all())

    def test_measure_rels(self):
        rels = ReliabilityBasedCMAES.measure_rels(self.responses)
        self.assertEqual(rels.all(), np.array([4, 2, 0, 2]).all())
