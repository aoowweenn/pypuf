"""This module tests the different functions which can be used to determine simulation properties."""
import unittest
from numpy import array, ones, zeros, mean
from numpy.random import RandomState
from pypuf.simulation.arbiter_based.ltfarray import LTFArray, NoisyLTFArray
from pypuf.tools import sample_inputs
from pypuf.property_test.base import PropertyTest


class TestPropertyTest(unittest.TestCase):
    """This class tests the property testing class."""

    def test_evaluate(self):
        """This method tests the evaluation of instances via evaluate function."""
        n = 8
        k = 1
        N = 2 ** n
        instance_count = 2
        transformation = LTFArray.transform_id
        combiner = LTFArray.combiner_xor
        instances = [
            LTFArray(weight_array=LTFArray.normal_weights(n=n, k=k, random_instance=RandomState(0xA1A1 + i)),
                     transform=transformation, combiner=combiner) for i in range(instance_count)
        ]
        challenges = array(list(sample_inputs(n, N, random_instance=RandomState(0xFAB10))))
        property_test = PropertyTest(instances)
        self.assertIsNone(property_test.evaluated_challenges,
                          'Member variable evaluated_challenges should be set in test().')
        property_test.evaluate(challenges)
        self.assertIsNotNone(property_test.evaluated_challenges,
                             'Member variable evaluated_challenges was not set in test().')

    def test_inter_distance(self):
        """This method tests the inter distance calculation for several responses."""
        n = 8
        # The responses differ in all indices
        responses = array([ones(n), zeros(n)])
        distance = PropertyTest.inter_distance(0, responses)
        self.assertEqual(distance, 1.0)

        # The responses differ in half of the responses
        responses = array([ones(n), array([0, 0, 0, 0, 1, 1, 1, 1])])
        distance = PropertyTest.inter_distance(0, responses)
        self.assertEqual(distance, 0.5)

        # The mean of [0.5, 0.75] should be 0.625
        responses = array([ones(n), array([0, 0, 0, 0, 1, 1, 1, 1]), array([0, 0, 0, 0, 0, 0, 1, 1])])
        distance = PropertyTest.inter_distance(0, responses)
        self.assertEqual(distance, 0.625)

        # The distance must be different for different indices
        distance2 = PropertyTest.inter_distance(1, responses)
        distance3 = PropertyTest.inter_distance(2, responses)

        self.assertNotEqual(distance, distance2)
        self.assertNotEqual(distance, distance3)
        self.assertNotEqual(distance2, distance3)

    def test_inter_distance_leuven(self):
        """This method tests the inter distance calculation from leuven."""
        n = 8

        responses = array([ones(n), zeros(n)])
        distance = PropertyTest.inter_distance_leuven(0, 1, responses)
        self.assertEqual(distance, 1.0)

        # The responses differ in half of the responses
        responses = array([ones(n), array([0, 0, 0, 0, 1, 1, 1, 1])])
        distance = PropertyTest.inter_distance_leuven(0, 1, responses)
        self.assertEqual(distance, 0.5)

    def test_intra_distance(self):
        """This method tests the intra distance calculation."""
        n = 8
        k = 1
        N = 2 ** n
        transformation = LTFArray.transform_id
        combiner = LTFArray.combiner_xor
        instance = LTFArray(
            weight_array=LTFArray.normal_weights(n=n, k=k, random_instance=RandomState(0xA1A1)),
            transform=transformation,
            combiner=combiner,
        )
        challenges = array(list(sample_inputs(n, N, random_instance=RandomState(0xFAB1A))))
        distance = PropertyTest.intra_distance(instance, challenges)
        # For noiseless simulations the responses are always the same
        self.assertEqual(distance, 0.0)

        noisy_instance = NoisyLTFArray(
            weight_array=NoisyLTFArray.normal_weights(n=n, k=k, random_instance=RandomState(0xA1A1)),
            transform=transformation,
            combiner=combiner,
            sigma_noise=0.5,
            random_instance=RandomState(0x5015E),
        )
        noisy_distance = PropertyTest.intra_distance(noisy_instance, challenges)
        # For noisy simulations the responses should vary
        self.assertNotEqual(noisy_distance, 0.0)

    def test_intra_distance_leuven(self):
        """This method tests the intra distance calculation from leuven."""
        n = 8
        k = 1
        N = 2 ** n
        measurements = 10
        transformation = LTFArray.transform_id
        combiner = LTFArray.combiner_xor

        instance = LTFArray(
            weight_array=LTFArray.normal_weights(n=n, k=k, random_instance=RandomState(0xA1A1)),
            transform=transformation,
            combiner=combiner,
        )

        challenges = array(list(sample_inputs(n, N, random_instance=RandomState(0xFAB0))))

        distances = PropertyTest.intra_distance_leuven(instance, challenges, measurements=measurements)
        # The result is an array like with 1/2*(measurements-1)*(measurements) entries.
        self.assertEqual(len(distances), 1/2*(measurements-1)*(measurements))
        # For noiseless simulations the mean distance must be zero
        self.assertEqual(mean(distances), 0.0)

        noisy_instance = NoisyLTFArray(
            weight_array=NoisyLTFArray.normal_weights(n=n, k=k, random_instance=RandomState(0xA1A1)),
            transform=transformation,
            combiner=combiner,
            sigma_noise=0.5,
            random_instance=RandomState(0x5015C),
        )

        noisy_distances = PropertyTest.intra_distance_leuven(noisy_instance, challenges, measurements=measurements)
        # For a noisy simulation the mean distance must differ from zero
        self.assertNotEqual(mean(noisy_distances), 0.0)

    def test_reliability_leuven(self):
        """This method test the leuven reliability statistic of an instance set."""
        n = 8
        k = 1
        N = 2 ** n
        instance_count = 2
        measurements = 100
        transformation = LTFArray.transform_id
        combiner = LTFArray.combiner_xor

        instances = [
            LTFArray(weight_array=LTFArray.normal_weights(n=n, k=k, random_instance=RandomState(0xA1A1 + i)),
                     transform=transformation, combiner=combiner) for i in range(instance_count)
        ]

        challenges = array(list(sample_inputs(n, N, random_instance=RandomState(0xFAB10))))

        property_test = PropertyTest(instances)
        reliability_statistic = property_test.reliability_leuven(challenges, measurements=measurements)
        # For an noiseless set of simulations the average intra distance must be zero
        for key, value in reliability_statistic.items():
            self.assertEqual(value, 0.0, '{}'.format(key))

        noisy_instances = [
            NoisyLTFArray(
                weight_array=LTFArray.normal_weights(n=n, k=k, random_instance=RandomState(0xA1A1 + i)),
                transform=transformation,
                combiner=combiner,
                sigma_noise=0.5,
                random_instance=RandomState(0xCABE),
            ) for i in range(instance_count)
        ]

        noisy_property_test = PropertyTest(noisy_instances)
        noisy_reliability_statistic = noisy_property_test.reliability_leuven(challenges, measurements=measurements)
        self.assertNotEqual(noisy_reliability_statistic['mean'], 0.0)

    def test_reliability(self):
        """This method tests the reliability of an instance set."""
        n = 8
        k = 1
        N = 2 ** n
        instance_count = 2
        eval_count = 100
        transformation = LTFArray.transform_id
        combiner = LTFArray.combiner_xor
        instances = [
            LTFArray(weight_array=LTFArray.normal_weights(n=n, k=k, random_instance=RandomState(0xA1A1 + i)),
                     transform=transformation, combiner=combiner) for i in range(instance_count)
        ]
        challenges = array(list(sample_inputs(n, N, random_instance=RandomState(0xFAB10))))

        property_test = PropertyTest(instances)
        reliability = property_test.reliability(challenges, measurements=eval_count)
        # For an noiseless set of simulations the average intra distance must be zero
        self.assertEqual(reliability['mean'], 0.0)

        noisy_instances = [
            NoisyLTFArray(
                weight_array=LTFArray.normal_weights(n=n, k=k, random_instance=RandomState(0xA1A1 + i)),
                transform=transformation,
                combiner=combiner,
                sigma_noise=0.5,
                random_instance=RandomState(0xCABE),
            ) for i in range(instance_count)
        ]

        noisy_property_test = PropertyTest(noisy_instances)
        noisy_reliability = noisy_property_test.reliability(challenges, measurements=eval_count)
        # For a set of noisy simulations the reliability must differ from zero
        self.assertNotEqual(noisy_reliability, 0.0)

    def test_uniqueness(self):
        """
        This method test the function which can be used to calculate the uniqueness of a set of simulation instances.
        """
        n = 8
        k = 1
        N = 2 ** n
        instance_count = 100
        transformation = LTFArray.transform_id
        combiner = LTFArray.combiner_xor
        instances = [
            LTFArray(weight_array=LTFArray.normal_weights(n=n, k=k, random_instance=RandomState(0xA1A1 + i)),
                     transform=transformation, combiner=combiner) for i in range(instance_count)
        ]
        challenges = array(list(sample_inputs(n, N, random_instance=RandomState(0xFAB10))))

        property_test = PropertyTest(instances)
        uniqueness = property_test.uniqueness(challenges)
        # For normal distributed weights is the expected uniqueness for a sufficient hug set near 0.5
        self.assertEqual(round(uniqueness['mean'], 1), 0.5)

    def test_uniqueness_leuven(self):
        """This method tests the function which can be used a statistic about the leuven intra distance."""
        n = 8
        k = 1
        N = 2 ** n
        instance_count = 100
        transformation = LTFArray.transform_id
        combiner = LTFArray.combiner_xor
        instances = [
            LTFArray(weight_array=LTFArray.normal_weights(n=n, k=k, random_instance=RandomState(0xA1A1 + i)),
                     transform=transformation, combiner=combiner) for i in range(instance_count)
        ]
        challenges = array(list(sample_inputs(n, N, random_instance=RandomState(0xFAB10))))

        property_test = PropertyTest(instances)
        uniqueness_statistic = property_test.uniqueness_leuven(challenges)
        # For normal distributed weights is the expected uniqueness for a sufficient hug set near 0.5
        self.assertEqual(round(uniqueness_statistic['mean'], 1), 0.5)
