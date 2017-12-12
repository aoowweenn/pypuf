"""This module can be used to characterize the properties of a puf class."""
from numpy import array
from numpy.random import RandomState
from pypuf.simulation.arbiter_based.ltfarray import LTFArray, NoisyLTFArray, SimulationMajorityLTFArray
from pypuf.experiments.experiment.base import Experiment
from pypuf.property_test.base import PropertyTest
from pypuf.tools import sample_inputs


class ExperimentPropertyTest(Experiment):
    """
    This class can be used to test several puf simulations instances with the pypuf.property_test.base.PropertyTest
    class.
    """
    def __init__(self, log_name, test_function, challenge_count, measurements, challenge_seed, ins_gen_function,
                 param_ins_gen):
        """
        :param test_function:
        :param challenge_count:
        :param measurements:
        :param challenge_seed:
        :param ins_gen_function:
        :param param_ins_gen:
        :return:
        """
        super().__init__(log_name=log_name)
        self.log_name = log_name
        self.test_function = test_function
        self.challenge_count = challenge_count
        self.challenge_seed = challenge_seed
        self.measurements = measurements
        self.ins_gen_function = ins_gen_function
        self.param_ins_gen = param_ins_gen
        self.result = None

    def run(self):
        """Runs a property test."""
        instances = self.ins_gen_function(**self.param_ins_gen)
        n = self.param_ins_gen['n']
        challenge_prng = RandomState(self.challenge_seed)
        challenges = array(list(sample_inputs(n, self.challenge_count, random_instance=challenge_prng)))
        property_test = PropertyTest(instances, logger=self.progress_logger)
        self.result = self.test_function(property_test, challenges, measurements=self.measurements)

    def analyze(self):
        """Summarize the results of the search process."""
        assert self.result is not None
        mean = self.result.get('mean', float("inf"))
        median = self.result.get('median', float("inf"))
        minimum = self.result.get('min', float("inf"))
        maximum = self.result.get('max', float("inf"))
        standard_deviation = self.result.get('sd', float("inf"))
        msg = '{}\t{:f}\t{:f}\t{:f}\t{:f}\t{:f}'.format(self.log_name, mean, median, minimum, maximum, standard_deviation)
        self.result_logger.info(msg)

    @classmethod
    def create_ltf_arrays(cls, n=8, k=1, instance_count=10, transformation=LTFArray.transform_id,
                          combiner=LTFArray.combiner_xor, bias=None, mu=0, sigma=1, weight_random_seed=0x123):
        """
        This function can be used to create a list of LTFArrays.
        :param n:
        :param k:
        :param instance_count:
        :param transformation:
        :param combiner:
        :param bias:
        :param mu:
        :param sigma:
        :param weight_random_seed:
        :return:
        """
        instances = []
        for seed_offset in range(instance_count):
            weight_array = LTFArray.normal_weights(n, k, mu, sigma,
                                                   random_instance=RandomState(weight_random_seed + seed_offset))
            instances.append(
                LTFArray(
                    weight_array=weight_array,
                    transform=transformation,
                    combiner=combiner,
                    bias=bias,
                )
            )
        return instances

    @classmethod
    def create_noisy_ltf_arrays(cls, n=8, k=1, instance_count=10, transformation=LTFArray.transform_id,
                                combiner=LTFArray.combiner_xor, bias=None, mu=0, sigma=1, weight_random_seed=0x123,
                                sigma_noise=0.5, noise_random_seed=0x321):
        """
        This function can be used to create a list of NoisyLTFArray.
        :param n:
        :param k:
        :param instance_count:
        :param transformation:
        :param combiner:
        :param bias:
        :param mu:
        :param sigma:
        :param weight_random_seed:
        :param sigma_noise:
        :param noise_random_seed
        :return:
        """
        instances = []
        for seed_offset in range(instance_count):
            weight_array = LTFArray.normal_weights(n, k, mu, sigma,
                                                   random_instance=RandomState(weight_random_seed + seed_offset))
            instances.append(
                NoisyLTFArray(
                    weight_array=weight_array,
                    transform=transformation,
                    combiner=combiner,
                    sigma_noise=sigma_noise,
                    random_instance=RandomState(noise_random_seed),
                    bias=bias,
                )
            )
        return instances

    @classmethod
    def create_mv_ltf_arrays(cls, n=8, k=1, instance_count=10, transformation=LTFArray.transform_id,
                             combiner=LTFArray.combiner_xor, bias=None, mu=0, sigma=1, weight_random_seed=0x123,
                             sigma_noise=0.5, noise_random_seed=0x321, vote_count=3):
        """
        This function can be used to create a list of SimulationMajorityLTFArray.
        :param n:
        :param k:
        :param instance_count:
        :param transformation:
        :param combiner:
        :param bias:
        :param mu:
        :param sigma:
        :param weight_random_seed:
        :param sigma_noise:
        :param noise_random_seed:
        :param vote_count:
        :return:
        """
        instances = []
        for seed_offset in range(instance_count):
            weight_array = LTFArray.normal_weights(n, k, mu, sigma,
                                                   random_instance=RandomState(weight_random_seed + seed_offset))
            instances.append(
                SimulationMajorityLTFArray(
                    weight_array=weight_array,
                    transform=transformation,
                    combiner=combiner,
                    sigma_noise=sigma_noise,
                    random_instance_noise=RandomState(noise_random_seed),
                    bias=bias,
                    vote_count=vote_count,
                )
            )
        return instances
