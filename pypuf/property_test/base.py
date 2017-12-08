"""
This module provides a class for several property tests which can be used to check the attributes of an PUF.
"""
import sys
from numpy import array, mean, absolute
from scipy.spatial.distance import hamming


class PropertyTest(object):
    """
    This class executes essential routines for each property test.
    The set of instances is expected to be homogenous in n the number of stages.
    """

    def __init__(self, instances):
        """
        :param instances: list of pypuf.simulation.base.Simulation
        """
        self.instances = instances
        # array of evaluated challenges shape(len(self.instances), number of challenges)
        self.evaluated_challenges = None

    def evaluate(self, challenges):
        """
        This method evaluates the a set of challenges for every instance in self.instance and safes the result in
        self.evaluated_challenges.
        :param challenges: array of int shape(N,n)
        """
        result = []
        for instance in self.instances:
            result.append(instance.eval(challenges))
        self.evaluated_challenges = array(result, dtype=challenges.dtype)

    @staticmethod
    def intra_distance(instance, challenges, evaluation_count=100):
        """
        This function calculates the intra distance of an simulation instance. In detail the intra distance is defined
        by the average hamming distance
        :param instance: pypuf.simulation.base.Simulation
        :param challenges: array of int shape(N,n)
        :param evaluation_count: int
        :return: double
        """
        responses = instance.eval(challenges)
        for _ in range(evaluation_count - 1):
            responses = responses + (instance.eval(challenges))
        # average distance over all distances
        mean_distance = mean(1 - absolute(responses / evaluation_count))
        return mean_distance

    @staticmethod
    def inter_distance(instance_index, responses):
        """
        This function calculates the inter distance between responses of one simulation instance and all other
        responses in self.evaluated_challenges.
        The inter_distance between two instances is defined by the hamming distance over the responses.
        :param instance_index: int
                               Index of instance which inter_distance should be calculated
        :param responses: array of int with shape(k, N)
                          Array of N responses of k simulation instances
        :return: double
                 Mean of the hamming distances
        """
        instance_count = len(responses)
        indices = list(range(instance_count))
        del indices[instance_index]
        hamming_distances = []
        for index in indices:
            hamming_distances.append(
                hamming(responses[instance_index], responses[index])
            )

        return mean(hamming_distances)

    def uniqueness(self, challenges):
        """
        This calculates the mean inter distance between a set of simulation instances for a set of challenges.
        A value for the mean inter distance calculation is based on the mean inter distance between an instance to all
        other instances. The inter distance between two instances is calculated by the hamming distance of the response
        arrays.
        :param challenges: array of int shape(N,n)
        :return: double
                 A value which represents the uniqueness of
        """
        self.evaluate(challenges)
        instance_count = len(self.instances)
        indices = range(instance_count)
        hamming_distances = []
        # Calculates the hamming distance from an instance to each other instance.
        # The hamming distance here is the mean of mismatching entries.
        for instance_index in indices:
            hamming_distances.append(self.inter_distance(instance_index, self.evaluated_challenges))
            current_mean_distance = mean(array(hamming_distances))
            msg = 'The Current uniqueness is {:.2f} .'.format(current_mean_distance)
            print_progress(msg, instance_index+1, max_progress_value=instance_count)
        return current_mean_distance

    def reliability(self, challenges, evaluation_count=100):
        """
        This function can be used to calculate the mean reliability (average intra distance) of an set of simulation
        instances.
        :param challenges: array of int shape(N,n)
        :param evaluation_count: int
                                 Number of evaluations of the Simulation.
        :return: double
                 Value which represents the reliability of a set of simulation instances.
        """
        assert evaluation_count >= 1
        distances = []
        i = 1
        for instance in self.instances:
            distances.append(self.intra_distance(instance, challenges, evaluation_count))
            msg = 'The Current reliability is {:.2f} .'.format(mean(distances))
            print_progress(msg, i, len(self.instances))
            i = i + 1
        return mean(distances)


def print_progress(message, progress_value, max_progress_value=100):
    """
    This function can be used to print a progressbar to stderr.
    :param message: string
                    A message which is displayed on the left to the progress bar.
    :param progress_value: int
                           A Value to determine the current progress, which should be lower equal max_progress_value.
    :param max_progress_value: int default is 100
                               An upper bound for the current progress.
    """
    progress_fractal = progress_value/max_progress_value
    progress_bar = ''.join(['#' for i in range(int(progress_fractal*10))])
    sys.stderr.write('\r{} [{:<10}] {:>3.0%}'.format(message, progress_bar, progress_fractal))
    sys.stderr.flush()
