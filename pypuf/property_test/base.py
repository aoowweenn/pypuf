"""
This module provides a class for several property tests which can be used to check the attributes of an PUF.
"""
import sys
from numpy import array, mean, median, absolute, sqrt
from numpy import min as np_min
from numpy import max as np_max
from numpy import sum as np_sum
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
        :return: float
        """
        responses = instance.eval(challenges)
        for _ in range(evaluation_count - 1):
            responses = responses + (instance.eval(challenges))
        # average distance over all distances
        mean_distance = mean(1 - absolute(responses / evaluation_count))
        return mean_distance

    @staticmethod
    def intra_distance_leuven(instance, challenges, measurements=10):
        """
        This function calculates the intra distance of a puf instance like in:
        Physically Unclonable Functions: Constructions, Properties and Applications page 20
        https://lirias.kuleuven.be/bitstream/123456789/353455/1/thesis_online.pdf
        This function uses the fractional Hamming distance.
        :param instance: pypuf.simulation.base.Simulation
        :param challenges: array of int shape(N,n)
        :param meas_index: int
                           Index which is used to select a response which is used to calculate the intra distance.
                           0 <= meas_index measurements
        :param measurements: int
                             Number of evaluations of the puf instance.
        :return: list of float
                 List of all possible distinct distances.
        """
        # Calculate the responses
        responses = [
            instance.eval(challenges) for _ in range(measurements)
        ]
        distances = []
        # Calculate the intra distances between arrarys of responses
        for response_index in reversed(range(measurements-1)):
            # Safe the responses which should be compared
            response = responses[response_index]
            # Delete response from responses because the hamming distance
            # must be calculated between distinct measurements.
            del responses[response_index]
            # Calculate the hamming distances
            dists = [hamming(response, res) for res in responses]
            # Add the distances to the list for distances
            distances = distances + dists
        return distances

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
        :return: float
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

    @staticmethod
    def inter_distance_leuven(instance_1, instance_2, responses):
        """
        This function calculates the inter distance fo two puf instances for a set of challenges like in:
        Physically Unclonable Functions: Constructions, Properties and Applications page 22
        https://lirias.kuleuven.be/bitstream/123456789/353455/1/thesis_online.pdf
        This function uses the fractional Hamming distance.
        :param instance_1: int
                           Index to reference responses from instance_1 in responses.
        :param instance_2: int
                           Index to reference responses from instance_2 in responses.
        :param responses: array of int shape(N_puf,N,n)
                          Array of evaluated challenges for N_puf challenges.
        :return: float
                 Fractional Hamming distance between the simulation instances.
        """
        return hamming(responses[instance_1], responses[instance_2])

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

    def uniqueness_leuven(self, challenges, measurements=1):
        """
        This function generates a statistic about the inter distance of a set of simulation instances like in:
        Physically Unclonable Functions: Constructions, Properties and Applications page 22
        https://lirias.kuleuven.be/bitstream/123456789/353455/1/thesis_online.pdf
        :param challenges: array of int shape(N,n)
        :return: Dictionary of float
                 Statistic of the set of instances {mean:, median:, min:, max:, standard_deviation:}
        """
        N_puf = len(self.instances)
        # This is one because we calculate the hamming distance over N bit arrays which results in one distance value.
        N_chal = 1
        N_meas = measurements
        distances = []
        # Calculate the inter distances for N_meas instance evaluations.
        for _ in range(N_meas):
            self.evaluate(challenges)
            for puf_index in range(N_puf):
                # The instances must be distinct
                for puf_index_2 in range(puf_index, N_puf):
                    distances.append(self.inter_distance_leuven(puf_index, puf_index_2, self.evaluated_challenges))
        distances = array(distances)

        factor = 2/(N_puf * (N_puf - 1) * N_chal * N_meas)
        sample_mean = factor * np_sum(distances)

        sd_factor = 2/(N_puf * (N_puf -1) * N_chal * N_meas - 2)
        standard_deviation = sqrt(sd_factor * sum((distances - sample_mean)**2))

        min_dist = np_min(distances)
        max_dist = np_max(distances)

        median_dist = median(distances)

        return {
            'mean' : sample_mean,
            'median' : median_dist,
            'min' : min_dist,
            'max' : max_dist,
            'sd' : standard_deviation,
        }

    def reliability(self, challenges, evaluation_count=100):
        """
        This function can be used to calculate the mean reliability (average intra distance) of an set of simulation
        instances.
        :param challenges: array of int shape(N,n)
        :param evaluation_count: int
                                 Number of evaluations of the Simulation.
        :return: float
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

    def reliability_leuven(self, challenges, measurements=10):
        """
        This function calculates the reliability statistic like in:
        Physically Unclonable Functions: Constructions, Properties and Applications page 20
        https://lirias.kuleuven.be/bitstream/123456789/353455/1/thesis_online.pdf
        :param challenges: array of int shape(N,n)
        :param measurements: int
                             Number of evaluations of the puf instance.
        :return: Dictionary of float
                 Statistic of the set of instances {mean:, median:, min:, max:, standard_deviation:}
        """
        # Define experiment parameter
        N_puf = len(self.instances)
        # This is one because we calculate the hamming distance over N bit arrays which results in one distance value.
        N_chal = 1
        N_meas = measurements

        distances = []
        i = 1
        for instance in self.instances:
            dists = self.intra_distance_leuven(instance, challenges, N_meas)
            distances = distances + dists
            print_progress('Calculate leuven reliability', i, N_puf)
            i = i + 1
        distances = array(distances)

        # Calculate the sample mean
        factor = 2 / (N_puf * N_chal * N_meas * (N_meas - 1))
        sample_mean = factor * np_sum(distances)

        # Calculate the standard deviation
        factor = 2 / (N_puf * N_chal * N_meas * (N_meas - 1) - 2)
        standard_deviation = sqrt(factor * np_sum((distances-sample_mean)**2))

        min_dist = min(distances)
        max_dist = max(distances)

        median_dist = median(distances)

        return {
                'mean' : sample_mean,
                'median' : median_dist,
                'min' : min_dist,
                'max' : max_dist,
                'sd' : standard_deviation,
                }



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
