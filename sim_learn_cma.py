from numpy.random import RandomState

from pypuf.simulation.arbiter_based.ltfarray import NoisyLTFArray
from pypuf.learner.evolution_strategies.reliability_based_cmaes import ReliabilityBasedCMAES
from pypuf import tools

n = 32
k = 1
seed = 0x1234
random_instance = RandomState(seed)
weight_array = NoisyLTFArray.normal_weights(n, k, mu=0, sigma=1, random_instance=random_instance)
transform = NoisyLTFArray.transform_id
combiner = NoisyLTFArray.combiner_xor
sigma_noise = 1

instance = NoisyLTFArray(weight_array, transform, combiner, sigma_noise, random_instance)

N = 2**13
reps = 8
training_set = tools.TrainingSet(instance, N, reps)
pop_size = 24
step_size_limit = 1/2**12
iteration_limit = 2**12
learner = ReliabilityBasedCMAES(training_set, k, n, pop_size, step_size_limit, iteration_limit, seed_model=seed)

res = learner.learn()

accuracy = 1 - tools.approx_dist(instance, res, 2 ** 14)
print('accuracy =', accuracy)