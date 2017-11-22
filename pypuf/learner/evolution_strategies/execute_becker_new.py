import numpy as np
import itertools as it
import time

from pypuf import tools
from pypuf.simulation.arbiter_based.ltfarray import LTFArray, NoisyLTFArray
from pypuf.learner.evolution_strategies.reliability_based_cmaes import ReliabilityBasedCMAES as Becker


def get_particular_accuracies(instance, model, k, challenges):
    challenge_num = np.shape(challenges)[0]
    assert instance.transform == model.transform
    assert instance.combiner == model.combiner
    transform = instance.transform
    combiner = instance.combiner
    accuracies = np.zeros(k)
    for i in range(k):
        model_single_LTFArray = LTFArray(model.weight_array[i, np.newaxis, :], transform, combiner)
        responses_model = model_single_LTFArray.eval(challenges)
        for j in range(k):
            original_single_LTFArray = LTFArray(instance.weight_array[j, np.newaxis, :], transform, combiner)
            responses_original = original_single_LTFArray.eval(challenges)
            accuracy = 0.5 + np.abs(0.5 - (np.count_nonzero(responses_model == responses_original) / challenge_num))
            if accuracy > accuracies[i]:
                accuracies[i] = accuracy
    return accuracies

# set path
path = 'hansen.csv'

seed = 0x777
prng = np.random.RandomState(seed)

# measure time
start_time = time.time()

# set test parameters
N = 2**13
reps = 8
noisiness = 1 / 2**4
k = 1
n = 32
limit_s = 1 / 2**12
limit_i = 2**12

# build instance of XOR Arbiter PUF to learn
transform = LTFArray.transform_id
combiner = LTFArray.combiner_xor
sigma_weight = 1
sigma_noise = NoisyLTFArray.sigma_noise_from_random_weights(n, sigma_weight, noisiness)
mu = 0
sigma = sigma_weight
weight_array = LTFArray.normal_weights(n, k, mu, sigma, prng)
instance = NoisyLTFArray(weight_array, transform, combiner, sigma_noise, prng)

# sample challenges
challenges = tools.sample_inputs(n, N, prng)

# extract responses from instance
responses_repeated = np.zeros((reps, N))
for i in range(reps):
    challenges, cs = it.tee(challenges)
    responses_repeated[i, :] = instance.eval(np.array(list(cs)))

# set parameters for CMA-ES
pop_size = 16
challenges = np.array(list(challenges))

class TrainingSet():
    def __init__(self, challenges, reps, responses):
        self.challenges = challenges
        self.reps = reps
        self.responses = responses

#training_set = TrainingSet(challenges, reps, responses_repeated)
training_set = tools.TrainingSet(instance, N, prng, reps)
"""becker = Becker(k, n, transform, combiner, challenges, responses_repeated,
            repetition, limit_step_size, limit_iteration, prng)"""
becker = Becker(training_set, k, n, pop_size, limit_s, limit_i, seed)
#becker.pop_size = pop_size

# learn instance and evaluate solution
model = becker.learn()
responses_model = model.eval(challenges)
responses_instance = becker.common_responses(responses_repeated)
assert len(responses_model) == len(responses_instance)
accuracy = 1 - tools.approx_dist(instance, model, 2 ** 14)
accuracy_training = 1 - (N - np.count_nonzero(responses_instance == responses_model)) / N
accuracy_instance = 1 - tools.approx_dist(instance, instance, 2 ** 14)
iterations = 0  # becker.iterations
abortions = 0  # becker.abortions
challenge_num = N
causes = 0  # becker.termination_causes
termination1 = 0  # causes[0]
termination2 = 0  # causes[1]
termination3 = 0  # causes[2]
particular_accuracies = get_particular_accuracies(instance, model, k, challenges)
runtime = time.time() - start_time

# write results into csv-file
import csv

with open(path, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(
        [accuracy] + [accuracy_training] + [accuracy_instance]
        + [particular_accuracies] + [iterations] + [abortions] + [challenge_num]
        + [reps] + [noisiness] + [k] + [n] + [limit_s]
        + [limit_i] + [termination1] + [termination2] + [termination3]
        + [pop_size] + [runtime])

print('...learned...')

# print results
print('accuracy =', accuracy)
print('accuracy_training =', accuracy_training)
print('accuracy_instance =', accuracy_instance)
print('particular_accuracies =', particular_accuracies)
print('iterations =', iterations)
print('abortions =', abortions)
print('challenge_num =', challenge_num)
print('repetitions =', reps)
print('noisiness =', noisiness)
print('k =', k)
print('n =', n)
print('pop_size =', pop_size)
print('step_size_limit =', limit_s)
print('iteration_limit =', limit_i)
print('causes =', causes)
print("--- %s seconds ---" % runtime)

print('___finished___')
