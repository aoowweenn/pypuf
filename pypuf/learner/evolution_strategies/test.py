import numpy as np
from numpy import squeeze, array, zeros
import itertools
from pypuf.tools import sample_inputs
from pypuf.simulation.arbiter_based.ltfarray import LTFArray, NoisyLTFArray
import cma

opts = cma.CMAOptions('tol')
#print(opts)

for i in range(2**2):
    print('i =', i)

a = np.zeros((1, 2))
print('a =', a)
a = np.squeeze(a, axis=0)
print('a =', a)

def foo(b=None):
    if b==None:
        print('none')
    else:
        print('yes')

foo()


def init(self, instance, N, reps=None):
    """
    :param instance: pypuf.simulation.base.Simulation
                     Instance which is used to generate responses for random challenges.
    :param N: int
              Number of desired challenges
    """
    self.instance = instance
    self.challenges = array(list(sample_inputs(instance.n, N)))
    if reps == None:
        reps = 1
    self.responses = zeros((reps, N))
    for i in range(reps):
        self.challenges, cs = itertools.tee(self.challenges)
        self.responses[i, :] = instance.eval(array(list(cs)))
    self.responses = instance.eval(self.challenges)
    if reps == 1:
        self.responses = squeeze(self.responses, axis=0)
    self.N = N
    self.reps = reps

challenge_num = int(challenge_num)
repetition = int(repetition)
noisiness = noisiness
k = int(k)
n = int(n)
transform = LTFArray.transform_id
combiner = LTFArray.combiner_xor
sigma_weight = 1
sigma_noise = NoisyLTFArray.sigma_noise_from_random_weights(n, sigma_weight, noisiness)
mu = 0
sigma = sigma_weight
weight_array = LTFArray.normal_weights(n, k, mu, sigma, prng)
instance = NoisyLTFArray(weight_array, transform, combiner, sigma_noise, prng)

instance = LTFArray()

init(instance, 8)


