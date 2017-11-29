from pypuf.experiments.experiment.reliability_based_cmaes import ExperimentReliabilityBasedCMAES
from pypuf.experiments.experimenter import Experimenter

log_name = 'xor_cma_log'
seed_instance = 0x248
k = 1
n = 32
noisiness = 1 / 2**4
seed_challenges = 0x777
num = 2**13
reps = 8
seed_model = 0x123
pop_size = 32
stagnation_limit = 1 / 2 ** 12
iteration_limit = 2**12

experiment = ExperimentReliabilityBasedCMAES(
    log_name, seed_instance, k, n, noisiness,
    seed_challenges, num, reps,
    seed_model, pop_size, stagnation_limit, iteration_limit,
)

experiments = []
experiments.append(experiment)
experimenter = Experimenter(log_name, experiments)
# Run the instances
experimenter.run()

"""
experiment.run()
experiment.analyze()
experiment.result_logger.addHandler('result_handler')
"""
# git fetch
# git cherry-pick 5eea03440c0f59cafe569c46526064cabd8503c0
