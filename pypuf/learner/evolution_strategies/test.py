from pypuf.experiments.experiment.reliability_based_cmaes import ExperimentReliabilityBasedCMAES


log_name = 'cma_log'
seed_instance = 0xAbc
k = 1
n = 32
noisiness = 1 / 2**4
seed_challenges = 0x777
num = 2**13
reps = 8
seed_model = 0x123
pop_size = 16
step_size_limit = 1 / 2**12
iteration_limit = 2**12

experiment = ExperimentReliabilityBasedCMAES(
    log_name, seed_instance, k, n, noisiness,
    seed_challenges, num, reps,
    seed_model, pop_size, step_size_limit, iteration_limit,
)

experiment.run()
experiment.analyze()
experiment.result_logger.addHandler('result_handler')

# git fetch
# git cherry-pick 5eea03440c0f59cafe569c46526064cabd8503c0
