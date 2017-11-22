"""
This module is a command line tool which provides an interface for experiments which are designed to learn an arbiter
PUF LTFarray simulation with the reliability based CMAES learning algorithm. If you want to use this tool you will have
to define nine parameters which define the experiment.
"""
from sys import argv, stderr
from pypuf.experiments.experiment.reliability_based_cmaes import ExperimentReliabilityBasedCMAES
from pypuf.experiments.experimenter import Experimenter


def main(args):
    """
    This method includes the main functionality of the module it parses the argument vector and executes the learning
    attempts on the PUF instances.
    """
    if len(args) < 9 or len(args) > 13:
        stderr.write('LTF Array Simulator and Reliability based CMAES Learner\n')
        stderr.write('Usage:\n')
        stderr.write(
            'python sim_rel_cmaes_attack.py n k noisiness num reps pop_size [seed_i] [seed_c] [seed_m] [log_name]\n')
        stderr.write('      n: number of bits per Arbiter chain\n')
        stderr.write('      k: number of Arbiter chains\n')
        stderr.write('      noisiness: proportion of noise scale related to the scale of variability\n')
        stderr.write('      num: number of different challenges in the training set\n')
        stderr.write('      reps: number of responses for each challenge in the training set\n')
        stderr.write('      pop_size: number of solution points sampled per iteration within the CMAES algorithm\n')
        stderr.write('      restarts: number of repeated initializations of the learner\n')
        stderr.write('      instances: number of repeated initializations of the instance\n')
        stderr.write('              The number of total learning attempts is restarts*instances.\n')
        stderr.write('      seed_i: random seed used for creating LTF array instance and for simulating noise\n')
        stderr.write('      seed_c: random seed used for sampling challenges\n')
        stderr.write('      seed_m: random seed used for the model in first learning attempt\n')
        stderr.write('      [log_name]: path to the logfile which contains results from all instances. The tool '
                     'will add a ".log" to log_name. The default path is ./sim_learn.log\n')
        quit(1)

    n = int(args[1])
    k = int(args[2])
    noisiness = float(args[3])
    num = int(args[4])
    reps = int(args[5])
    pop_size = int(args[6])
    restarts = int(args[7])
    instances = int(args[8])

    seed_i = 0x0123
    seed_c = 0x4567
    seed_m = 0x8910
    if len(args) >= 12:
        seed_i = int(args[9], 16)
        seed_c = int(args[10], 16)
        seed_m = int(args[11], 16)

    log_name = 'sim_rel_cmaes'
    if len(args) == 13:
        log_name = args[12]

    stderr.write('Learning %s-bit %s XOR Arbiter PUF with %f noisiness'
                 'using %s different %s times repeated CRPs and %s restarts,'
                 'where %s solution points are sampled each iteration of the CMAES algorithm.\n\n'
                 % (n, k, num, noisiness, reps, restarts, pop_size))
    stderr.write('The following seeds are used for generating pseudo random numbers.\n')
    stderr.write('  seed for instance: 0x%x\n' % seed_i)
    stderr.write('  seed for challenges: 0x%x\n' % seed_c)
    stderr.write('  seed for model:    0x%x\n' % seed_m)
    stderr.write('\n')

    # Fix limits for CMAES (this should be changed later!)
    limit_s = 1 / 2**12
    limit_i = 2**12

    # Create different experiment instances
    experiments = []
    for j in range(instances):
        for start_number in range(restarts):
            l_name = '%s_%i_%i' % (log_name, j, start_number)
            experiment = ExperimentReliabilityBasedCMAES(
                log_name=l_name,
                seed_i=seed_i + j,
                k=k,
                n=n,
                noisiness=noisiness,
                seed_c=seed_c,
                num=num,
                reps=reps,
                seed_m=seed_m + j + start_number,
                pop_size=pop_size,
                step_size_limit=limit_s,
                iteration_limit=limit_i,
            )
            experiments.append(experiment)

    experimenter = Experimenter(log_name, experiments)
    # run the instances
    experimenter.run()


if __name__ == '__main__':
    main(argv)
