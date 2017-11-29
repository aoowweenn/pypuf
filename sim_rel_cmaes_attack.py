"""
This module is a command line tool which provides an interface for experiments which are designed to learn an arbiter
PUF LTFarray simulation with the reliability based CMAES learning algorithm. If you want to use this tool you will have
to define nine parameters which define the experiment.
"""
from sys import argv, stderr
from pypuf.experiments.experiment.reliability_based_cmaes import ExperimentReliabilityBasedCMAES
from pypuf.experiments.experimenter import Experimenter


def main(args):
    """This method includes the main functionality of the module it parses the argument vector
    and executes the learning attempts on the PUF instances.
    """
    if len(args) < 11 or len(args) > 15:
        stderr.write('\n***LTF Array Simulator and Reliability based CMAES Learner***\n\n')
        stderr.write('Usage:\n')
        stderr.write(
            'python sim_rel_cmaes_attack.py n k noisiness num reps pop_size [seed_i] [seed_c] [seed_m] [log_name]\n')
        stderr.write('      n:          number of bits per Arbiter chain\n')
        stderr.write('      k:          number of Arbiter chains\n')
        stderr.write('      noisiness:  proportion of noise scale related to the scale of variability\n')
        stderr.write('      num:        number of different challenges in the training set\n')
        stderr.write('      reps:       number of responses for each challenge in the training set\n')
        stderr.write('      pop_size:   number of solution points sampled per iteration within the CMAES algorithm\n')
        stderr.write('      limit_s:    max number of iterations with consistent fitness within the CMAES algorithm\n')
        stderr.write('      limit_i:    max number of overall iterations within the CMAES algorithm\n')
        stderr.write('      restarts:   number of repeated initializations of the learner\n')
        stderr.write('      instances:  number of repeated initializations of the instance\n')
        stderr.write('                   The number of total learning attempts is restarts times instances.\n')
        stderr.write('      seed_i:     random seed used for creating LTF array instance and for simulating noise\n')
        stderr.write('      seed_c:     random seed used for sampling challenges\n')
        stderr.write('      seed_m:     random seed used for the model in first learning attempt\n')
        stderr.write('      [log_name]: path to the logfile which contains results from all instances.\n'
                     '                   The tool will add a ".log" to log_name. The default path is ./sim_learn.log\n')
        quit(1)

    n = int(args[1])
    k = int(args[2])
    noisiness = float(args[3])
    num = int(args[4])
    reps = int(args[5])
    pop_size = int(args[6])
    limit_s = float(args[7])
    limit_i = int(args[8])
    restarts = int(args[9])
    instances = int(args[10])

    seed_i = 0x0123
    seed_c = 0x4567
    seed_m = 0x8910
    if len(args) >= 14:
        seed_i = int(args[11], 16)
        seed_c = int(args[12], 16)
        seed_m = int(args[13], 16)

    log_name = 'sim_rel_cmaes'
    if len(args) == 15:
        log_name = args[14]

    stderr.write('Learning %i times each %i (%i,%i)-XOR Arbiter PUF(s) with %f noisiness, '
                 'using %i different %i times repeated CRPs.\n'
                 'There, %i solution points are sampled each iteration of the CMAES algorithm. '
                 'Among other termination criteria, it stops if the fitness stagnates since %i iterations '
                 'or the total number of iterations equals %i.\n\n'
                 % (restarts, instances, n, k, noisiness, num, reps, pop_size, limit_s, limit_i))
    stderr.write('The following seeds are used for generating pseudo random numbers.\n')
    stderr.write('  seed for instance: 0x%x\n' % seed_i)
    stderr.write('  seed for challenges: 0x%x\n' % seed_c)
    stderr.write('  seed for model:    0x%x\n' % seed_m)
    stderr.write('\n')

    # Create different experiment instances
    experiments = []
    for instance in range(instances):
        for attempt in range(restarts):
            l_name = '%s_%i_%i' % (log_name, instance, attempt)
            experiment = ExperimentReliabilityBasedCMAES(
                log_name=l_name,
                seed_i=seed_i + instance,
                k=k,
                n=n,
                noisiness=noisiness,
                seed_c=seed_c + instance,
                num=num,
                reps=reps,
                seed_m=seed_m + attempt,
                pop_size=pop_size,
                step_size_limit=limit_s,
                iteration_limit=limit_i,
            )
            experiments.append(experiment)

    experimenter = Experimenter(log_name, experiments)
    # Run the instances
    experimenter.run()


if __name__ == '__main__':
    main(argv)
