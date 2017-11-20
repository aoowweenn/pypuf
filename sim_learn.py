"""
This module is a command line tool which provides an interface for experiments which are designed to learn an arbiter
PUF LTFarray simulation with the logistic regression learning algorithm. If you want to use this tool you will have to
define nine parameters which define the experiment.
"""
from sys import argv, stderr
import argparse
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf.experiments.experiment.logistic_regression import ExperimentLogisticRegression
from pypuf.experiments.experimenter import Experimenter


def main(args):
    """
    This method includes the main functionality of the module it parses the argument vector and executes the learning
    attempts on the PUF instances.
    """
    parser = argparse.ArgumentParser(
        prog='sim_learn',
        description="LTF Array Simulator and Logistic Regression Learner",
    )
    parser.add_argument("n", help="number of bits per Arbiter chain", type=int)
    parser.add_argument("k", help="number of Arbiter chains", type=int)
    parser.add_argument(
        "transformation",
        help="used to transform input before it is used in LTFs. Current available: "
             '"1_1_bent",'
             '"1_n_bent",'
             '"atf,id",'
             '"lightweight_secure",'
             '"lightweight_secure_original",'
             '"mm",'
             '"permutation_atf",'
             '"polynomial,random",'
             '"shift",'
             '"shift_lightweight_secure",'
             '"soelter_lightweight_secure"',
        type=str,
    )
    parser.add_argument(
        'combiner',
        help='used to combine the output bits to a single bit. Current available: "ip_mod2", "xor"',
        type=str,
    )
    parser.add_argument('N', help='number of challenge response pairs in the training set', type=int)
    parser.add_argument('restarts', help='number of repeated initializations the learner', type=int)
    parser.add_argument(
        'instances',
        help='number of repeated initializations the instance\n'
             'The number total learning attempts is restarts*instances.',
        type=int,
    )
    parser.add_argument('seed_instance', help='random seed used for LTF array instance', type=str)
    parser.add_argument('seed_model', help='random seed used for the model in first learning attempt', type=str)
    parser.add_argument(
        '--log_name',
        help='path to the logfile which contains results from all instances. The tool '
             'will add a ".log" to log_name. The default path is ./sim_learn.log',
        default='sim_learn',
        type=str,
    )
    parser.add_argument(
        '--seed_challenges',
        help='random seed used to draw challenges for the training set',
        type=str,
    )
    parser.add_argument('--seed_distance', help='random seed used to calculate the accuracy', type=str)

    args = parser.parse_args(args)

    n = args.n
    k = args.k
    transformation_name = args.transformation
    combiner_name = args.combiner
    N = args.N
    restarts = args.restarts

    instances = args.instances

    seed_instance = int(args.seed_instance, 16)
    seed_model = int(args.seed_model, 16)

    seed_challenges = 0x5A551
    if args.seed_challenges is not None:
        seed_challenges = int(args.seed_challenges, 16)
    seed_distance = 0xB055
    if args.seed_distance is not None:
        seed_distance = int(args.seed_distance, 16)

    transformation = None
    combiner = None

    try:
        transformation = getattr(LTFArray, 'transform_%s' % transformation_name)
    except AttributeError:
        stderr.write('Transformation %s unknown or currently not implemented\n' % transformation_name)
        quit()

    try:
        combiner = getattr(LTFArray, 'combiner_%s' % combiner_name)
    except AttributeError:
        stderr.write('Combiner %s unknown or currently not implemented\n' % combiner_name)
        quit()

    log_name = args.log_name

    stderr.write('Learning %s-bit %s XOR Arbiter PUF with %s CRPs and %s restarts.\n\n' % (n, k, N, restarts))
    stderr.write('Using\n')
    stderr.write('  transformation:       %s\n' % transformation)
    stderr.write('  combiner:             %s\n' % combiner)
    stderr.write('  instance random seed: 0x%x\n' % seed_instance)
    stderr.write('  model random seed:    0x%x\n' % seed_model)
    stderr.write('\n')

    # create different experiment instances
    experiments = []
    for j in range(instances):
        for start_number in range(restarts):
            l_name = '%s_%i_%i' % (log_name, j, start_number)
            experiment = ExperimentLogisticRegression(
                log_name=l_name,
                n=n,
                k=k,
                N=N,
                seed_instance=seed_instance + j,
                seed_model=seed_model + j + start_number,
                transformation=transformation,
                combiner=combiner,
                seed_challenge=seed_challenges,
                seed_chl_distance=seed_distance,
            )
            experiments.append(experiment)

    experimenter = Experimenter(log_name, experiments)
    # run the instances
    experimenter.run()

    # output format
    str_format = '{:<15}\t{:<10}\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<18}\t{:<15}\t{:<6}\t{:<8}\t{:<8}\t{:<8}'
    headline = str_format.format(
        'seed_instance', 'seed_model', 'i', 'n', 'k', 'N', 'trans', 'comb', 'iter', 'time', 'accuracy',
        'model_values\n'
    )
    # print the result headline
    stderr.write(headline)

    log_file = open(log_name + '.log', 'r')

    # print the results
    result = log_file.readline()
    while result != '':
        stderr.write(str_format.format(*result.split('\t')))
        result = log_file.readline()

    log_file.close()


if __name__ == '__main__':
    main(argv[1:])
