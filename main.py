
import argparse
import sys
from os.path import exists
from os import mkdir
from torch.multiprocessing import Process, Queue
import torch
import numpy as np
from train import RolloutGenerator, T1Solution
from nsga2 import NSGAII
import multiprocessing
#print("version ",multiprocessing.__version__)

torch.set_num_threads(1)

def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--pop-size', type=int, default = 10, help='Population size.')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')

    parser.add_argument('--generations', type=int, default=1000, metavar='N',
                            help='number of generations to train')

    parser.add_argument('--threads', type=int, default=5, metavar='N',
                            help='threads')

    parser.add_argument('--inno', type=int, default=1, metavar='N',
                            help='0 = no protection, 1 = protection')
    

    parser.add_argument('--test', type=str, default='', metavar='N',
                            help='0 = no protection, 1 = protection')

    parser.add_argument('--folder', type=str, default='results', metavar='N',
                            help='folder to store results')

    
    parser.add_argument('--top', type=int, default=3, metavar='N',
                            help='top elites that should be re-evaluated')
    
    
    parser.add_argument('--elite_evals', type=int, default=10, metavar='N',
                            help='how many times should the elite be evaluated')                        


    parser.add_argument('--timelimit', type=int, default=500, metavar='N',
                            help='time limit on driving task')    

    args = parser.parse_args()

    device = 'cpu'

    if args.test!='':
        to_evaluate = []

        t1 = T1Solution("test", 'cpu', 10000000, 1, 0, multi=False)
        t1.load_solution(args.test)
        to_evaluate.append(t1)
        exit()

        log_file = open("log.txt", 'a')

        for ind in to_evaluate:
            if (args.threads == 1):
                average = []
                print("Evaluting genome ",args.test)
                for i in range(1):
                    f = ind.r_gen.do_rollout(False, True, 0, False)
                    average += [f]

                print(np.average(average), np.std(average) )
            else:
                print("Evaluating on threads ",args.threads)
                pool = multiprocessing.Pool(args.threads)
                ind.multi = True
                ind.run_solution(pool, 100, early_termination=False, force_eval = True)
                avg_f, _, sd = ind.evaluate_solution(100)
                print(avg_f, sd)
                log_file.write("%f\t%f" % (avg_f, sd))
                log_file.flush()

        log_file.close()
        #print (t1.evaluate_on_test (args.frames) )
        exit()

    if not exists(args.folder):
        mkdir(args.folder)

    nsga2 = NSGAII(2, 0.9, 1.0, args.elite_evals, args.top, args.threads, args.timelimit, args.pop_size, args.inno) #mutation rate, crossover rate

    nsga2.run(args.pop_size, args.generations, "{0}_{1}_".format(args.inno, args.seed), args.folder ) #pop_size, num_gens

if __name__ == '__main__':

    main(sys.argv)
