#Adapted NSGA-II Implementation from https://github.com/jamiebull1/nsga-ii-python/blob/master/nsga2.py

import sys, random
import numpy as np
import pickle
import torch
import time
import math
from os.path import join, exists
import multiprocessing
import gc
import copy

from multiprocessing import set_start_method
set_start_method('forkserver', force=True)

class Solution:
    '''
    Abstract solution. To be implemented.
    '''
    
    def __init__(self, num_objectives):
        '''
        Constructor. Parameters: number of objectives. 
        '''
        self.num_objectives = num_objectives
        self.objectives = []
        for _ in range(num_objectives):
            self.objectives.append(None)
        self.attributes = []
        self.rank = sys.maxsize
        self.distance = 0.0
        
    def evaluate_solution(self):
        '''
        Evaluate solution, update objectives values.
        '''
        raise NotImplementedError("Solution class have to be implemented.")
    
    def crossover(self, other):
        '''
        Crossover operator.
        '''
        raise NotImplementedError("Solution class have to be implemented.")
    
    def mutate(self):
        '''
        Mutation operator.
        '''
        raise NotImplementedError("Solution class have to be implemented.")
    
    def __rshift__(self, other):
        '''
        True if this solution dominates the other (">>" operator).
        '''
        dominates = False
        
        for i in range(len(self.objectives)):
            if self.objectives[i] > other.objectives[i]:
                return False
                
            elif self.objectives[i] < other.objectives[i]:
                dominates = True
        
        return dominates
        
    def __lshift__(self, other):
        '''
        True if this solution is dominated by the other ("<<" operator).
        '''
        return other >> self

    def set_rank(self, rank):
        self.rank = rank


def crowded_comparison(s1, s2):
    '''
    Compare the two solutions based on crowded comparison.
    '''

    if s1.rank < s2.rank:
      return 1
        
    elif s1.rank > s2.rank:
        return -1
     
    elif s1.objectives[1] < s2.objectives[1]: #
        return 1
        
    elif s1.objectives[1] > s2.objectives[1]: 
        return -1
        
    else:
        return 0


class NSGAII:
    '''
    Implementation of NSGA-II algorithm.
    '''
    current_evaluated_objective = 0

    def __init__(self, num_objectives, mutation_rate, crossover_rate, elite_evals, top, threads, timelimit, pop_size, inno):
        '''
        Constructor. Parameters: number of objectives, mutation rate (default value 10%) and crossover rate (default value 100%). 
        '''
        self.num_objectives = num_objectives
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        self.elite_evals = elite_evals
        self.top  = top  #Number of top individuals that should be reevaluated

        random.seed()
        self.threads = threads
        multi_process = threads>1

        from train import T1Solution

        self.P = []
        for i in range(pop_size):
            self.P.append(T1Solution('cpu', timelimit, inno, multi= multi_process ) )

        self.vae_individual = T1Solution('cpu', timelimit, inno,  multi= multi_process )

        
    def run(self, population_size, max_generations, filename, folder):
        '''
        Run NSGA-II. 
        '''

        Q = []
        
        self.new_max = False
        max_fitness = -sys.maxsize

        fitness_file = open(folder+"/fitness_"+filename+".txt", 'a')

        ind_fitness_file = open(folder+"/individual_fitness_"+filename+".txt", 'a')
        
        generations = 0
        i = 0

        P = self.P

        pop_name = folder+"/pop_"+filename+".p"
        if False and exists( pop_name ): #Disabled loading existing population
            pop_tmp = torch.load(pop_name)

            print("Loading existing population ",pop_name, len(pop_tmp))

            idx = 0
            for s in pop_tmp:
                 P[idx].r_gen.vae.load_state_dict ( s['vae'].copy() )
                 P[idx].r_gen.controller.load_state_dict ( s['controller'].copy() )
                 P[idx].r_gen.mdrnn.load_state_dict ( s['mdrnn'].copy() )
                 P[idx].age = s['age']

                 i = s['generation'] + 1
                 idx+=1

        while (True): 
            pool = multiprocessing.Pool(self.threads)

            start_time = time.time()

            print("Iteration ", i, " ",generations)
            sys.stdout.flush()

            R = []
            R.extend(P)
            R.extend(Q)

            print("Evaluating x individuals: ",len(R) )
            for s in R:  
                s.run_solution(pool, 1, force_eval=True)
                s.age += 1

            for s in R:
                 fitness, fr, _ = s.evaluate_solution(1) 
                 #save_pop += [{'vae': s.vae_parameters, 'controller': s.controller_parameters, 'mdrnn':s.mdrnn_parameters, 'age':s.age, 'fitness':fitness, 'generation':i}]
                 generations += fr

            fronts = self.fast_nondominated_sort(R)

            del P[:]
            
            front_idx = 0

            #print(fronts.keys()[0])
            for front in fronts.values():
                #print(front.objectives[0], front.objectives[1])

                if len(front) == 0:
                    break
                
                self.crowding_distance_assignment(front)      #Not doing anything if we don't sort_crowding afterw
                
                for p in front:
                   p.set_rank( front_idx )     #! NEW

                front_idx += 1

                #print(front_idx)
                P.extend(front)
                
                if len(P) >= population_size: 
                    break


            fitness = np.asarray( [-p.objectives[1] for p in P] )
            sorted_idx = np.argsort(fitness, kind='mergesort') #! NEW

            max_fitness_gen = -sys.maxsize #keep track of highest gen fitness this generation

            elitism = True

            if elitism:
                print("Evaluating elite. top ", self.top)

                for k in sorted_idx[-self.top:]:      #-3
                    P[k].run_solution(pool, self.elite_evals, force_eval=True) #Added 21.7.19 (force_eval=True)
                
                for k in sorted_idx[-self.top:]:
                    f, _, _ = P[k].evaluate_solution(self.elite_evals) 

                    if f>max_fitness_gen:
                        max_fitness_gen = f
                        elite = P[k]

                    if f > max_fitness: #best fitness ever found
                        max_fitness = f
                        print("\tFound new champion ", max_fitness, P[k].age )

                        best_ever = P[k]
                        sys.stdout.flush()
                        
                        torch.save({'vae': elite.r_gen.vae.state_dict(), 'controller': elite.r_gen.controller.state_dict(), 'mdrnn':elite.r_gen.mdrnn.state_dict(), 'age':elite.age, 'fitness':f}, "{0}/best_{1}G{2}.p".format(folder, filename, i))

                elite.rank = -1  #! The best rank

            else:
                k = sorted_idx[-1] 

                if fitness[k]>max_fitness_gen:
                        max_fitness_gen = fitness[k]
                        elite = P[k]

                if fitness[k] > max_fitness: #best fitness ever found
                    max_fitness = fitness[k]
                    print("\tFound new champion ", max_fitness, P[k].age)

                    best_ever = P[k]
                    sys.stdout.flush()

                    torch.save({'vae': elite.r_gen.vae.state_dict(), 'controller': elite.r_gen.controller.state_dict(), 'mdrnn':elite.r_gen.mdrnn.state_dict(), 'age':elite.age, 'fitness':fitness[k], 'latents':elite.r_gen.latent_vector, 'obs':elite.r_gen.observations}, "{0}/best_{1}G{2}.p".format(folder, filename, i))

            
            self.sort_crowding(P)

            sys.stdout.flush()

            pool.close()

            Q = []    

            if len(P) > population_size:
                del P[population_size:]

            save_pop = []


            for s in P:
                 print( s.objectives[1], s.objectives[0] )
                 ind_fitness_file.write( "Gen\t%d\tFitness\t%f\tAge\t%f\tID\t%f\tParent\t%f\n" % (i, -s.objectives[1], s.age, s.id, s.parent_id )  )  # python will convert \n to os.linesep
                 ind_fitness_file.flush()

                 save_pop += [{'vae': s.r_gen.vae.state_dict(), 'controller': s.r_gen.controller.state_dict(), 'mdrnn':s.r_gen.mdrnn.state_dict(), 'age':s.age, 'fitness':fitness, 'generation':i}]
                 
        
            if (i % 25 == 0):
                print("saving population")
                torch.save(save_pop, folder+"/pop_"+filename+".p")
    
            print("Creating new population ...", len(P))
            Q = self.make_new_pop(P)

            elapsed_time = time.time() - start_time

            print( "%d\tAverage\t%f\tMax\t%f\tMax ever\t%f\t%f\n" % (i, np.mean(fitness), max_fitness_gen, max_fitness, elapsed_time) )  # python will convert \n to os.linesep

            fitness_file.write( "%d\tAverage\t%f\tMax\t%f\tMax ever\t%f\t%f\n" % (i, np.mean(fitness), max_fitness_gen, max_fitness, elapsed_time) )  # python will convert \n to os.linesep
            fitness_file.flush()

            if (i > max_generations):
                break

            gc.collect()

            i += 1

        print("Testing best ever: ")
        pool = multiprocessing.Pool(self.threads)

        best_ever.run_solution(pool, 100, early_termination=False, force_eval = True)
        avg_f, _, sd = best_ever.evaluate_solution(100)
        print(avg_f, sd)
        
        fitness_file.write( "Test\t%f\t%f\n" % (avg_f, sd) ) 

        fitness_file.close()

        ind_fitness_file.close()

            
    def sort_ranking(self, P):
        for i in range(len(P) - 1, -1, -1):
            for j in range(1, i + 1):
                s1 = P[j - 1]
                s2 = P[j]
                
                #print (s1. rank, s2.rank)
                if s1.rank > s2.rank:
                    #print("Switch rank")
                    P[j - 1] = s2
                    P[j] = s1
                    
    def sort_objective(self, P, obj_idx):
        for i in range(len(P) - 1, -1, -1):
            for j in range(1, i + 1):
                s1 = P[j - 1]
                s2 = P[j]
                
                if s1.objectives[obj_idx] > s2.objectives[obj_idx]:
                    P[j - 1] = s2
                    P[j] = s1
                    
    def sort_crowding(self, P):
        for i in range(len(P) - 1, -1, -1):
            for j in range(1, i + 1):
                s1 = P[j - 1]
                s2 = P[j]
                
                if crowded_comparison(s1, s2) < 0:
                    P[j - 1] = s2
                    P[j] = s1
                
    def make_new_pop(self, P):
        '''
        Make new population Q, offspring of P. 
        '''
        Q = []
        
        while len(Q) != len(P):
            selected_solutions = [None, None]
            
            if (True):
                while selected_solutions[0] == selected_solutions[1]:
                    for i in range(2):
                        s1 = random.choice(P)
                        s2 = s1
                        while s1 == s2:
                            s2 = random.choice(P)
                        
                        if crowded_comparison(s1, s2) > 0:
                            selected_solutions[i] = s1
                                
                        else:
                            selected_solutions[i] = s2

                        #print (s1.objectives, s2.objectives, crowded_comparison(s1, s2), s2.rank )

            else:
                s1 = random.choice(P)
                s2 = s1
                while s1 == s2:
                    s2 = random.choice(P)
                
                if crowded_comparison(s1, s2) > 0:
                    selected_solutions[0] = s1
                else:
                    selected_solutions[0] = s2        
            

            child_solution = selected_solutions[0].crossover(None)  #selected_solutions[1] is not really used and is
            child_solution.mutate()
            
            if (not child_solution in Q):     #If it's already in there we started evaluating it
                #child_solution.run_solution()
                Q.append(child_solution)
        
        return Q
        
    def fast_nondominated_sort(self, P):
        '''
        Discover Pareto fronts in P, based on non-domination criterion. 
        '''
        fronts = {}
        
        S = {}
        n = {}
        for s in P:
            S[s] = []
            n[s] = 0
            
        fronts[1] = []
        
        for p in P:
            for q in P:
                if p == q:
                    continue
                
                if p >> q:
                    #print("D ",p.objectives[0], p.objectives[1])
                    S[p].append(q)
                
                elif p << q:
                    n[p] += 1
            
            if n[p] == 0:    #! If not dominated add to front
                fronts[1].append(p)
                #p.rank = -1
        
        i = 1
        
        while len(fronts[i]) != 0:
            next_front = []
            
            for r in fronts[i]:
                for s in S[r]:
                    n[s] -= 1
                    if n[s] == 0:
                        next_front.append(s)
            
            i += 1
            fronts[i] = next_front
                    
        return fronts
        
    def crowding_distance_assignment(self, front):
        '''
        Assign a crowding distance for each solution in the front. 
        '''
        for p in front:
            p.distance = 0
        
        for obj_index in range(self.num_objectives):
            self.sort_objective(front, obj_index)
            
            front[0].distance = float('inf')
            front[len(front) - 1].distance = float('inf')
            
            for i in range(1, len(front) - 1):
                front[i].distance += (front[i + 1].objectives[obj_index] - front[i - 1].objectives[obj_index])  #Fixed crowding
                #front[i].distance += (front[i + 1].distance - front[i - 1].distance)
