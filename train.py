""" Various auxiliary utilities """
import math 
from os.path import join, exists
import torch
from torchvision import transforms
from multiprocessing import Lock
import numpy as np
from models import MDRNNCell, VAE, Controller
import gym
from nsga2 import Solution
import copy
from torch.nn import functional as F
from torch import optim
from models.mdrnn import MDRNN, gmm_loss

import time
import random

import pickle

#gym.envs.box2d.car_racing.STATE_W, gym.envs.box2d.car_racing.STATE_H = 64, 64

ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE =\
    3, 32, 256, 64, 64

COLLECT_DATA = False

# Same
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.ToTensor()
])

#transform2 = transforms.Compose([
#    transforms.Resize((RED_SIZE, RED_SIZE)),
#    transforms.ToTensor()
#])


hidden_vector_array = []
gen_counter = 0

class RolloutGenerator(object):
    """ Utility to generate rollouts.
    """
    def __init__(self, device, time_limit):
        self.env = gym.make('CarRacing-v0')
        
        self.device = device

        self.mus_old, self.sigmas_old, self.logpi_old = 0, 0, 0

        self.time_limit = time_limit

        self.vae = VAE(3, LSIZE, 1024)#.to(device)

        self.mdrnn = MDRNNCell(LSIZE, ASIZE, RSIZE, 5)#.to(device)
        self.controller = Controller(LSIZE, RSIZE, ASIZE)#.to(device)


    def get_action_and_transition(self, obs, hidden):
        """ Get action and transition.

        Encode obs to latent using the VAE, then obtain estimation for next
        latent and next hidden state using the MDRNN and compute the controller
        corresponding action.

        :args obs: current observation (1 x 3 x 64 x 64) torch tensor
        :args hidden: current hidden state (1 x 256) torch tensor

        :returns: (action, next_hidden)
            - action: 1D np array
            - next_hidden (1 x 256) torch tensor
        """
        _, latent_mu, _ = self.vae(obs)

        action = self.controller(latent_mu, hidden[0] )

        mus, sigmas, logpi, rs, d, next_hidden = self.mdrnn(action, latent_mu, hidden)

        self.mus_old, self.sigmas_old, self.logpi_old = mus, sigmas, logpi   #Do we need to do deep copy here? 

        return action.squeeze().cpu().numpy(), next_hidden


    def do_rollout(self, render=False, eval_num = 0, early_termination=True):


        with torch.no_grad():

            self.env = gym.make('CarRacing-v0')
            
            #!l = Lock()
            #!l.acquire()
            obs = self.env.reset()
            #!l.release()

            self.env.render('rgb_array')  

            #model.env.render('rgb_array')
            hidden = [
                torch.zeros(1, RSIZE)#.to(self.device)
                for _ in range(2)]

            neg_count = 0
            last_reward = 0
            cumulative = 0

            i = 0
            while True:

                obs = transform(obs).unsqueeze(0)#.to(self.device)
              
                if (COLLECT_DATA):

                    if i==0:
                        self.observations = obs
                    else:
                        self.observations = torch.cat( (self.observations, obs ), 0)
                    #print(self.observations.shape[0])
                    
                #print(obs.shape)
                action, hidden = self.get_action_and_transition(obs, hidden)

                obs, reward, done, _ = self.env.step(action)

                neg_count = neg_count+1 if reward < 0.0 else 0

                #render = False
                if render:
                    o = self.env.render("human")

                
                #print(early_termination)
                if (neg_count>20 and early_termination):
                    done = True
                    #cumulative = max(0, cumulative-100)

                cumulative += reward

                if (early_termination and i > self.time_limit):
                    self.env.close()

                    return cumulative, None

                i += 1


def fitness_eval_parallel(pool, r_gen, eval_num, early_termination=True):#, controller_parameters):

    return pool.apply_async(r_gen.do_rollout, args=(False, eval_num, early_termination) )


class T1Solution(Solution):
    '''
    Solution for the T1 function.

    multi = flag to switch multiprocessing on or off
    '''
    def __init__(self, device, time_limit, inno_setting, multi=True):
        '''
        Constructor.
        '''
        Solution.__init__(self, 2)

        global gen_counter
        self.id = gen_counter
        gen_counter += 1

        self.parent_id  = -1

        self.device = device
        self.time_limit = time_limit
        self.multi = multi
        self.random_seed = [] #For compression of networks

        self.mutation_power = 0.01 #0.01 worked well 
            
        self.inno_setting = inno_setting

        self.r_gen = RolloutGenerator(device, time_limit)

        self.age = 0

        self.async_results = []
        self.calculated_results = {}

    def run_solution(self, pool, evals=5, early_termination=True, force_eval=False):
        if force_eval:
            self.calculated_results.pop(evals, None)

        if (evals in self.calculated_results.keys()): #Already caculated results
            return

        self.async_results = []

        for i in range(evals):
            if self.multi:

                self.async_results.append (fitness_eval_parallel(pool, self.r_gen, i, early_termination))#, self.controller_parameters) )
            else:
                self.async_results.append (self.r_gen.do_rollout ( False, i, False) ) 

            # self.r_gen.rollout(flatten_parameters ( self.parameters) )  )#fitness_eval_parallel(self.pNet, self.env_name) )

    def evaluate_solution(self, evals):
        '''
        Implementation of method evaluate_solution() for T1 function.
        '''
      
        if (evals in self.calculated_results.keys()): #Already calculated?
            mean_fitness, std_fitness = self.calculated_results[evals]
        else:
            if self.multi:
                results = [t.get()[0] for t in self.async_results]
                #print(results)
            else:

                results = [t[0] for t in self.async_results]
                print(results)

            mean_fitness = np.mean ( results )
            std_fitness = np.std( results )

            self.calculated_results[evals] = (mean_fitness, std_fitness)

        fr = 0
        self.objectives[0] = self.age 

        self.objectives[1] = -mean_fitness

        return mean_fitness, fr, std_fitness#, self.objectives[0], self.objectives[1]
        

    def load_solution(self, filename):

        s = torch.load(filename)#  pickle.load( open( filename, "rb" ) )
        #print(filename, s)
        self.r_gen.vae.load_state_dict( s['vae'])
        self.r_gen.controller.load_state_dict( s['controller'])
        self.r_gen.mdrnn.load_state_dict( s['mdrnn'])

  
    def crossover(self, other):
        child_solution = T1Solution(self.device, self.time_limit, self.inno_setting, multi=True)
        child_solution.age = self.age
        child_solution.parent_id = self.id #just for tracking purposes

        child_solution.multi = self.multi
        #child_solution.calculated_results = self.calculated_results
        child_solution.objectives[0] = self.objectives[0]
        child_solution.objectives[1] = self.objectives[1]
        child_solution.random_seed = self.random_seed.copy()

        if (other != None):
            #Currently no crossover implemented
            exit()
        else:
            child_solution.r_gen.controller = copy.deepcopy (self.r_gen.controller)
            child_solution.r_gen.vae = copy.deepcopy (self.r_gen.vae)
            child_solution.r_gen.mdrnn = copy.deepcopy (self.r_gen.mdrnn)
            

        return child_solution
    
    def mutate_params(self, params):

        for key in params: 
               #mask = np.random.binomial(1, 0.1,  params[key].size())
               params[key] += torch.from_numpy( np.random.normal(0, 1, params[key].size()) * self.mutation_power).float()
               #params[key] += torch.from_numpy( np.random.normal(0, 1, params[key].size()) * self.mutation_power * mask).float()

    def increase_age(self):
        self.age += 1

    def mutate(self):

        if (self.inno_setting==1):

            #Protect innovication with DIP by reseeting age
            c = np.random.randint(0,3)

            if c==0:
                self.mutate_params(self.r_gen.vae.state_dict())
                self.age = 0
            elif c==1:
                self.mutate_params(self.r_gen.mdrnn.state_dict() )
                self.age = 0
            else:
                self.mutate_params(self.r_gen.controller.state_dict())
        else:
            c = np.random.randint(0,3)

            if c==0:
                self.mutate_params(self.r_gen.vae.state_dict())
            elif c==1:
                self.mutate_params(self.r_gen.mdrnn.state_dict() )
            else:
                self.mutate_params(self.r_gen.controller.state_dict())


