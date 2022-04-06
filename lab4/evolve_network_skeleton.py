# CS 420/CS 527 Lab 4: Neuroevolution with LEAP
# Catherine Schuman
# March 2022
import os
import sys
import gym
import matplotlib.pyplot as plt
import numpy as np
import argparse
from toolz import pipe
from leap_ec import Individual, Representation, test_env_var
from leap_ec import probe, ops, util
from leap_ec.algorithm import generational_ea
from leap_ec.real_rep.initializers import create_real_vector
from leap_ec.real_rep.ops import mutate_gaussian
from leap_ec.binary_rep.problems import ScalarProblem
from leap_ec.decoder import IdentityDecoder
from distributed import Client
from leap_ec.distrib import DistributedIndividual
from leap_ec.distrib import synchronous
import csv

class Network:
    # The network constructor takes as input an array where
    # layers[0] is the number of neurons in the first (input) layer
    # layers[1] is the number of neurons in the hidden layer
    # layers[2] is the number of neurons in the output layer 
    def __init__(self, layers):
        self.layers = layers
        self.i_h = 0
        self.h_0 = 0
   
    # TODO: This function will take as input a list of the weight values and 
    #       setup the weights in the network based on that list
    def set_weights(self, weights):
        # TODO: Complete this function
        self.i_h = weights[:self.layers[0]*self.layers[1]].reshape(self.layers[0], self.layers[1])
        self.h_o = weights[self.layers[0]*self.layers[1]:].reshape(self.layers[1], self.layers[2])

    # TODO: This network will take as input the observation and it will
    #       calculate the forward pass of the network with that input value
    #       It should return the output vector
    def forward_pass(self, obs):
        # TODO: Complete this function
        return np.dot(np.dot(obs, self.i_h), self.h_o)
    
# Implementation of a custom problem
class OpenAIGymProblem(ScalarProblem):
    def __init__(self, layers, env_name):
        super().__init__(maximize=True)
        self.layers = layers
        self.env_name = env_name
        self.net = Network(layers)

    # TODO: Implement the evaluate function as described in the write-up
    def evaluate(self, ind):
        # TODO: Complete this function
        self.net.set_weights(ind)
        env = gym.make(self.env_name)
        scores = []
        for i in range(5):
            reward = 0
            obs = env.reset()
            done = False
            while not done:
                action = np.argmax(self.net.forward_pass(obs))
                obs, tmp_reward, done, info = env.step(action)
                reward += tmp_reward
            scores.append(reward)
        return np.mean(scores)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CS 420/CS 527: Neuroevolution")
    parser.add_argument("--environment", type=str, help="OpenAI Gym Environment")
    parser.add_argument("--inputs", type=int, help="Number of inputs")
    parser.add_argument("--hidden", type=int, help="Number of hidden")
    parser.add_argument("--outputs", type=int, help="Number of outputs")
    parser.add_argument("--trial", type=int, default=0, help="Trial number")
    parser.add_argument("--trn_size", type=int, default=5, help="Tournament size")
    args = parser.parse_args()
    max_generation = 50
    N = 100

    num_inputs = args.inputs 
    num_actions = args.outputs
    num_hidden = args.hidden
    trn_size = args.trn_size
    layers = [num_inputs, num_hidden, num_actions]

    
    # Calculate the total length of the genome
    total_weights = 0
    for i in range(len(layers)-1):
        total_weights += layers[i]*layers[i+1]

    # Spin up Dask for distributed evaluation

    print("layers: ", layers)
    print("trial: ", args.trial)

    with Client() as client:
   
        # Set up the parents 
        parents = DistributedIndividual.create_population(N,
                                           initialize=create_real_vector(bounds=([[-1, 1]]*total_weights)),
                                           decoder=IdentityDecoder(),
                                           problem=OpenAIGymProblem(layers, args.environment))

        # Calculate initial fitness values for the parents
        parents = synchronous.eval_population(parents, client=client)

        # Loop over generations
        for current_generation in range(max_generation):
            offspring = pipe(parents,
                         ops.tournament_selection(k=5),
                         ops.clone,
                           mutate_gaussian(std=0.05, hard_bounds=(-1, 1), expected_num_mutations=int(0.01*total_weights)),
                         ops.uniform_crossover,
                         synchronous.eval_pool(client=client, size=len(parents)))

            fitnesses = [net.fitness for net in offspring]
            print("Generation ", current_generation, "Max Fitness ", max(fitnesses))
            parents = offspring

    # Find the best network in the final population
    index = np.argmax(fitnesses)
    best_net = parents[index]
    
    # TODO: You may want to change how you save the best network
    print("Best network weights:") 
    print(best_net.genome)

    with open('LunarLander-v2/{}_{}_{}_{}_{}.csv'.format(num_inputs, num_hidden, num_actions, trn_size, args.trial), 'w') as csvfile:
        writ = csv.writer(csvfile)
        writ.writerow(["num_inputs", "num_hidden", "num_actions", "trn_size", "trial", "genome", "fitness"])
        writ.writerow([num_inputs, num_hidden, num_actions, trn_size, args.trial, best_net.genome.tolist(), best_net.fitness])