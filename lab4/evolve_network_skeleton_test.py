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
import pandas as pd
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
        #print("about to set weights")
        self.i_h = weights[:self.layers[0]*self.layers[1]].reshape(self.layers[0], self.layers[1])
        self.h_o = weights[self.layers[0]*self.layers[1]:].reshape(self.layers[1], self.layers[2])
        #print("shape of i_h: ", self.i_h.shape, ", shape of h_o: ", self.h_o.shape, ", size of weights: ", weights.shape)

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
        max_iter = 1000
        for i in range(100):
            reward = 0
            iter = 0

            #if(iter >= max_iter):
            #    break

            obs = env.reset()
            done = False
            while not done:
                action = np.argmax(self.net.forward_pass(obs))
                obs, tmp_reward, done, info = env.step(action)
                reward += tmp_reward
                #if done:
                #    print("Episode finished after {} timesteps".format(ep+1))
                #    break
                iter += 1
            scores.append(reward)
        return np.mean(scores)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CS 420/CS 527: Neuroevolution")
    parser.add_argument("--environment", type=str, help="OpenAI Gym Environment")
    parser.add_argument("--filename", type=str, help="Filename of best genome")
    args = parser.parse_args() 

    df = pd.read_csv(args.filename)
    df["genome"] = pd.eval(df["genome"])
    df["genome"] = df["genome"].map(np.array)
    genome = df['genome'].values[0]
    num_inputs = df['num_inputs'].values[0]
    num_hidden = df['num_hidden'].values[0]
    num_actions = df['num_actions'].values[0]
    trn_size = df['trn_size'].values[0]
    trial = df['trial'].values[0]
    layers = [num_inputs, num_hidden, num_actions]
    
    # Calculate the total length of the genome
    total_weights = 0
    for i in range(len(layers)-1):
        total_weights += layers[i]*layers[i+1]

    prob = OpenAIGymProblem(layers, args.environment)
    
    fitness = prob.evaluate(genome)

    with open('LunarLander-v2/{}_{}_{}_{}_{}_test.csv'.format(num_inputs, num_hidden, num_actions, trn_size, trial), 'w') as csvfile:
        writ = csv.writer(csvfile)
        writ.writerow(["num_inputs", "num_hidden", "num_actions", "trn_size", "trial", "genome", "fitness"])
        writ.writerow([num_inputs, num_hidden, num_actions, trn_size, trial, genome.tolist(), fitness])