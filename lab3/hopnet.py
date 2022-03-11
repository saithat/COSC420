import numpy as np
import csv
import sys

class hopnet():
    def __init__(self, num_patterns, num_neurons) -> None:
        self.num_patterns = num_patterns
        self.num_neurons = num_neurons

    def generatePatterns(self):
        return np.random.choice([-1, 1], size=(self.num_patterns, self.num_neurons))

    def imprintPatterns(self, p, patterns):
        weights = np.zeros((self.num_neurons, self.num_neurons))
        for i in range(self.num_neurons): #pairs of neurons
            for j in range(self.num_neurons):
                sum = 0
                if i != j:
                    for k in range(p+1): # p from 1-p (patterns)
                        sum += patterns[k][i] * patterns[k][j]
                weights[i][j] = sum/self.num_neurons
        return weights

    def sign(self, x):
        if (x >= 0):
            return 1
        else:
            return -1

    def stableTest(self, p, patterns, weights):
        h = 0
        sprime = 0
        unstable = 0
        for k in range(p+1):
            neurons = patterns[k]
            for i in range(self.num_neurons):
                
                h = 0
                
                for j in range(self.num_neurons):
                    h += weights[i][j]*neurons[j]
                    
                sprime = self.sign(h)

                if(sprime != neurons[i]):
                    unstable += 1
                    break
                 
        return unstable


if __name__ == '__main__':

    num_runs = int(sys.argv[1])
    num_patterns = int(sys.argv[2])
    num_neurons = int(sys.argv[3])

    header = ['run', 'p', 'num_stable', 'frac_unstable']
    f = open('{}_{}_{}.csv'.format(num_runs, num_patterns, num_neurons), 'w')
    writer = csv.writer(f)
    writer.writerow(header)

    hnet = hopnet(num_patterns, num_neurons)

    for i in range(num_runs):
        patterns = hnet.generatePatterns()
        stable_imprints = np.zeros(num_patterns)
        frac_unstable = np.zeros(num_patterns)
        for p in range(num_patterns):
            weights = hnet.imprintPatterns(p, patterns)
            num_unstable = hnet.stableTest(p, patterns, weights)
            stable_imprints[p] = p+1-num_unstable
            frac_unstable[p] = num_unstable/(p+1)
            writer.writerow([i+1, p+1, stable_imprints[p], frac_unstable[p]])
        print("Run: {}, stable: {}, unstable_frac: {}".format(i+1, stable_imprints, frac_unstable))

    f.close()