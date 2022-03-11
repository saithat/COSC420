import os
import sys

N = [25, 50, 75, 100]
p_m = [0, 0.01, 0.03, 0.05]
p_c = [0, 0.1, 0.3, 0.5]
trn_size = [2, 3, 4, 5]

for n in N:
    for pm in p_m:
        for pc in p_c:
            for ts in trn_size:
                for i in range(20):
                    command = "python lab2.py --n " + str(n) + " --p_m " + str(pm) + " --p_c " + str(pc) + " --trn_size " + str(ts) + " --csv_output lab2_data/results_" +str(n) + "_" + str(pm) + "_" + str(pc) + "_" + str(ts) + "_" + str(i) + ".csv"
                    os.system(command)
