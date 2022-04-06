import os

trials = 1
hidden_sizes = [10, 20]

for h in hidden_sizes:
    for t in range(1, trials+1):
        train_command = "python evolve_network_skeleton.py --environment='LunarLander-v2' --inputs=8 --hidden={} --outputs=4 --trial={}".format(h, t)
        test_command = "python evolve_network_skeleton_test.py --environment='LunarLander-v2' --filename='LunarLander-v2/8_{}_4_5_{}.csv'".format(h, t)

        os.system(train_command)
        os.system(test_command)