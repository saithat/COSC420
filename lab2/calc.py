from glob import glob
import pandas as pd
import numpy as np
import ast

def calc_diversity(div_input):
    pop = div_input
    dist_mat = np.zeros((len(pop), len(pop)))
    for i in range(len(pop)):
        for j in range(len(pop)):
            dist_mat[i, j] = np.linalg.norm(np.array(pop[i]).astype(float)-np.array(pop[j]).astype(float))

    avg_ind_diversity = dist_mat.mean(axis=1)
    avg_pop_diversity = avg_ind_diversity.mean()

    return avg_ind_diversity, avg_pop_diversity

rows_list = []

for file in glob('lab2_data/*.csv'):
    print(file)
    file_split = file.split("_")
    file_split[-1] = file_split[-1].split(".")[0]
    file_split = file_split[2:]

    pop_size = file_split[0]
    mut_prob = file_split[1]
    prob_unif_cross = file_split[2]
    tourn_size = file_split[3]
    iteration = file_split[4]

    tmp = pd.read_csv(file)
    tmp = tmp[tmp['step'] != 'step']
    tmp['fitness'] = pd.to_numeric(tmp['fitness'])
    tmp['step'] = pd.to_numeric(tmp['step'])
    tmp['genome'] = tmp['genome'].apply(lambda x: ast.literal_eval(x.replace('\n', "").replace(' ', ",")))
    gens = int(tmp['step'].max()) + 1
    
    for i in range(gens):
        tmp_gen = tmp[tmp['step'] == i].reset_index()
        #print(tmp_gen)
        avg_fit = tmp_gen['fitness'].mean()
        #print(tmp_gen['fitness'].idxmax())
        max_row = tmp_gen.iloc[tmp_gen['fitness'].idxmax()]
        best_fit = max_row['fitness']
        best_genome = max_row['genome']
        sol_found = 0
        num_sol_found = 0
        if (max_row['fitness'] == 1):
            sol_found = 1
            num_sol_found = tmp[tmp['fitness'] == 1].shape[0]
        avg_ind, avg_pop = calc_diversity(tmp_gen['genome'].values)

        row = {'N' : pop_size,'p_m' : mut_prob ,'p_c':prob_unif_cross,'tournament_size':tourn_size,
        'iteration':iteration,'generation':i,'average_fitness':avg_fit,'best_fitness':best_fit,
        'best_genome':best_genome,'solution_found':sol_found,'num_solutions_found':num_sol_found,
        'diversity_metric': avg_pop}

        rows_list.append(row)
        """
        df = df.append(row, ignore_index=True)

        if run==0:
            df.to_csv("results_test.csv", index=False)
        else:
            df.to_csv("results_test.csv", header=None, index=False, mode="a")
        
        run += 1"""

df = pd.DataFrame(rows_list)
df.to_csv("results_v2.csv", index=False)
