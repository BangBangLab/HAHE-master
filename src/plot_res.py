import os
import numpy as np 
import re

from torch import int64 


def plot_res(res_path, eval_res_dim, max_len, interval):
    with open(res_path, "r") as res_file:
        steps_list = np.zeros(shape=(max_len), dtype=np.int)
        lr_list = np.zeros(shape=(max_len), dtype=np.float)
        loss_list = np.zeros(shape=(max_len), dtype=np.float)
        best_loss_list = np.zeros(shape=(max_len), dtype=np.float)
        average_loss_list = np.zeros(shape=(max_len), dtype=np.float)
        ht_list = np.zeros(shape=(max_len, eval_res_dim), dtype=np.float)
        pr_list = np.zeros(shape=(max_len, eval_res_dim), dtype=np.float)
        ent_list = np.zeros(shape=(max_len, eval_res_dim), dtype=np.float)
        rel_list = np.zeros(shape=(max_len, eval_res_dim), dtype=np.float)
        steps = 0
        iter = 0
        eval_iter = 0
        for line in res_file:
            if iter>=max_len:
                break
            elif line.startswith("Epoch"):
                data = line.split()
                steps = int(data[data.index("Steps")+1])
                if steps==steps_list[-1] or steps%interval!=0:
                    continue
                steps_list[iter] = steps
                lr_list[iter] = float(data[data.index("lr:")+1])
                loss_list[iter] = float(data[data.index("loss:")+1])
                best_loss_list[iter] = float(data[data.index("best_loss:")+1])
                average_loss_list[iter] = float(data[data.index("average_loss:")+1])
                iter += 1
            elif line.startswith("HEAD/TAIL"):
                data = line.split()
                ht_list[iter, :] = data[1:]
                eval_iter += 1
            elif line.startswith("PRIMARY_R"):
                data = line.split()
                pr_list[iter, :] = data[1:]
            elif line.startswith("ENTITY"):
                data = line.split()
                ent_list[iter, :] = data[1:]
            elif line.startswith("RELATION"):
                data = line.split()
                rel_list[iter, :] = data[1:]
            else:
                continue
    if iter < max_len:
        steps_list = steps_list[:iter]
        lr_list = lr_list[:iter]
        loss_list = loss_list[:iter]
        best_loss_list = best_loss_list[:iter]
        average_loss_list = average_loss_list[:iter]
        ht_list = ht_list[:eval_iter]
        pr_list = pr_list[:eval_iter]
        ent_list = ent_list[:eval_iter]
        rel_list = rel_list[:eval_iter]   

if __name__ == '__main__':
    res_dir = "./"
    eval_res_dim = 5
    max_len = 5000
    interval = 1000
    for res_filename in os.listdir(res_dir):
        if res_filename.endswith(".txt"):
            res_path = os.path.join(res_dir, res_filename)
            plot_res(res_path, eval_res_dim, max_len, interval)


