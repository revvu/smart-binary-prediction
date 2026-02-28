from tqdm import tqdm
import numpy as np
import time
import math
from algorithms import *
from graphing import *
from uv_plane import *
import line_profiler 

def Blackwells_action(uv):
    u, v = uv[0], uv[1]
    if u <= 0.5 and v >= u:
        return 0
    elif u > 0.5 and v >= (1 - u):
        return 1
    return (u-v)/(1-2*v)

def truncate_uv(split):
    return lambda uv: tuple(map(lambda x: int(x*split)/split, uv)) if split else uv

def Blackwell_regret(t, uv, n, split, memo={}):
    memo_key = (t, uv, n)
    truncate = truncate_uv(split)
    if memo_key in memo:
        return memo[memo_key]
    
    if t == n:
        regret = n*max(uv[0]-uv[1], 1-uv[0]-uv[1])
        memo[memo_key] = (regret, [])
    else:
        u, v = uv
        eps = 1 / (t+1)
        p = Blackwells_action(uv)
        
        truncate_and_regret = lambda c: Blackwell_regret(t+1, truncate(c), n, split, memo)

        coords = [((1-eps)*u, (1-eps)*v+(1-p)*eps), ((1-eps)*u+eps, (1-eps)*v+p*eps)]
        # prune coords outside of interesting region
        def out_of_region(uv):
            (u,v) = uv 
            prune_eps = 0.01 # buffer zone
            return (u <= 0.5 and v >= u+prune_eps) or (u > 0.5 and v >= (1-u)+prune_eps)
        
        # don't prune on the first 3
        if t<=3:
            reg1, reg2 = map(truncate_and_regret, coords)
            memo[memo_key] = (reg1[0], [0]+reg1[1]) if reg1[0] >= reg2[0] else (reg2[0], [1]+reg2[1])
        else:
            # if both out of region, make this 0 regret and ignore this branch
            point1, point2 = coords[0], coords[1]
            if out_of_region(point1) and out_of_region(point2):
                memo[memo_key] = (-1, [])
            elif out_of_region(point1):
                reg2 = truncate_and_regret(coords[1])
                memo[memo_key] = (reg2[0], [1]+reg2[1])
            elif out_of_region(point2):
                reg1 = truncate_and_regret(coords[0])
                memo[memo_key] = (reg1[0], [0]+reg1[1])
            else:
                reg1, reg2 = map(truncate_and_regret, coords)
                memo[memo_key] = (reg1[0], [0]+reg1[1]) if reg1[0] >= reg2[0] else (reg2[0], [1]+reg2[1])

    return memo[memo_key]

n = 32
split = None
#split = 100000
start_time = time.process_time()
regret, seq = Blackwell_regret(0, (0.5, 0), n, split)

print(f"""n: {n}
Discretization: {split}
Runtime: {time.process_time()-start_time:.4f} secs
""")

seq, Blackwells_seq = np.array(seq), Blackwells(seq)
print(f"""Approx Regret: {regret:.4f}
True Regret: {calc_regret(seq, Blackwells_seq):.4f}
Normalized Regret: {calc_regret(seq, Blackwells_seq) / n:.4f}
Regret / sqrt(n): {calc_regret(seq, Blackwells_seq) / np.sqrt(n):.4f}
Loss: {calc_loss(seq, Blackwells_seq):.4f}
Best Expert Loss: {calc_best_binary_expert_loss(seq)}

Sequence: {seq}
Actions: {Blackwells_seq}
""")


plot_walk(seq)
run_animation("Blackwell on Blackwells Worst Case", calc_uv(seq, Blackwells_seq), speed=10)
# run_animation("FTL on Blackwells Worst Case", calc_uv(seq, FTL_binary(seq)))
# run_animation("AdaHedge on Blackwells Worst Case", calc_uv(seq, adahedge(seq)))