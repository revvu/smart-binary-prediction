import numpy as np
from tqdm import tqdm
from algorithms import *
from graphing import meta_plot
from scipy.special import logsumexp
from line_profiler import profile

# DP search over all attainable sequences. 
# Alg_params fully represents a sequence with respect to an algorithm in terms of play.
# Examples: For FTL, alg_params is (St). For FlipFlop, alg_params is (Delta1, Delta2). 
# Next action takes in alg_params and returns the next weight given those alg_params.
# Update alg_params takes in alg_params and the next observed bit, updating alg_params 
# represent the new sequence. 

# DP Loss Time complexity is O(n*len(alg_params))
def dp_search_loss(n, next_action, update_alg_params):
    # t, alg_params -> loss, seq
    dp_losses = {t: {} for t in range(-1, n)}
    dp_losses[-1][None] = (0, [])

    for t in tqdm(range(-1,n-1)):
        if t%(int(n/10)) == 0: print("t:", t, " num of alg_params:", len(dp_losses[t]))
        for alg_params, (prev_loss, prev_seq) in dp_losses[t].items():
            wt = next_action(alg_params)
            for next_bit in [0,1]:
                new_alg_params = update_alg_params(alg_params, next_bit)
                new_loss = wt*(1-next_bit)+(1-wt)*next_bit+prev_loss
                # new new_alg_params
                if new_alg_params not in dp_losses[t+1]:
                    dp_losses[t+1][new_alg_params] = (0, None)
                if new_loss > dp_losses[t+1][new_alg_params][0]:
                    dp_losses[t+1][new_alg_params] = (new_loss, prev_seq+[next_bit])
    
    return [max(dp_losses[t].values(), key=lambda x: x[0]) for t in range(n)]


def dp_search_regret(n, next_action, update_alg_params):
    # t, alg_params, St, loss -> regret, seq
    dp_regret = {t: {} for t in range(-1, n)}
    dp_regret[-1][None] = (0, [])

    for t in tqdm(range(-1,n-1)):
        if t%(int(n/10)) == 0: print("t:", t, " num of alg_params:", len(dp_regret[t]))
        for (alg_params, St, prev_loss), (prev_regret, prev_seq) in dp_regret[t].items():
            wt = next_action(alg_params)
            for next_bit in [0,1]:
                new_alg_params = update_alg_params(alg_params, next_bit)
                new_loss = wt*(1-next_bit)+(1-wt)*next_bit+prev_loss
                new_St = St+ 2*next_bit-1
                # new new_alg_params
                if (new_alg_params, new_St, new_loss) not in dp_regret[t+1]:
                    dp_regret[t+1][(new_alg_params, new_St, new_loss)] = (0, None)
                new_regret = new_loss-(t-abs(St))/2
                if new_regret > dp_regret[t+1][(new_alg_params, new_St, new_loss)][0]:
                    dp_regret[t+1][(new_alg_params, new_St, new_loss)] = (new_regret, prev_seq+[next_bit])
    
    return [max(dp_regret[t].values(), key=lambda x: x[0]) for t in range(n)]

################################################################################

# Next actions should take in alg_params and return the next weight to play. 
# Update alg_params should take in alg_params and the next observed bit, 
# updating alg_params to represent the new sequence.
# Alg_params must be hashable.


# For FTL: alg_params = sum of -1, 1 representation of seq (St)
def next_action_FTL(St):
    if St is None: return 0.5
    return (np.sign(St)+1)/2
def update_alg_params_FTL(St, next_bit):
    if St is None: return 2*next_bit-1
    return St+2*next_bit-1

# For Hedge: alg_params = Vector loss of experts (L)
def next_action_Hedge(L):
    if L is None: return 0.5
    # action is normalized e^L
    eta = 1 # adjust as needed
    return np.exp(-eta*L[:,1]-logsumexp(-eta*L, axis=1)) 
def update_alg_params_Hedge(L, next_bit):
    if L is None: return (next_bit, 1-next_bit)
    return (L+next_bit, L+1-next_bit)

# For AdaHedge: alg_params = Cumulative mixability gap (Delta), Vector loss of experts (L)
# In practice, runs with Time Complexity 2n^3
def next_action_AdaHedge(alg_params):
    if alg_params is None: return 0.5
    (Delta, L) = alg_params
    # Hedged Play
    if Delta > 0:
        eta = np.log(2) / Delta
        return np.exp(-eta*L[1]-logsumexp([-eta*L[0], -eta*L[1]]))
    # FTL play
    return [0,1][int(L[0]>L[1])]
def update_alg_params_AdaHedge(alg_params, next_bit):
    if alg_params is None: 
        Delta, L = 0, (0,0)
    else:
        (Delta, L) = alg_params
    l = np.asarray([next_bit, 1-next_bit])
    L = (L[0]+l[0], L[1]+l[1])
    #Hedged Play
    if Delta > 0:
        eta = np.log(2) / Delta
        new_w = np.exp(-eta*L[1]-logsumexp([-eta*L[0], -eta*L[1]]))
        Delta += logsumexp(-eta*l[:], b=[new_w,1-new_w])/eta
    # FTL Play
    else:
        new_w = [0,1][int(L[0]>L[1])]
    Delta += np.dot([new_w,1-new_w], l[:])
    # Truncate for speed
    Delta = round(Delta,2)
    L = (round(L[0],2), round(L[1],2))
    return (Delta, L)

# For FlipSimple: alg_params = St
def next_action_FlipSimple(St):
    if St is None: return 0.5
    return [0.5, (np.sign(St)+1)/2][abs(St)>=3]
def update_alg_params_FlipSimple(St, next_bit):
    return update_alg_params_FTL(St, next_bit)

###############################################################################
# DP Time Complexity is n * len(alg_params) * len(wt).

# DP Time Complexity is n * n * n = n^3.
# In practice this is very fast...wt might not need truncation.

# # DP Time Complexity is 
# def next_action_FlipFlop(alg_params, next_bit):
#     alpha, phi = 1.243, 2.37
#     scale = np.array([phi / alpha, alpha])

#     if alg_params==None: return ((0,0), (0,0), 0), 0.5
#     # alg_param keys: Delta vector, Loss vector, regime 0=FTL 1=AH
#     Delta, L, regime = alg_params
#     l = np.asarray([next_bit, 1-next_bit])
#     L = (L[0]+l[0], L[1]+l[1])

#     # Hedged Play
#     if regime and Delta[regime] > 0:
#         eta = np.log(2) / Delta[regime]
#         new_w = np.exp(-eta*L[1]-logsumexp([-eta*L[0], -eta*L[1]]))
#         Delta = (Delta[0], Delta[1]+logsumexp(-eta*l[:], b=[new_w,1-new_w])/eta)
#     # FTL Play
#     else:
#         new_w = [0,1][int(L[0]>L[1])]
#         if regime:
#             Delta = (Delta[0], Delta[1]+np.dot([new_w,1-new_w], l[:]))
#         else:
#             Delta = (Delta[0]+np.dot([new_w,1-new_w], l[:]), Delta[1])

#     if Delta[regime] > scale[regime]*Delta[1-regime]:
#         regime = 1-regime
#     return (Delta, L, regime), new_w

###############################################################################
def partition_function(value, n):
    log_n_squared = np.log(n) ** 2
    floor_sqrt_value_log_n = np.floor(np.sqrt(abs(value)) / np.log(n)) ** 2
    result = log_n_squared * floor_sqrt_value_log_n
    return -result if value < 0 else result


###############################################################################
action_names = ["Cover", "FlipSimple", "AdaHedge", "FlipFlop"]
funcs = [Cover_binary, Flip_simple_binary, adahedge, flipflop]

n = 100
worst_case_sequences = dp_search_loss(n, next_action=next_action_AdaHedge, update_alg_params=update_alg_params_AdaHedge)
print(worst_case_sequences[:10]) 
n_list = np.linspace(1, n, 15)
seq_list = []
for point in n_list:
    seq_list.append(np.asarray(worst_case_sequences[int(point)-1][1]))

meta_plot(action_names, funcs, n_list, seq_list, "Length of Sequence", "AdaHedge Worst Loss Sequences")
