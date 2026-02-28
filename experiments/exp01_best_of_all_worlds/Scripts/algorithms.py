import numpy as np
import bisect
from scipy.stats import binom
from scipy.special import logsumexp

def calc_loss(seq, actions):
  actions = np.asarray([round(action, 10) for action in actions])
  assert all(actions >= 0) and all(actions <= 1)
  assert all(x in {0, 1} for x in seq)
  return sum((1-seq)*actions+seq*(1-actions))

def calc_best_binary_expert_loss(seq):
  assert all(x in {0, 1} for x in seq)
  return min(np.sum(seq == 0), np.sum(seq == 1))

def calc_regret(seq, actions, find_best_expert_loss=calc_best_binary_expert_loss):
  assert all(actions >= 0) and all(actions <= 1)
  assert all(x in {0, 1} for x in seq)
  return calc_loss(seq, actions) - find_best_expert_loss(seq)

def FTL_binary(seq):
  n = np.size(seq)
  S = np.append(0, np.cumsum(2*seq-1)[:-1])
  actions = 0.5+np.sign(S)/2
  return actions

def anti_FTL_binary(seq):
  actions = 1- FTL_binary(seq)
  return actions

def Cover_binary(seq):
  n = len(seq)
  y = np.append(0, np.cumsum(2*seq-1)[:-1])
  h = np.linspace(n-1,0,n)
  actions = 0.5*binom.pmf((y+h)/2.0,h,0.5) + binom.cdf((y+h-1)/2.0,h,0.5)

  return actions

def SMART_det_binary(seq):
  n = np.size(seq)
  S = np.append(0, np.cumsum(2*seq-1)[:-1])

  theta = np.sqrt(n/(2*np.pi))
  reg_ftl = np.cumsum(S==0)/2

  actions = np.zeros(n)
  ftl_actions = FTL_binary(seq)
  cover_actions = Cover_binary(seq)

  if reg_ftl[-1] < theta:
    # Play FTL
    actions = FTL_binary(seq)
  else:
    # Switch to Cover
    # index at which ftl regret exceeds theta
    tsw = bisect.bisect_left(reg_ftl, theta)
    actions = np.append(ftl_actions[:tsw], cover_actions[tsw:])
  return actions

def SMART_random_binary(seq):
  n = len(seq)
  gn = np.sqrt(n/(2*np.pi))
  # Convert to Rademachers and compute sum
  S = np.append(0, np.cumsum(2*seq[:-1]-1))
  reg_ftl = np.cumsum(S == 0) / 2
  reg_cov = np.zeros(n)
  for i in range(n): reg_cov[i] = np.sqrt((n-i)/(2*np.pi))
  reg_t = reg_ftl + reg_cov

  # Compute weight for playing FTL
  r = np.minimum(reg_ftl, gn)
  rshift = np.append(0.5,r[:-1])
  p_cov = (np.exp(r/gn)-1)/((np.e-1))
  p_ftl = 1 - p_cov

  actions = p_ftl*FTL_binary(seq) + p_cov*Cover_binary(seq)
  return actions

def Flip_simple_binary(seq):
    n = np.size(seq)
    S = np.cumsum(np.append(0, 2 * seq[:-1] - 1))

    actions_dict = [FTL_binary(seq), 0.5*np.ones(n)]
    actions = np.zeros(n)
    actions[0] = 0.5

    for i in range(1, n):
        actions[i] = actions_dict[int(np.abs(S[i])<3)][i]
    return actions

def randomized_Flip_simple_binary(seq):
  n = np.size(seq)
  S = np.cumsum(np.append(0, 2 * seq[:-1] - 1))

  actions_dict = [FTL_binary(seq), 0.5*np.ones(n)]
  actions = np.zeros(n)
  actions[0] = 0.5

  for i in range(1, n):
      actions[i] = actions_dict[int(np.abs(S[i])<np.random.uniform(2,4))][i]
  return actions

def SMART_Flip_simple_binary(seq):
    n = np.size(seq)
    S = np.cumsum(np.append(0, 2 * seq[:-1] - 1))
    theta = np.sqrt(n/(2*np.pi))
    Delta = 0.5

    actions_dict = [FTL_binary(seq), 0.5*np.ones(n)]

    actions = np.zeros(n)
    actions[0] = 0.5

    for i in range(1, n):
        if Delta > theta:
          return np.append(actions[:i], Cover_binary(seq[i:]))
        actions[i] = actions_dict[int(np.abs(S[i])<3)][i]
        Delta += (np.abs(S[i])==3)
    return actions

def Hedge(seq, eta=1):
  n = np.size(seq)
  seq = np.append(0.5, seq[:-1])
  # Loss of playing expert 1 and 0
  L = np.column_stack((np.cumsum(seq), np.cumsum(1-seq)))
  # logsum used to prevent overflows
  w = np.exp(-eta*L[:,1]-logsumexp(-eta*L, axis=1))
  return w

def adahedge(seq):
    n = np.size(seq)
    l = np.vstack([np.zeros(2), np.column_stack((seq, 1-seq))])
    L = np.vstack([np.zeros(2), np.column_stack((np.cumsum(seq), np.cumsum(1-seq)))])

    Delta = 0
    w = np.zeros(n)
    for t in range(n):
      # Hedged Play
      if Delta > 0:
        eta = np.log(2) / Delta
        w[t] = np.exp(-eta*L[t,1]-logsumexp(-eta*L[t,:]))
        mt = -logsumexp(-eta*l[t+1,:], b=[1-w[t],w[t]])/eta
      # FTL Play
      else:
        w[t] = 0.5*(1+np.sign(L[t,0]-L[t,1]))
        mt = np.min(L[t+1])-np.min(L[t])
      delta = np.dot([1-w[t],w[t]], l[t+1,:])-mt
      assert delta >= -0.1, f"{delta} Mixability Gap is not monotonic"
      delta = max(0, delta)
      Delta += delta
    return w

def flipflop(seq):
    n = np.size(seq)
    l = np.vstack([np.zeros(2), np.column_stack((seq, 1-seq))])
    L = np.vstack([np.zeros(2), np.column_stack((np.cumsum(seq), np.cumsum(1-seq)))])
    alpha, phi = 1.243, 2.37
    scale = np.array([phi / alpha, alpha])
    Delta = [0,0]
    regime = 0 # 0=FTL 1=AH

    w = np.zeros(n)
    for t in range(n):
      delta = 0
      # Hedged Play, check Delta[regime]>0 for AdaHedge chooses FTL play 
      if regime and Delta[regime] > 0:
        eta = np.log(2) / Delta[regime]
        w[t] = np.exp(-eta*L[t,1]-logsumexp(-eta*L[t,:]))
        mt = -logsumexp(-eta*l[t+1,:], b=[1-w[t],w[t]])/eta
      # FTL Play
      else:
        w[t] = 0.5*(1+np.sign(L[t,0]-L[t,1]))
        mt = np.min(L[t+1])-np.min(L[t])
      delta = np.dot([1-w[t],w[t]], l[t+1,:]) - mt
      assert delta >= -0.1, f"{delta} Mixability Gap is not monotonic"
      delta = max(0, delta)
      Delta[regime] += delta
      if Delta[regime] > scale[regime]*Delta[1-regime]:
        regime = 1-regime

    return w

def ABprod(seq):
  n = np.size(seq)
  C = np.sqrt(n) # an upper-bound on the total benchmark loss
  eta = 0.5*np.sqrt(np.log(C)/C)
  FTL_actions = FTL_binary(seq)
  Hedge_actions = adahedge(seq)
  calc_iterated_loss = lambda seq, actions: (1-seq)*actions+seq*(1-actions)
  delta = calc_iterated_loss(seq, FTL_actions)-calc_iterated_loss(seq, Hedge_actions)

  w = np.cumprod(np.append(eta, 1+eta*delta[:-1]))
  st = w/(w+1-eta)
  actions = st*FTL_actions + (1-st)*Hedge_actions
  return actions

def Blackwells(seq):
   n = np.size(seq)
   u = 0.5
   v = 0
   actions = np.zeros(n)
   for t in range(n):
      if u <= 0.5 and v >= u:
         actions[t] = 0
      elif u>0.5 and v>=(1-u):
         actions[t]=1
      else:
         actions[t] = (u-v)/(1-2*v)
        
      u = (t*u + seq[t])/(t+1)
      v = (t*v + [1-actions[t], actions[t]][int(seq[t])])/(t+1)
   return actions

# only play the new action if the new action would have less loss at t-1
def make_improving(seq, actions):
  n = len(seq)  
  for t in range(1, n):
    if calc_loss(seq[:t], np.ones(t)*actions[t-1]) < calc_loss(seq[:t], np.ones(t)*actions[t]):
      actions[t] = actions[t-1]
  return actions

def improving_FTL(seq):
   return make_improving(seq, FTL_binary(seq))

def improving_Cover(seq):
   return make_improving(seq, Cover_binary(seq))

def improving_adahedge(seq):
    return make_improving(seq, adahedge(seq))

def improving_blackwells(seq):
   return make_improving(seq, Blackwells(seq))