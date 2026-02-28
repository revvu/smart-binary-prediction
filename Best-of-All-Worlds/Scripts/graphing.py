from Scripts.algorithms import *
from Scripts.sequence_generation import *
import numpy as np
from tqdm import tqdm
# Configuring matplotlib for our plots
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (8,8)
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams.update({'font.size': 52})
#plt.style.use('dark_background')
plt.style.use('default')
cmap = plt.get_cmap('Set2')
#cmap = plt.get_cmap('tab10')

from Scripts.uv_plane import *

def meta_plot(action_names, funcs, c_list, seq_list, xlabel, plot_title, ylabel=None, replications=1, best_expert_loss_list=None):
  num_points = len(c_list)
  regs = {name: np.zeros(num_points) for name in action_names}
  regs_CI = {name: [np.zeros(num_points), np.zeros(num_points)] for name in action_names}

  # check that there are sufficient sequences
  assert len(seq_list) >= num_points*replications

  # run every action on every sequence
  for i in tqdm(range(num_points)):

    for action_name, func in zip(action_names, funcs):

      # run replications
      replication_reg = np.zeros(replications)
      for j in range(replications):
        if best_expert_loss_list is not None:
          replication_reg[j] = calc_loss(seq_list[i*replications+j], func(seq_list[i*replications+j]))-best_expert_loss_list[i*replications+j]
        else:
          replication_reg[j] = calc_regret(seq_list[i*replications+j], func(seq_list[i*replications+j]))

      # update the main sequence and set bounds for 95% confidence interval
      regs[action_name][i] = np.mean(replication_reg)
      regs_CI[action_name][0][i] = regs[action_name][i]-2*np.std(replication_reg)/np.sqrt(replications)
      regs_CI[action_name][1][i] = regs[action_name][i]+2*np.std(replication_reg)/np.sqrt(replications)


  plt.figure(figsize=[8, 8])

  for i in range(len(action_names)):
    plt.plot(c_list, regs[action_names[i]], '.-', label="Reg$("+action_names[i]+")$", linewidth=1, color=cmap(i))
    if replications > 1:
      plt.fill_between(c_list, regs_CI[action_names[i]][0], regs_CI[action_names[i]][1], color=cmap(i), alpha=0.2)

  plt.xlabel(xlabel)
  if ylabel is not None: plt.ylabel(ylabel)
  else: plt.ylabel("Regret")
  plt.title(plot_title)
  plt.legend(loc=2)
  plt.show()

def plot_walk(sequence, target_function=None):
    n = len(sequence)
    walk = [0]
    for step in sequence:
        walk.append(walk[-1] + (1 if step == 1 else -1))
    
    t_values = list(range(len(sequence) + 1))
    plt.plot(t_values, walk, label='Walk Sequence')

    if target_function:
      target_values = [target_function(t,n) for t in t_values]
      plt.plot(t_values, target_values, label='Target Function', linestyle='--')
    
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Walk Sequence and Target Function')
    plt.legend()
    plt.show()

def titled_plot_walk(sequence, target_function=None):
    n = len(sequence)
    walk = [0]
    for step in sequence:
        walk.append(walk[-1] + (1 if step == 1 else -1))
    
    t_values = list(range(len(sequence) + 1))
    plt.plot(t_values, walk, label='True Loss Difference', alpha=0.5)

    if target_function:
      target_values = [target_function(t,n) for t in t_values]
      plt.plot(t_values, target_values, label='Target Loss Difference', linestyle='--')
    
    plt.xlabel('Length of sequence ($n$)')
    plt.ylabel('Difference in Loss Accrued between Experts')
    plt.title('Generating Loss $n^{0.4}$ Sequences')
    plt.legend()
    plt.show()

def main():
  # action_names = ["FTL","SMART","Cover", "RandomizedSMART", "AdaHedge", "ABProd", "Blackwell's", "Improving AdaHedge"]
  # funcs = [FTL_binary, SMART_det_binary, Cover_binary, SMART_random_binary, adahedge, ABprod, Blackwells, improving_adahedge]
  # action_names = ["FTL","Cover", "AdaHedge", "Blackwell's", "Improving FTL", "Improving Cover", "Improving AdaHedge", "Improving Blackwell's"]
  # funcs = [FTL_binary, Cover_binary, adahedge, Blackwells, improving_FTL, improving_Cover, improving_adahedge, improving_blackwells]
  
  # n = 10000
  # c = list(range(0, 100, 10))

  # seq_list = np.zeros((len(c), n))
  # for i in range(len(c)):
  #   seq_list[i] = generate_highlossFMG_ftl(n, c[i], 0.3)

  # meta_plot(action_names, funcs, c, seq_list, "Number of zero crossings", "Binary prediction on high-loss FMG sequences with $n=$"+str(n))

  # n = 10000
  # c = list(range(0, 200, 10))

  # seq_list = np.zeros((len(c), n))
  # for i in range(len(c)):
  #   seq_list[i] = generate_FMG_sequence(n, c[i])

  # meta_plot(action_names, funcs, c, seq_list, "Number of zero crossings (i.e., $|\{t\in[n]\,:\sum_{s\leq t}X_s = t/2$)}|)", "Binary prediction on FMG sequences with $n=$"+str(n))
  
  # n = 500
  # replications = 10
  # p_list = np.linspace(0.2, 0.5, 10)

  # seq_list = []
  # for p in p_list:
  #   for i in range(replications):
  #     seq_list.append(generate_iid_sequence(n, p))

  # meta_plot(action_names, funcs, p_list, seq_list, "Probability of Generating 1", "Binary prediction on i.i.d. Bernoulli sequences with $n=500$", replications=replications)
  # seq = generate_FMG_sequence(n=100, c=50)
  # run_animation("Blackwell on FMG n=100 c=50", calc_uv(seq, Blackwells(seq)))
  # run_animation("FTL on FMG n=100 c=50", calc_uv(seq, FTL_binary(seq)))
  # run_animation("AdaHedge on FMG n=100 c=50", calc_uv(seq, adahedge(seq)))
  # run_animation("Cover on on FMG n=100 c=50", calc_uv(seq, Cover_binary(seq)))

  # seq = generate_iid_sequence(n=100, p=0.3)
  # run_animation("Blackwell on iid n=100 p=0.3", calc_uv(seq, Blackwells(seq)))
  # run_animation("FTL on iid n=100 p=0.3", calc_uv(seq, FTL_binary(seq)))
  # run_animation("AdaHedge on iid n=100 p=0.3", calc_uv(seq, adahedge(seq)))
  # run_animation("Cover on on iid n=100 p=0.3", calc_uv(seq, Cover_binary(seq)))

  # n = 10000
  # n_list = list(range(10, n, 200))

  # seq_list = [0]*len(n_list)
  # for i in range(len(n_list)):
  #   seq_list[i] = loss_based_sequence_generation(i, lambda x: pow(x, 0.4))

  # meta_plot(action_names, funcs, n_list, seq_list, "Length of sequence", "t^0.4 sequences")

  # n = 10000
  # n_list = list(range(10, n, 200))

  # seq_list = [0]*len(n_list)
  # for i in range(len(n_list)):
  #   seq_list[i] = loss_based_sequence_generation(i, lambda x: pow(x, 0.5)/(np.log(x)))

  # meta_plot(action_names, funcs, n_list, seq_list, "Length of sequence", "sqrt(t)/log(t) sequences")

  # n = 1000
  # n_list = list(range(10, n, 50))

  # # target_loss_function = lambda x: pow(x, 0.4)
  # # lambda t,n to function
  target_loss_function_dict = {
    "Loss $t^{0.3}$": lambda t,n : pow(t, 0.3),
    "Loss $t^{0.4}$": lambda t,n : pow(t, 0.4),
    "Loss $t^{0.5}$": lambda t,n : pow(t, 0.5),
    #"Loss $n^{0.25} * sin(2 \pi \\frac{t}{\sqrt{n}} )$" : lambda t,n: pow(n, 0.25)*np.sin(2*np.pi*t/(np.sqrt(n))),
    #"Loss Constant $n^{0.25}$": lambda t,n : pow(n, 0.25),
    #"Loss ${(n-t)}^{0.25} * sin(2 \pi \\frac{t}{\sqrt{n}} )$" : lambda t,n: pow(n-t, 0.25)*np.sin(2*np.pi*t/(np.sqrt(n))),
    #"Loss ${n}^{0.25}-{t}^{0.25} * sin(2 \pi \\frac{t}{\sqrt{n}} )$" : lambda t,n: (pow(n,0.25)-pow(t, 0.25))*np.sin(2*np.pi*t/(np.sqrt(n)))
  }

  # action_names = ["Cover", "AdaHedge", "SMART", "FlipFlop", "Blackwell"]
  # funcs = [Cover_binary, adahedge, SMART_random_binary, flipflop, Blackwells]

  # action_names = ["AdaHedge", "FlipFlop", "FTL"]
  # funcs = [adahedge, flipflop, FTL_binary]


  # for loss_name, target_loss_function in target_loss_function_dict.items(): 
  #   seq_list = [0]*len(n_list)
  #   for i in range(len(n_list)):
  #     seq_list[i] = loss_based_sequence_generation(n_list[i], target_loss_function)
  #   plot_walk(seq_list[-1], target_loss_function)
  #   meta_plot(action_names, funcs, n_list, seq_list, "Length of sequence", loss_name+" sequences")
  
  n = 1000
  n_list = list(range(10, n, 50))
  seq = loss_based_sequence_generation(1000, target_loss_function=target_loss_function_dict['Loss $t^{0.4}$'])
  plot_walk(seq, target_loss_function=target_loss_function_dict['Loss $t^{0.4}$'])

if __name__ == '__main__': main()