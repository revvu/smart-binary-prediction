from Scripts.graphing import *


def main():
  target_loss_function_dict = {
    "Loss $t^{0.3}$": lambda t,n : pow(t, 0.3),
    "Loss $n^{0.4}$": lambda t,n : pow(t, 0.4),
    "Loss $t^{0.5}$": lambda t,n : pow(t, 0.5),
    #"Loss $n^{0.25} * sin(2 \pi \\frac{t}{\sqrt{n}} )$" : lambda t,n: pow(n, 0.25)*np.sin(2*np.pi*t/(np.sqrt(n))),
    #"Loss Constant $n^{0.25}$": lambda t,n : pow(n, 0.25),
    #"Loss ${(n-t)}^{0.25} * sin(2 \pi \\frac{t}{\sqrt{n}} )$" : lambda t,n: pow(n-t, 0.25)*np.sin(2*np.pi*t/(np.sqrt(n))),
    #"Loss ${n}^{0.25}-{t}^{0.25} * sin(2 \pi \\frac{t}{\sqrt{n}} )$" : lambda t,n: (pow(n,0.25)-pow(t, 0.25))*np.sin(2*np.pi*t/(np.sqrt(n)))
  }
    
  n = 1000
  n_list = list(range(10, n, 50))
  seq = loss_based_sequence_generation(1000, target_loss_function=target_loss_function_dict['Loss $n^{0.4}$'])
  plt.rcParams.update({'font.size': 12})
  titled_plot_walk(seq, target_function=target_loss_function_dict['Loss $n^{0.4}$'])

if __name__ == '__main__': main()