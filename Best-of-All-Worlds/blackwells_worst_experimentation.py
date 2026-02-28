# Reevu Adakroy
# 7/29/2024

from Scripts.uv_plane import *
from Scripts.algorithms import *
from Scripts.sequence_generation import *
from Scripts.graphing import *

target_loss_function_dict = {
  "Loss $n^{0.25} * sin(2 \pi \\frac{t}{\sqrt{n}} )$" : lambda t,n: pow(n, 0.25)*np.sin(2*np.pi*t/(np.sqrt(n))),
  # "Loss ${(n-t)}^{0.25} * sin(2 \pi \\frac{t}{\sqrt{n}} )$" : lambda t,n: pow(n-t, 0.25)*np.sin(2*np.pi*t/(np.sqrt(n))),
  # "Loss ${n}^{0.25}-{t}^{0.25} * sin(2 \pi \\frac{t}{\sqrt{n}} )$" : lambda t,n: (pow(n,0.25)-pow(t, 0.25))*np.sin(2*np.pi*t/(np.sqrt(n))),
  "Loss $n^{0.15} * sin(2 \pi \\frac{t}{\sqrt{n}} )$" : lambda t,n: pow(n, 0.15)*np.sin(2*np.pi*t/(np.sqrt(n))),
  "Loss $n^{0.1} * sin(2 \pi \\frac{t}{\sqrt{n}} )$" : lambda t,n: pow(n, 0.1)*np.sin(2*np.pi*t/(np.sqrt(n))),
}

n = 1000
for loss_name, target_loss_function in target_loss_function_dict.items():
  seq = loss_based_sequence_generation(n, target_loss_function)
  plot_walk(seq, target_loss_function)
  run_animation("Blackwell on " + loss_name, calc_uv(seq, Blackwells(seq)), speed=10)