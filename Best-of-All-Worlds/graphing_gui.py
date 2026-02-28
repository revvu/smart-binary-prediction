# https://github.com/revvu/CustomTkinter
import time
import tkinter
import customtkinter
from Scripts.sequence_generation import *
from Scripts.graphing import *
from Scripts.algorithms import *
from Scripts.classified_Hedge import *
plt.rcParams.update({'font.size': 16})
customtkinter.set_appearance_mode("Dark")
customtkinter.set_default_color_theme("green")

ALGS = [FTL_binary, SMART_det_binary, Cover_binary, SMART_random_binary, adahedge, flipflop, ABprod, Blackwells]
ALG_NAMES = ["FTL", "SMART", "Cover", "RandomizedSMART", "AdaHedge", "FlipFlop", "ABProd", "Blackwell's"]
LOSS_FUNCTIONS = {
  "Loss $t^{0.3}$": lambda t, n: t**0.3,
  "Loss $n^{0.4}$": lambda t, n: t**0.4,
  "Loss $t^{0.5}$": lambda t, n: t**0.5,
  "Loss $n^{0.15} * sin(2 \pi \\frac{t}{n^0.1} )$" : lambda t,n: pow(n, 0.15)*np.sin(2*np.pi*t/(np.pow(n, 0.1))),
}
IID_NAMES = ["0.3", "0.4", "0.5", "0.6"]
FMG_NAMES = ["FMG Sequence", "High-Loss FMG Sequence"]
CONTEXT_ALGS = [FTL_context_tree, Hedge_context_tree, AdaHedge_context_tree, SMART_context_tree]
CONTEXT_ALG_NAMES = ["FTL", "Hedge", "AdaHedge", "SMART"]

class App(customtkinter.CTk):
  def __init__(self):
    super().__init__()
    self.title("Sequential Bit Prediction Regret Visualizations")
    #self.geometry("1100x580")
    self.chosen_algs = [False] * len(ALG_NAMES)
    self.seq_gen_id = 0
    self.chosen_context_algs = [False] * len(CONTEXT_ALG_NAMES)
    self.context_degree_entry_list = [0] * len(CONTEXT_ALG_NAMES)

    self.setup_left_sidebar()
    self.setup_center()
    self.setup_right_sidebar()

  def setup_left_sidebar(self):
    sidebar = customtkinter.CTkFrame(self, width=140, corner_radius=0)
    sidebar.grid(row=0, column=0, rowspan=4, sticky="nsew", padx=20, pady=20)
    customtkinter.CTkLabel(sidebar, text="Two-Expert Methods", font=customtkinter.CTkFont(size=20, weight="bold")).grid(row=0, padx=20, pady=(20, 10))

    for i, name in enumerate(ALG_NAMES):
      customtkinter.CTkCheckBox(sidebar, text=name, command=lambda i=i: self.toggle_algorithm(i)).grid(row=i+1, pady=(20, 0), padx=20, sticky="n")

    customtkinter.CTkButton(sidebar, text="Graph", command=self.on_graph).grid(row=len(ALG_NAMES)+4, padx=20, pady=40)
    self.progressbar_1 = customtkinter.CTkProgressBar(sidebar)
    self.progressbar_1.grid(row=len(ALG_NAMES)+5)
    self.progressbar_1.set(1)

  def setup_center(self):
    center = customtkinter.CTkFrame(self, width=140, corner_radius=0)
    center.grid(row=0, column=1, rowspan=4, sticky="nsew", padx=20, pady=20)
    
    def add_radio_buttons(names, start_row, button_id_shift):
      for i, name in enumerate(names):
        customtkinter.CTkRadioButton(
          center, text=name, variable=self.radio_var, value=i+button_id_shift, command=lambda i=i+button_id_shift: self.set_seq_gen_id(i)
        ).grid(row=start_row + i, pady=10, padx=20, sticky="n")
      return start_row + len(names)
    
    next_row = 0
    self.radio_var = tkinter.IntVar(value=0)
    customtkinter.CTkLabel(center, text="iid Sequences (p=)", font=customtkinter.CTkFont(size=20, weight="bold")).grid(row=next_row, padx=20, pady=(20, 10))
    next_row = add_radio_buttons(IID_NAMES, next_row+1, button_id_shift=0)

    customtkinter.CTkLabel(center, text="Loss Function", font=customtkinter.CTkFont(size=20, weight="bold")).grid(row=next_row, padx=20, pady=(20, 10))
    next_row = add_radio_buttons(LOSS_FUNCTIONS.keys(), next_row+1, button_id_shift=len(IID_NAMES))

    customtkinter.CTkLabel(center, text="FMG Sequences", font=customtkinter.CTkFont(size=20, weight="bold")).grid(row=next_row, padx=20, pady=(20, 10))
    add_radio_buttons(FMG_NAMES, next_row+1, button_id_shift=len(IID_NAMES)+len(LOSS_FUNCTIONS))

  def setup_right_sidebar(self):
    sidebar = customtkinter.CTkFrame(self, width=140, corner_radius=0)
    sidebar.grid(row=0, column=2, rowspan=4, sticky="nsew", padx=20, pady=20)
    customtkinter.CTkLabel(sidebar, text="Context Tree Methods", font=customtkinter.CTkFont(size=20, weight="bold")).grid(row=0, padx=20, pady=(20, 10))

    for i, name in enumerate(CONTEXT_ALG_NAMES):
      customtkinter.CTkCheckBox(sidebar, text=name+" (enter degree below)", command=lambda i=i: self.toggle_context_algorithm(i)).grid(row=2*i+1, pady=(20, 0), padx=20, sticky="n")
      self.context_degree_entry_list[i] = customtkinter.CTkEntry(sidebar, placeholder_text="0, 1, 2, 4")
      self.context_degree_entry_list[i].grid(row=2*i+2, sticky="nsew", padx=80, pady=10)
    next_row = 2*len(CONTEXT_ALG_NAMES)+1
    
    customtkinter.CTkLabel(sidebar, text="Hindsight Expert Degree:", font=customtkinter.CTkFont(size=20, weight="bold")).grid(row=next_row, padx=20, pady=20)
    next_row += 1
    
    self.entry = customtkinter.CTkEntry(sidebar, placeholder_text="0")
    self.entry.grid(row=next_row, sticky="nsew", padx=80)

  def toggle_algorithm(self, index):
    self.chosen_algs[index] = not self.chosen_algs[index]
  
  def toggle_context_algorithm(self, index):
    self.chosen_context_algs[index] = not self.chosen_context_algs[index]

  def set_seq_gen_id(self, index):
    self.seq_gen_id = index

  def on_graph(self):
    self.progressbar_1.configure(mode="indeterminate")
    self.progressbar_1.start()
    self.graph_event()
    #time.sleep(10)
    self.progressbar_1.stop()
    self.progressbar_1.configure(mode="determinate")
    self.progressbar_1.set(1)

  def graph_event(self):
    selected_algs = [ALG_NAMES[i] for i, chosen in enumerate(self.chosen_algs) if chosen]
    funcs = [ALGS[i] for i, chosen in enumerate(self.chosen_algs) if chosen]

    selected_context_algs = []
    context_funcs = []
    for i,chosen in enumerate(self.chosen_context_algs):
      if chosen:
        try:
          degs = self.context_degree_entry_list[i].get().split(',')
          degs = [int(deg) for deg in degs]
        except:
          degs = [0,1,2,4]

        for deg in degs:
          selected_context_algs.append(CONTEXT_ALG_NAMES[i]+" "+str(deg))
          context_funcs.append(lambda seq, deg=deg, i=i: CONTEXT_ALGS[i](seq,deg))

    selected_algs = selected_algs + selected_context_algs
    funcs = funcs + context_funcs

    # IID, Loss-based, FMG 
    id_offset = self.seq_gen_id - len(IID_NAMES)
    if id_offset < 0:
      n=10000
      generator, title = lambda n: generate_iid_sequence(n, float(IID_NAMES[self.seq_gen_id])), f"iid {IID_NAMES[self.seq_gen_id]} sequences"
      x_label = "Length of sequence"
      n_list = range(10, n, n // 20)

    elif id_offset < len(LOSS_FUNCTIONS):
      n=1000
      loss_name = list(LOSS_FUNCTIONS)[id_offset]
      generator = lambda n: loss_based_sequence_generation(n, LOSS_FUNCTIONS[loss_name])
      title = f"Binary prediction on the {loss_name} sequence"
      x_label = "Length of sequence (n)"
      n_list = range(10, n, 20)

    else:
      id_offset = id_offset-len(LOSS_FUNCTIONS)
      n=10000
      fmg = [generate_FMG_sequence, generate_highlossFMG_ftl][id_offset]
      generator = lambda c: fmg(n, c)
      title = f"{'High-Loss' if id_offset else ''} FMG sequences  (n={n})"
      x_label = "Number of 0 Crossings"
      n_list = range(0, 100, 10)


    try:
      hindsight_deg = int(self.entry.get())
    except: # for any invalid input
      hindsight_deg = 0

    seq_list, hindsight_loss_list = build_sequence_list(n_list, sequence_generator=generator, baseline_expert_d=hindsight_deg)
    meta_plot(selected_algs, funcs, n_list, seq_list, x_label, title, 
              # ylabel=f"Regret Against Best Hindsight Degree {hindsight_deg}",
              ylabel=f"Regret",
              best_expert_loss_list=hindsight_loss_list)
    
if __name__ == "__main__":
  App().mainloop()
