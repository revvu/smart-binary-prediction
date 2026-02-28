import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from Scripts.graphing import *
from Scripts.sequence_generation import *

plt.rcParams.update({
    "figure.figsize": (8, 8),
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})
#plt.style.use('dark_background')
plt.style.use('default')

BIG_ETA = 1000000
EPSILON = 1e-10  # Small value for numerical stability

def calculate_context_losses(seq, max_d):
    """
    Calculate cumulative context prediction losses for a binary sequence.

    Args:
        seq (list of int): Binary sequence of 0s and 1s.
        max_d (int): Maximum context depth.

    Returns:
        np.ndarray: A 3D array of shape (T, 2^(max_d + 1), 2) representing the cumulative losses for each context.
    """
    T = len(seq)
    # Axis 0: t; Axis 1: L_X(h), T; Axis 2: y in {0,1}
    # Axis 1 starts with a dummy node to ease index calculation.     
    context_losses = np.ones((T+1, 2**(max_d + 1), 2))*EPSILON
    context_losses[:,0,:] = [None, None]  # Dummy initialization for easier indexing

    for t in range(max_d, T):
        context = seq[t-max_d:t]  # Extract context of length max_d
        context_indices = calculate_context_indices(context)
        
        # Update losses based on the observed bit
        context_losses[t+1][context_indices] = [seq[t], 1 - seq[t]]

    # Compute cumulative sum of losses across time
    # The context_losses[t] is the prior to observing turn t. 
    assert np.all((context_losses[:, 1:, :] >= 0) & (context_losses[:, 1:, :] <= 1)), "Losses are not between 0 and 1"
    return np.cumsum(context_losses, axis=0)

def calculate_context_indices(context):
    """
    Calculate the indices for the context in the context_losses array.

    Args:
        context (list of int): A binary context.

    Returns:
        list of int: Indices corresponding to the context in the losses array.
    """
    indices = [1]
    indices.extend(indices[-1] * 2 + bit for bit in reversed(context))
    return indices
 
def master_context_tree(seq, max_d, cum_context_losses=None, eta=None, g=None, adapt=False, SMART=(False, None)):
    """
    Implements the master Hedge algorithm using a context tree structure.

    Args:
        seq (list of int): Binary sequence of 0s and 1s.
        max_d (int): Maximum context depth.
        eta (float, optional): Learning rate parameter. 
            - If `adapt` is False, Hedge is played with the provided eta. 
            - If eta is not provided, eta defaults to sqrt(8*ln(K)/T), where K is the number of experts.
        g (function, optional): Function to assign prior probabilities over tree orders.
            - If g is None, a uniform prior is used.
        adapt (bool, optional): If True, AdaHedge is played (adapts eta based on the mixability gap).
        SMART (tuple, optional): Parameters for the SMART variant of Hedge.
            - SMART[0]: Boolean to enable SMART strategy.
            - SMART[1]: Threshold for switching to Hedge.

    Returns:
        np.ndarray: An array of actions taken by the algorithm.
    """
    assert all(bit in [0, 1] for bit in seq), "Sequence must be binary."
    T = len(seq)
    actions = np.full(T, 0.5)  # Initialize actions with 0.5


    if eta is None:
        eta = np.sqrt(8 * np.log(2) / T)
    if g is None:
        g = lambda h: 1

    mixability_gap = 0
    switched_to_Hedge = False

    # Compute cumulative context losses
    if cum_context_losses is None:
        cum_context_losses = calculate_context_losses(seq, max_d)
    assert len(cum_context_losses) >= 2**(max_d + 1), "Context losses must be calculated for the appropriate degree."

    for t in range(max_d, T):
        context = seq[t - max_d:t]
        context_indices = calculate_context_indices(context)

        # Compute the weights for playing 1 using cumulative context losses
        summed_losses = logsumexp(-eta * cum_context_losses[t], axis=1)
        
        def g_prime(h):
            irrelevant_indices = [i for i in range(2**h, 2**(h + 1)) if i not in context_indices]
            if not irrelevant_indices:
                return g(h)
            return g(h) * np.exp(np.sum(summed_losses[irrelevant_indices]))

        w1_numerator = logsumexp([
            np.log(g_prime(h)) - eta * cum_context_losses[t][context_indices[h]][1]
            for h in range(max_d + 1)
        ])
        w1_denominator = logsumexp([
            np.log(g_prime(h)) + summed_losses[context_indices[h]]
            for h in range(max_d + 1)
        ])
        w1 = np.exp(w1_numerator - w1_denominator)
        actions[t] = w1

        # Observe next bit and calculate the mixability gap
        l = np.asarray([seq[t], 1 - seq[t]])
        mixability_gap += np.dot([1-w1, w1], l) + (logsumexp(-eta * l, b=[1-w1, w1]) / eta)

        # Adapt eta if required
        if adapt:
            eta = np.log(2) / (mixability_gap + EPSILON) if mixability_gap > 0 else BIG_ETA

        # SMART strategy: switch to Hedge if the mixability gap is large enough
        if SMART[0] and not switched_to_Hedge:
            if mixability_gap < SMART[1]:
                eta = BIG_ETA
            else:
                switched_to_Hedge = True
                eta = np.sqrt(8 * np.log(2) / T)
    assert np.all((actions>=0) & (actions<=1))
    return actions

def find_best_hindsight_expert_loss(seq, target_d, context_losses=None):
    """
    Calculate the best hindsight expert loss for a binary sequence at a given context depth.

    Parameters:
    seq (list of int): Binary sequence.
    target_d (int): Target context depth.
    context_losses (list of list of float, optional): Precomputed context losses. 
                                                      If None, calculate them up to target_d.

    Returns:
    float: Sum of the minimum losses for each context at the target depth.

    Raises:
    AssertionError: If context_losses length is insufficient for target_d.
    """
    if context_losses is None:
        context_losses = calculate_context_losses(seq, target_d)[-1]

    assert len(context_losses) >= 2**(target_d + 1), "Calculate context losses to the appropriate depth."

    return sum(min(context_losses[context_index]) for context_index in range(2**target_d, 2**(target_d + 1)))


def FTL_context_tree(seq, max_d, cum_context_loss=None):
    """
    Plays the Follow The Leader (FTL) strategy using the context tree structure.
    
    Args:
        seq (list of int): Binary sequence of 0s and 1s.
        max_d (int): Maximum context depth.
        cum_context_loss (np.ndarray, optional): Cumulative context losses.

    Returns:
        np.ndarray: An array of actions taken by the FTL algorithm.
    """
    return master_context_tree(seq=seq, max_d=max_d, eta=BIG_ETA, cum_context_losses=cum_context_loss)

def Hedge_context_tree(seq, max_d, cum_context_loss=None):
    """
    Plays the Hedge strategy using the context tree structure.
    
    Args:
        seq (list of int): Binary sequence of 0s and 1s.
        max_d (int): Maximum context depth.
        cum_context_loss (np.ndarray, optional): Cumulative context losses.

    Returns:
        np.ndarray: An array of actions taken by the Hedge algorithm.
    """
    return master_context_tree(seq=seq, max_d=max_d, cum_context_losses=cum_context_loss)

def AdaHedge_context_tree(seq, max_d, cum_context_loss=None):
    """
    Plays the AdaHedge strategy using the context tree structure.
    
    Args:
        seq (list of int): Binary sequence of 0s and 1s.
        max_d (int): Maximum context depth.
        cum_context_loss (np.ndarray, optional): Cumulative context losses.

    Returns:
        np.ndarray: An array of actions taken by the AdaHedge algorithm.
    """
    return master_context_tree(seq=seq, max_d=max_d, eta=BIG_ETA, adapt=True, cum_context_losses=cum_context_loss)

def Bartlett_context_tree(seq, max_d, cum_context_loss=None):
    """
    Plays the Bartlett strategy using the context tree structure with a specific prior.
    
    Args:
        seq (list of int): Binary sequence of 0s and 1s.
        max_d (int): Maximum context depth.
        cum_context_loss (np.ndarray, optional): Cumulative context losses.

    Returns:
        np.ndarray: An array of actions taken by the Bartlett algorithm.
    """
    return master_context_tree(seq=seq, max_d=max_d, g=lambda h: pow(2, -pow(2, h + 1)), adapt=True, cum_context_losses=cum_context_loss)

def SMART_context_tree(seq, max_d, cum_context_loss=None):
    """
    Plays the SMART strategy using the context tree structure.
    
    Args:
        seq (list of int): Binary sequence of 0s and 1s.
        max_d (int): Maximum context depth.
        cum_context_loss (np.ndarray, optional): Cumulative context losses.

    Returns:
        np.ndarray: An array of actions taken by the SMART algorithm.
    """
    return master_context_tree(seq=seq, max_d=max_d, eta=BIG_ETA, SMART=(True, np.sqrt(len(seq) * np.log(2) / 2)), cum_context_losses=cum_context_loss)



