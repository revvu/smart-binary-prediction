import numpy as np
import random
from Scripts.classified_Hedge import find_best_hindsight_expert_loss
from tqdm import tqdm

def generate_FMG_sequence(n, c):
    """
    Generate an FMG sequence of length n with specified crossings.

    Parameters:
        n (int): Length of the sequence.
        c (int): Number of crossings (pairs of alternating bits).

    Returns:
        list[int]: List representing the generated sequence.

    Raises:
        ValueError: If 2c is greater than n.
    """
    # Check if 2c > n
    if 2 * c > n: raise ValueError("2c must be less than or equal to n")
    sequence = [xs for x in random.choices([[0,1], [1,0]], k=c) for xs in x]
    sequence.extend([1]*(n-2*c))
    return np.asarray(sequence)

def generate_iid_sequence(n, p):
    """
    Generate an iid binary sequence of length n with specified probability.

    Parameters:
        n (int): Length of the sequence.
        p (float): Probability of each bit being 1.

    Returns:
        np.ndarray[int]: Generated sequence.

    Raises:
        ValueError: If p not in [0,1].
    """
    # Check if p \notin [0,1]
    if p < 0 or p > 1:
        raise ValueError("p must be in [0,1]")
    # Initialize an empty binary sequence
    return np.random.binomial(1,p,n)

def generate_highlossFMG_ftl(n, c , p=0.3):
    """
    Generate a binary sequence of length n with specified crossings.

    Parameters:
        n (int): Length of the binary sequence.
        c (int): Number of crossings (pairs of alternating bits).

    Returns:
        np.ndarray[int]: Generated sequence.

    Raises:
        ValueError: If 2c is not less than n.
    """

    # Check if 2c < n
    if 2 * c >= n:
        raise ValueError("2c must be less than n")

    # Initialize an empty binary sequence
    binary_sequence = []

    # Generate the first 2c bits with (0,1) or (1,0) pairs with 50% probability
    for _ in range(c):
        # Randomly choose either (0,1) or (1,0) pair
        pair = random.choice([(0, 1), (1, 0)])
        # Append the pair to the binary sequence
        binary_sequence.extend(pair)

    # Append the remaining bits with 1,1 followed by random Ber(p)
    binary_sequence.extend([1] * 2)
    binary_sequence.extend(np.random.binomial(1,p,[n-2*c-2]))
    return np.asarray(binary_sequence)

def loss_based_sequence_generation(n, target_loss_function):
    L0 = L1 = 0
    seq = []
    for t in range(n):
        diff = target_loss_function(t,n)
        seq.append(0 if abs(diff - (L0-L1-1)) <= abs(diff - (L0-L1+1)) else 1)
        L1 += (seq[-1] == 0)
        L0 += (seq[-1] == 1)

    return np.asarray(seq)

#print(loss_based_sequence_generation(100, lambda x: pow(x, 0.4)))

def Bartlett_sequence(n):
    # XOR with the last three bits
    assert n>3, "Bartlett sequences must be longer than 3"
    seq = np.random.choice(2, size=3)
    for t in range(3, n):
        p = 0.6*((seq[t-1]+seq[t-2]+seq[t-3])%2)+0.2
        seq = np.append(seq, np.random.choice([0,1], p=[1-p, p]))

    return np.asarray(seq)


def build_sequence_list(c_list, sequence_generator, baseline_expert_d=0):
    """
    Generate sequences from c_list using a sequence generator and calculate the best hindsight expert loss for each sequence.

    Parameters:
    c_list (list): List of elements to generate sequences from.
    sequence_generator (function): Function to generate a sequence given an element from c_list.
    baseline_expert_d (int, optional): Target context depth for calculating the best hindsight expert loss. Default is 0.

    Returns:
    tuple: A list of generated sequences and an array of best hindsight expert losses.
    """
    seq_list = [sequence_generator(c) for c in c_list]
    best_expert_loss = np.array([find_best_hindsight_expert_loss(seq, target_d=baseline_expert_d) for seq in tqdm(seq_list)])

    return seq_list, best_expert_loss