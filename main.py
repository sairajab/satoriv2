# from run_experiments_relative import *

# if __name__ == "__main__":
#     run("combine")
    
    
import random

def dinucleotide_shuffle(sequence):
    # Convert the sequence to a list of dinucleotides
    dinucleotides = [sequence[i:i+2] for i in range(0, len(sequence), 2)]

    # Shuffle the dinucleotides randomly
    random.shuffle(dinucleotides)

    # Recombine the shuffled dinucleotides into a shuffled sequence
    shuffled_sequence = ''.join(dinucleotides)

    return shuffled_sequence

# Example usage:
original_sequence = "ATCGATCGATCG"
shuffled_sequence = dinucleotide_shuffle(original_sequence)
print("Original Sequence:", original_sequence)
print("Shuffled Sequence:", shuffled_sequence)