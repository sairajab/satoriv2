import numpy
from readpwms import get_motif_score
nucleotide_alphabet = "ACGT" # The alphabet of the nucleotide sites
state_alphabet = "SHL" # The alphabet of the states (start, CG High, CG Low)


def GCRichSeq(length, tfs=None):
	# The emissions probability matrix. The rows correspond to the three states
	# (start, CG-rich, CG-poor). The columns and the probability of emitting
	# [A, C, G, T] conditional on the state.
	emission_probs = numpy.array([
		[0.00, 0.00, 0.00, 0.00],
		[0.13, 0.37, 0.37, 0.13],
		[0.37, 0.13, 0.13, 0.37]
	])
	emission_probs_human_promoters = [0.225, 0.278, 0.271, 0.226]
	# The transition probabilities matrix. The rows correspond to the state at
	# site i - 1, and the columns correspond to the probability of each state at
	# site i
	transition_probs = numpy.array([
		[0.00, 0.50, 0.50],
		[0.00, 0.63, 0.37],
		[0.00, 0.37, 0.63]
	])

	current_state = 0  # start at the start (state)
	simulated_sites = ""  # start with an empty sequence
	simulated_states = ""  # to record the sequence of states

	# the length of promoter sequence to simulate
	#length = 100
 
	for i in range(length):
		# choose a new state at site i, weighted by the transitional probability matrix
		current_state = numpy.random.choice(3, p=transition_probs[current_state])
		# choose a new nucleotide at site i, weighted by the emission probability matrix
		nucleotide = numpy.random.choice(4, p=emission_probs_human_promoters)

		# append the letter codes of the chosen state and nucleotide
		simulated_states += state_alphabet[current_state]
		simulated_sites += nucleotide_alphabet[nucleotide]
    
	return simulated_sites





