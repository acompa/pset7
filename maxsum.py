import numpy as np
import operator as op

from utils import get_feature_vec_slice, FEATURE_COUNT
from functools import reduce

def belief_propagation(xs, weights, crf):
	"""
	Passes messages along the part-of-speech linear-chain CRF.

	Node potentials := sum of weights for observed features in x
		* should have shape == # of tags

	Edge potentials := weight for (y, y+1)
		* should have shape == # of tags x # of tags
		* matrix is reduced as we traverse the CRF

	With node & edge potentials in hand, we pass messages up to the root.
	Then, after decoding the root's MAP assignment, we can simply decode all
	other node's assignments.

	xs:
		ndarray; shape == # of words x # of features
	weights:
		feature/weight vector produced by perceptron._create_feature_vec()
	crf:
		list representing linear-chain CRF

	returns:
		modified version of crf with tags decoded for each word
	"""
	num_tags = weights[0].shape[0]

	# Defense
	assert isinstance(weights, list)
	assert len(weights) == FEATURE_COUNT + 1
	assert isinstance(crf, list)
	assert len(crf) == xs.shape[0]
	assert isinstance(xs, np.ndarray)
	assert xs.shape[1] == FEATURE_COUNT

	# First node has no incoming message--use an array of zeros.
	message_in = np.zeros((num_tags, 1))
	messages_in = [[] for _ in crf]
	node_potentials = []
	edge_potentials = weights[-1]	# edge potentials are just transition w's!
	for idx, node in enumerate(crf):
		tag_features = xs[node, :]
		# Node potential is simply the sum of all tag vectors for each fixed
		# feature value in tag_features. Save the node potential.
		node_potential = reduce(
				op.add,
				[get_feature_vec_slice(weights, fidx, fvalue)
					for fidx, fvalue in enumerate(tag_features)])
		node_potential.resize((num_tags, 1))
		node_potentials.append(node_potential)

		# Edge potentials are simply the transition weights

		# Message outgoing from node in the chain CRF == sum of potentials and
		# all incoming messages.
		messages_in[idx].append(message_in)
		message_out = message_in + node_potential + edge_potentials
		message_in = message_out

	# After propagating to root, trace back and decode each node's assignment.
	# TODO: might not be necessary. Keep it anyway, for now.
	max_marginals = []
	messages_in[-1].append(np.zeros((num_tags, 1)))
	for idx, node in enumerate(crf[::-1]):
		message_out = (node_potentials[idx] + edge_potentials +
				reduce(op.add, messages_in[idx]))
		messages_in[idx-1].append(message_out)

	max_marginals = [
			node_potentials[node_idx] + reduce(op.add, messages_in[node_idx])
			for node_idx in range(len(crf))]
	assignments = [np.argmax(max_marg) for max_marg in max_marginals]

	return assignments 

