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

	With node & edge potentials in hand, we pass messages until convergence.
	Then we find the argmax assignment to the speech parts for the sentence
	in the CRF.

	xs:
		ndarray; shape == # of words x # of features
	weights:
		feature/weight vector produced by perceptron._create_feature_vec()
	crf:
		list representing linear-chain CRF

	returns:
		modified version of crf with tags decoded for each word
	"""
	# Defense
	assert isinstance(weights, list)
	assert len(weights) == FEATURE_COUNT + 1
	assert isinstance(crf, list)
	assert len(crf) == xs.shape[0]
	assert isinstance(xs, np.ndarray)
	assert xs.shape[1] == FEATURE_COUNT

	for node in crf:
		tag_features = xs[node, :]
		# Node potential is simply the sum of all tag vectors for each fixed
		# feature value in tag_features.
		node_potential = reduce(
				op.add,
				[get_feature_vec_slice(weights, fidx, fvalue)
					for fidx, fvalue in enumerate(tag_features)])

		# Edge potentials are simply the transition weights
		edge_potentials = weights[-1]

		# TODO: Create outgoing message for the node.
		message_out = None

	# TODO: after propagating to root, get root's MAP assignment and update crf
	# TODO: with root's MAP assignment, propgate back down and update crf

	return crf

