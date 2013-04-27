import numpy as np
import networkx as nx

from collections import defaultdict

def create_crf(length, parts, features):
	"""
	Populate a nx.Graph with a simple chain of parts of speech. Also attach a
	map from nodes to PartOfSpeechs (see below).
	"""
	g = nx.Graph()
	for node in range(0, length, 2):
		# Add both nodes and their POS maps.
		g.add_node(node, pos=part_of_speech_map(parts, features))
		g.add_node(node+1, post=part_of_speech_map(parts, features))
		
		# Add weighted edge between nodes.
		g.add_edge(node, node+1, weight=0.0)
	return g

def part_of_speech_map(parts, features):
	"""
	Method creating a part of speech. Each word has multiple parts of speech
	associated with it; this object thus maps from each word's potential part
	of speech to an object containing potentials for each of that word's
	different features.

	parts:
		list of potential part names
	features:
		map of feature names to cardinality of value set
	"""

	# TODO: shitty data structure. Do better!

	assert isinstance(parts, list)
	assert isinstance(features, dict)

	to_features = defaultdict(dict)
	for part in parts:
		for feature_name in features.keys():
			num_vals = features[feature_name]
			to_features[part][feature_name] = np.zeros(num_vals)

	return to_features

def belief_propagation(x, crf):
	"""
	Passes messages along the part-of-speech CRF produced by create_crf().

	Node potentials are created by restricting weights for each part of speech
	using the observed words x.	Edge potentials are...

	With node & edge potentials in hand, we pass messages until convergence.
	Then we find the argmax assignment to the speech parts for the sentence
	in the CRF.
	"""

	# TODO: edge potentials?!?
	assert isinstance(crf, nx.Graph)

	for node in crf:
		pos_map = crf[node]['pos']

		# Node potential is simply the vector in pos_map corresponding to the
		# observed words
		# TODO: organize weights s.t. I can grab a weight vector using x values
		observation = x[node, :]
		node_potential = None

	return
