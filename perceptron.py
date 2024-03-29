"""
The CRF here takes the form

 Y --> a_1, ..., a_5 --> v_1,1, ..., v_2,1, ..., v_5,N
 |
 Y'--> ...
 |
...

where Y is a part of speech tag, a is a feature (one of 5) for each
word, and v is the value set (one for each of the 5 features) each feature can
draw from. a's are observed, so we can treat them as single-node potentials
for each unobserved word tag.

Note that weights exist for each feature, tag, and feature value, aka
w_{Y, a, v}. Weights also exist between tags for parts of speech.
"""
import numpy as np
import maxsum
import sys

from utils import FEATURE_TUPLE, NGRAM_TUPLE, NGRAM_LENGTH
from string import join
from ipdb import set_trace

ITERATIONS = 50

def _feature_vec(xs, y):
	"""
	Feature vector (sufficient statistics) for structured perceptron. Returns a
	vector taking values = {0, 1}, where

	f == 1 if a word with features x takes the tag y AND the previous word
	          takes tag y'
	  == 0 o.w.

	xs:
		feature mat for a sample
	y:
		tag array for a sample
	"""
	f = _create_feature_vec()

	# Iterate over rows in x, values of y, and update f.
	count = y.shape[0]
	for idx in range(count):
		word = xs[idx, :]
		tag = y[idx]

		# Defense!
		assert len(word) + 1 == len(f)

		# Iterate over feature values in word, increment the vector
		for fidx, fvalue in enumerate(word):
			f[fidx][tag, fvalue] += 1

		# Update ngram matrix at the end of fvec. Must update edge potential
		# for previous AND next tag.
		if idx != 0:
			prev_tag = y[idx-1]
			f[-1][prev_tag, tag] += 1
		if idx != count - 1:
			next_tag = y[idx+1]
			f[-1][tag, next_tag] += 1

	return f

def perceptron(samples):
	"""
	Averaged structured perceptron algorithm for learning model weights.

	References a feature tuple containing name-cardinality pairs to build the
	weight vector, then updates that vector using max-sum BP.

	samples:
		list of tuples of the form (word_mat, tag_array)
			word_mat.shape == # of words x # of features
			tag_array.shape == # of words
	"""
	assert type(samples) == list
	assert len(samples) > 0

	# Use feature_tuple to build a multi-dim weight vector. Add ngram weights
	# onto the flattened weight array
	weights = _create_feature_vec()
	weights_bar = _create_feature_vec()

	sys.stdout.write("[")
	for iter in range(ITERATIONS):
		if (iter + 1) % (ITERATIONS / 10.0) == 0:
			sys.stdout.write("=")
			if iter + 1 == ITERATIONS:
			#	sys.stdout.write("]\nEstimates on final pass:\n")
				sys.stdout.write("]\n")
		for sample in samples:
			xs, y = sample

			# x should have the same # of words (rows) as y and one column for
			# each feature
			assert xs.shape[0] == y.shape[0]
			assert xs.shape[1] == len(FEATURE_TUPLE)

			# CRF is linear, represented as a list of ints. These ints will be
			# reassigned once we decode part-of-speech tags for each word.
			crf = [0 for _ in range(y.shape[0])]

			# Obtain an estimate of y using max-sum BP.
			y_est = maxsum.belief_propagation(xs, weights, crf)
			#if iter == ITERATIONS-1:
			#	print "Tag estimate: %s" % join([str(i) for i in y_est], ' ')
			#	print "Actual tags:  %s" % join([str(int(i)) for i in y], ' ')

			# Update weights if estimate doesn't match actual. Get the feature
			# vector for both, then update the weights.
			if (y != y_est).any():
				f_actual = _feature_vec(xs, y)
				f_estimate = _feature_vec(xs, y_est)
				update = [act - est for est, act in zip(f_estimate, f_actual)]
				weights = [w + u for w, u in zip(weights, update)]

			# Normalize weights.
			norm_weights = [w / (1.0 * ITERATIONS * len(samples))
					for w in weights]
			weights_bar = [wb + nw
					for wb, nw in zip(weights_bar, norm_weights)]

	return weights_bar

def estimate_tags(samples, weights):
	"""
	Method for estimating tags for a set of samples.
	"""
	estimates = []
	for sample in samples:
		xs, y = sample

		# CRF is linear, represented as a list of ints. These ints will be
		# reassigned once we decode part-of-speech tags for each word.
		crf = [0 for _ in range(y.shape[0])]

		# x should have the same # of words (rows) as y and one column for
		# each feature
		assert xs.shape[0] == y.shape[0]
		assert xs.shape[1] == len(FEATURE_TUPLE)
		y_new = maxsum.belief_propagation(xs, weights, crf)
		estimates.append(y_new)

	return estimates

def error_rate(dataset, estimates):
	"""
	Check the error rate on estimates produced by estimate_tags().
	"""
	incorrect = 0.0
	count = 0.0

	for idx in range(len(estimates)):
		estimate = estimates[idx]
		_, actual = dataset[idx]
		for e, a in zip(estimate, actual):
			count += 1.0
			if e != a:
				incorrect += 1.0
	
	return incorrect / count

def _create_feature_vec():
	"""
	Create a list of ndarrays, where

		* the list has len == F + 1, where F := # of features
		* the list's last element contains the ndarray of tag transition
		  weights

	returns:
		list of ndarrays; see above
	"""
	num_tags = NGRAM_TUPLE[0]
	fvec = []
	for _, size in FEATURE_TUPLE:
		fvec.append(np.zeros((num_tags, size)))

	# Append tag ngram weights to end
	fvec.append(np.zeros((num_tags, num_tags)))
	return fvec

def _update_feature_vec(fvec, word, tag_ngram):
	"""
	Increments the feature vector's values depending on word's and tag_ngram's
	features.

	fvec:
		list of ndarrays produced by _create_feature_vec
	word:
		observed features for a word; one row from a sample's xs
	tag_ngram:
		ngram for a word and its successor; a slice of a sample's y
	"""

