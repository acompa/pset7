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
		feature mat in a sample
	y:
		tag array in a sample
	"""
	f = _create_feature_vec()

	# Iterate over rows in x, values of y, and update f.
	count = y.shape[0]
	for idx in range(count):
		word = xs[idx, :]
		tag_ngram = y[idx:idx+NGRAM_LENGTH]
		f = _update_feature_vec(f, word, tag_ngram)
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

	for _ in range(ITERATIONS):
		for sample in samples:
			xs, y = sample

			# x should have the same # of words (rows) as y and one column for
			# each feature
			assert xs.shape[0] == y.shape[0]
			assert xs.shape[1] == len(FEATURE_TUPLE)

			# CRF is linear, represented as a list of ints. These ints will be
			# reassigned once we decode part-of-speech tags for each word.
			crf = [0 for _ in range(y.shape[0])]

			y_est = maxsum.belief_propagation(xs, weights, crf)
			print "Tag estimate: %s" % join([str(i) for i in y_est], ' ')
			#set_trace()
			if (y != y_est).any():
				# TODO: update weights properly--they're no longer ndarrays!!
				f_estimate = _feature_vec(xs, y_est)
				f_actual = _feature_vec(xs, y)
				update = [est - act for est, act in zip(f_estimate, f_actual)]
				weights = [w + u for w, u in zip(weights, update)]
				print "\tIncorrect tag estimate! Weights updated to:"
				print weights
				norm_weights = [w / (1.0 * ITERATIONS * len(samples))
						for w in weights]
				weights_bar = [wb + nw
						for wb, nw in zip(weights_bar, norm_weights)]

	return weights_bar

def estimate_tags(samples, weights):
	"""
	Method for estimating tags for a set of samples.
	"""
	estimates = set([])
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
		estimates.add(y_new)

	return estimates

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
	assert isinstance(fvec, list)
	assert len(fvec) == len(word) + 1
	assert len(tag_ngram) >= 1
	tag = tag_ngram[0]

	# Iterate over feature values in word, increment fvec
	for idx, fvalue in enumerate(word):
		fvec[idx][tag, fvalue] += 1

	# Update ngram matrix at the end of fvec
	fvec[-1][tuple(tag_ngram)] += 1
	return fvec

