"""
The CRF here takes the form

 Y --> a_1, ..., a_5 --> v_1,1, ..., v_2,1, ..., v_5,N
 |
 Y'--> ...
 |
...

where Y is a part of speech tag, a is a feature (one of 5) for each
word, and v is the value set (one for each of the 5 features) each feature can
draw from.

Note that weights exist for each feature, tag, and feature value, aka
w_{Y, a, v}. Weights also exist between tags for parts of speech.
"""
import numpy as np

from maxsum import belief_propagation

ITERATIONS = 50

def predict_pos(x, weights, crf):
	"""
	Predict the part-of-speech for a given feature vector and weights.
	Performed by variable elimination or max-sum BP on the CRF.

	"""
	# TODO: implement max-sum BP.
	y_new = belief_propagation(x, crf)
	assert x.shape[0] == y_new.shape[0]
	return y_new

def f(x, y):
	""" Feature vector (sufficient statistics) for the CRF. """
	# TODO: where do these come from?
	return

def perceptron(samples):
	"""
	Averaged structured perceptron algorithm for learning model weights.

	Learns weights from a set of samples, then returns those weights.
	"""
	# TODO: get samples in here
	assert type(samples) == list
	assert len(samples) > 0

	# TODO: initialize weights and weights_bar properly.
	weights = 0
	weights_bar = 0

	# TODO: create CRF
	crf = None

	for _ in range(ITERATIONS):
		for sample in samples:
			y, x = sample

			# x should have the same # of words (rows) as y and one column for
			# each feature
			assert x.shape[0] == y.shape[0]
			y_new = predict_pos(x, weights, crf)
			if y != y_new:
				weights = weights + f(x, y) - f(x, y_new)
				weights_bar = (weights_bar +
						(weights / (ITERATIONS * len(samples))))

	return weights_bar
