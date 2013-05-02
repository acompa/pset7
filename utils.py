import numpy as np
import os
import re

from random import shuffle

TAG_COUNT = 10
FEATURE_COUNT = 5

def read_file(fname, dataset):
	"""
	Add a sample with the file's contents to dataset. A sample takes the form
	(xs, y)	where x is a word_num x feature_num matrix.
	"""
	assert isinstance(dataset, list)

	with open(fname) as f:
		contents = f.read().strip().split('\n')
		count = len(contents)
		y = np.zeros(count)
		xs = np.zeros((count, FEATURE_COUNT))

		# Populate y and each row of xs.
		for idx in range(count):
			line = contents[idx].split(',')
			assert len(line) == 7		# as specified in hw
			assert len(line[2:]) == 5	# sanity check

			# Get features and tag, update sample. Decrease bias, prefix,
			# tags, and suffix features by 1 (so they can be represented as
			# idxs)
			xs[idx, :] = [int(n) for n in line[2:]]
			xs[idx, 0] -= 1
			xs[idx, 3] -= 1
			xs[idx, 4] -= 1
			y[idx] = int(line[1]) - 1
			token = line[0]

		dataset.append((xs, y))

def populate_set_with_data(settype, limit=None):
	"""
	Populate a set with all training data or all test data. Set type must be
	'train' or 'test'.

	settype:	'train' or 'test'
	limit:		int; number of samples to add
	"""
	assert settype == 'train' or settype == 'test'
	assert type(limit) is int or limit == None

	dataset = []

	# Get a sorted list of train or test files.
	filenames = [fname for fname in os.listdir('Data/') if settype in fname]
	filenames = sorted(filenames,
			key=lambda x: int(re.findall('[0-9]{1,4}', x)[0]), reverse=True) 

	# Read up to $limit samples.
	if not limit or limit > len(filenames):
		limit = len(filenames)
	for _ in xrange(limit):
		fname = "Data/" + filenames.pop()
		read_file(fname, dataset)

	return dataset

def _build_feature_tuple():
	"""
	Generate the tuple of features used to build weight and feature vectors
	for structured perceptron.
	"""
	# Features, as defined in hw
	bias = 1
	initial_cap = 2
	all_caps = 2
	prefix_id = 201
	suffix_id = 201

	# Return the feature tuple.
	return (('bias', bias),
			('initial_cap', initial_cap),
			('all_caps', all_caps),
			('prefix_id', prefix_id),
			('suffix_id', suffix_id))

def _build_ngram_tuple():
	"""
	Generate a tuple used for building the pairwise weights & features.
	"""
	# Number of tags and ngram size
	ngram_size = 2

	# Build tuple
	dims = [TAG_COUNT for _ in range(ngram_size)]
	return tuple(dims)

# Ugly ugly singletons, but whatever.
FEATURE_TUPLE = _build_feature_tuple()
NGRAM_TUPLE = _build_ngram_tuple()
NGRAM_LENGTH = len(NGRAM_TUPLE)

def get_feature_vec_slice(fvec, fidx, fvalue):
	"""
	For a fixed value of a given vector, return a slice of that feature's
	ndarray.

	fvec:
		list of ndarrays; produced by perceptron.create_feature_vec()
	fidx:
		index of feature we're fixing
	fvalue:
		fix feature to this value

	returns:
		a vector of weights for all tag values, holding a feature value
		constant
	"""
	slice = fvec[fidx][:, fvalue]
	assert slice.shape[0] == TAG_COUNT
	return slice
