import numpy as np
import os

from collections import defaultdict
from random import shuffle

FEATURE_COUNT = 5

def read_file(fname, dataset):
	"""
	Add a sample with the file's contents to dataset. A sample takes the form
	(y, xs)	where x is a word_num x feature_num matrix.
	"""
	assert isinstance(dataset, set)

	with open(fname) as f:
		contents = f.read().strip().split('\n')
		count = len(contents)
		y = np.zeros(count)
		xs = np.zeros((count, FEATURE_COUNT))

		# Populate y and each row of xs.
		for idx in range(count):
			line = contents[idx].split(',')

			# Get features and tag, update sample
			xs[idx, :] = [int(n) for n in line[2:]]
			y[idx] = int(line[1])
			token = line[0]

		dataset.add((y, xs))

def populate_set_with_data(settype, limit=None):
	"""
	Populate a set with all training data or all test data. Set type must be
	'train' or 'test'.

	settype:	'train' or 'test'
	limit:		int; number of samples to add
	"""
	assert settype == 'train' or settype == 'test'
	assert type(limit) is int

	dataset = set([])
	filenames = [fname for fname in os.listdir('Data/') if settype in fname]
	shuffle(filenames)

	# Read up to $limit samples.
	if not limit or limit < len(filenames):
		limit = len(filenames)
	for _ in xrange(limit):
		fname = filenames.pop()
		read_file(fname, dataset)

	return dataset


