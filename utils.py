import numpy as np
import os

from collections import defaultdict
from random import shuffle

def read_file(fname, datamap):
	"""
	Add data for each token in the file to a data map.

	Map is expected to map from tokens to a list of samples:

	datamap['foo'] = [(y_foo1, x_foo1), (y_foo2, x_foo2), ...]
		See p.2 of HW for more information.
	"""
	with open(fname) as f:
		for line in f:
			line = line.split(',')

			# Get features and tag, then add them to token.
			xs = [int(n) for n in line[2:]]
			y = int(line[1])
			token = line[0]
			datamap[token].append(tuple(y, xs))

def populate_map_with_data(maptype, limit=None):
	"""
	Populate a map with all training data or all test data. Map type must be
	'train' or 'test'.

	maptype:	'train' or 'test'
	limit:		int; number of samples to add
	"""
	assert maptype == 'train' or maptype == 'test'
	assert type(limit) is int

	datamap = defaultdict(list)
	filenames = [fname for fname in os.listdir('Data/') if maptype in fname]
	shuffle(filenames)

	# Read up to $limit samples.
	if not limit or limit < len(filenames):
		limit = len(filenames)
	for _ in xrange(limit):
		fname = filenames.pop()
		read_file(fname, datamap)

	return datamap


