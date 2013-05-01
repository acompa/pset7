from utils import populate_set_with_data
from perceptron import perceptron, estimate_tags

def q2():
	dataset = populate_set_with_data('train', limit = 100)
	weights = perceptron(dataset)

	# Use learned weights to estimate y's
	estimates = estimate_tags(dataset, weights)

if __name__ == "__main__":
	q2()
