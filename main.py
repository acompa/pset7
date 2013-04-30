from utils import populate_set_with_data
from perceptron import perceptron, estimate_tags

def main():
	dataset = populate_set_with_data('train')
	weights = perceptron(dataset)

	# Use learned weights to estimate y's
	estimates = estimate_tags(dataset, weights)

if __name__ == "__main__":
	main()
