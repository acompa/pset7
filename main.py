import matplotlib.pyplot as plt

from utils import populate_set_with_data
from perceptron import perceptron, estimate_tags, error_rate

def q2():
	dataset = populate_set_with_data('train', limit = 100)
	weights = perceptron(dataset)

	# Use learned weights to estimate y's
	estimates = estimate_tags(dataset, weights)
	print "Training error rate: %0.4f" % error_rate(dataset, estimates)

	# Now check error rate against test set.
	test_set = populate_set_with_data('test', limit = 1000)
	test_estimates = estimate_tags(test_set, weights)
	print "Test error rate: %0.4f" % error_rate(test_set, test_estimates)

def q3():
	errors = []
	for tlimit in range(100, 1001, 100):
		dataset = populate_set_with_data('train', limit = tlimit)
		weights = perceptron(dataset)

		# Use learned weights to estimate y's
		estimates = estimate_tags(dataset, weights)
		print ("Training error rate, %i samples: %0.4f" %
				(tlimit, error_rate(dataset, estimates)))

		# Now check error rate against test set.
		test_set = populate_set_with_data('test', limit = 1000)
		test_estimates = estimate_tags(test_set, weights)
		err = error_rate(test_set, test_estimates)
		errors.append(err)
		print ("Test error rate (trained on %i samples): %0.4f" %
				(tlimit, err))

	# Plot
	plt.plot(range(100, 1001, 100), errors)
	g = plt.gcf()
	g.set_size_inches((12, 16))
	plt.savefig("errors.png")

if __name__ == "__main__":
	print "** QUESTION #2"
	q2()
	print "** QUESTION #3"
	q3()
