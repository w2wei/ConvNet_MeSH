input_list = ['all', 'this', 'happened', 'more', 'or', 'less']

def find_ngrams(input_list, n):
	# return zip([input_list[i:] for i in range(n)])
	return zip(*[input_list[i:] for i in range(n)])

print find_ngrams(input_list, 2)