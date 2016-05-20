#CSCI 49371 - Project II
#Coded by Kaitlin Smith & Kevin Lam

# -*- coding: UTF-8 -*-
import sys
from pyspark import SparkContext
from math import log10, pow, sqrt
import time
import itertools

#filters words that indicate document
def doc_filter(str):
	if (str.startswith('doc')) and str[3:].isdigit():
		return str
		
#filters out any words that don't fit required criteria, gene_abc_gene and disease_xyz_disease
def tf_word_filter(list):
	filtered = {}
	for word in list:
		if ((word.startswith('gene_') and word.endswith('_gene')) \
			or (word.startswith('disease_') and word.endswith('_disease'))):
			filtered[word] = filtered.get(word, 0) + 1
	return filtered
	
#special implementation to fit df algorithm
def df_word_filter(list):
	filtered = []
	for word in list:
		if ((word.startswith('gene_') and word.endswith('_gene')) \
			or (word.startswith('disease_') and word.endswith('_disease'))):
			filtered.append(word)
	return set(filtered)

#merging together tf and idf matrices
def tfidf_join(tf,idf):

	tfidf = []
	for tf_term in tf:
		tfidf.append((tf_term, tf[tf_term]*idf[tf_term]))
	return dict(tfidf)

#computing term term relevancy based on similarity formula between two different terms
def term_term_relevance(termA, termB):

	multiplication_list = []
	power_list_A = []
	power_list_B = []
	for elementA in termA:
		power_list_A.append(pow(elementA, 2))
	
	#need to multiply the sum of products of every term in a and b in order to compute numerator
	multiplication_list = [a*b for a,b in zip(termA, termB)]
	
	for elementB in termB:
		power_list_B.append(pow(elementB, 2))

	numerator = sum(multiplication_list)
	#need to take the sum of powers of every element in a and b, then take the square root of the sums and multiply them in order to compute denominator
	bottomA = sum(power_list_A)
	bottomB = sum(power_list_B) 
	denominator = sqrt(bottomA) * sqrt(bottomB)

	#divide by 0 error catch
	if denominator != 0:
		return (numerator / denominator)
	else:
		return 0

#sorts values of a list descendingly based on their 3rd term, in this case, the relevancy score
def sort_descending(input):
	sorted_values = sorted(input, key=lambda tup: -tup[2])
	return sorted_values

#calculating term term relevancy of a given term with every other term in the corpus
def term_term_query(tf_idf, query_term):
	f = open('query_result.txt','w')

	#query_list keeps track of the corresponding tfidf vector to the given query term
	query_list = []
	for element in tf_idf:
		query_list.append(element[1].get(query_term, 0))

	#dictionary_keys retrieves the name of every term in the corpus, not including the given query term
	dictionary_keys = []
	for element in tf_idf:
		for key in element[1]:
			if key != query_term:
				dictionary_keys.append(key)

	#takes distinct values
	dictionary_keys = list(set(dictionary_keys))
	key_list = []
	corpus_list = []

	#retrieving the tfidf vector for every term in the corpus minus the given querying term
	for key in dictionary_keys:
		score_list = []
		key_list.append(key)
		for element in tf_idf:
			score_list.append(element[1].get(key,0))
		corpus_list.append(score_list)

	#relevance list keeps track of the query term, compared term in corpus, and their term term relevancy
	relevance_list = []
	for index, element in enumerate(corpus_list):
			relevance_list.append((query_term, key_list[index], term_term_relevance(list(element),query_list)))

	#eliminating 0 values from the relevancy list
	nonzero_list = 	sc.parallelize(relevance_list) \
					.filter(lambda a: a[2] > 0) \
					.collect()

	#sorting term term relevancy list in descending order for maximum readability
	descending_list = sort_descending(nonzero_list)
	f.write(str(descending_list))
	print("File written successfully.")
	
#filling tfidf matrix with 0s where there are no values
def tfidf_print_zero_filled(tfidf, idf):
	print_spooler = []
	header = []
	for key in idf:
		header.append(key)
	print_spooler.append(header)
	for row in tfidf:
		print_row = []
		for value in idf:
			print_row.append(row[1].get(value, 0))
		print_spooler.append(print_row)
	
	return print_spooler

#outputting tfidf matrix
def tfidf_print(tfidf, idf):
	print_spooler = []
	for row in tfidf:
		if row[1]:
			print_spooler.append((row[0], row[1]))
	return print_spooler
	
#retrieving the tfidf vector for a given term
def fetch_term_vector(tfidf, term):
	term_vector = []
	for row in tfidf:
		term_vector.append(row[1].get(term, 0))
	return term_vector
	
#computing similarity between two different terms
def semantic_similarity(vectorA, vectorB):
	numerator = sum(a*b for a,b in zip(vectorA,vectorB))
	denominator = 	sqrt(sum(i ** 2 for i in vectorA)) + \
					sqrt(sum(i ** 2 for i in vectorB))
	if(denominator != 0):
		return numerator / denominator
		
if __name__ == "__main__":
	#initializing file and spark
	filename = "project2_data.txt"
	sc = SparkContext(appName="TF-IDF")
	
	#crunching tf rdd
	tf = 	sc.textFile(filename).map(lambda line: line.split(" ")) \
			.map(lambda x: (x[0], tf_word_filter(x[1:])))
				
	#keeping track of corpus size to be used in idf calculation later
	corpus_size = tf.count()
	
	#crunching df rdd
	df =	sc.textFile(filename).map(lambda line: line.split(" ")) \
			.flatMap(lambda x: list(df_word_filter(x[1:]))) \
			.map(lambda x: (x, 1)) \
			.reduceByKey(lambda x, y: x + y)
		
	#crunching idf rdd	
	idf =	dict(df.map(lambda x: (x[0], log10(corpus_size/x[1]))).collect())
		
	#merging tf and idf rdd to make... tfidf rdd!
	tfidf = tf.map(lambda x: (x[0], tfidf_join(x[1], idf)))
	
	#feel free to comment back in the below code to output the tfidf matrix to a text file:
	# filepath = 'TFIDF_out.txt'
	# tfidf_output = tfidf_print(tfidf.collect(), idf)
	# with open(filepath, 'w') as file:
	# 	for item in tfidf_output:
	# 		file.write("{}\n".format(item))
		
	# save = 	tfidf.coalesce(1).saveAsTextFile('test_out_'+str(time.time()))

	#checking if user wants to calculate term term relevancy for a given term
	if ((len(sys.argv) >=1) & (str(sys.argv[1]) != "term")):
		term_term_query(tfidf.collect(), sys.argv[1])
	
	

	#checking if user wants to calculate term term relevancy for the entire corpus
	#this is very slow on the data text file, need more computing power! maybe can be further optimized?
	if len(sys.argv) >=1:
		for argument in sys.argv:
			if str(argument) == "term":
				print("Calculating term term relevancy for entire corpus. This may take a while. . .")
				term_term = sc.parallelize(idf)
				lookup = tfidf.collect()
				term_vectors = term_term.map(lambda x: (x, fetch_term_vector(lookup, x)))
				combinations =  []
				for pair in itertools.combinations(term_vectors.collect(), 2):
					combinations.append(pair)
					term_pairs = 	sc.parallelize(combinations) \
									.map(lambda x: ((x[0][0],x[1][0]), semantic_similarity(x[0][1], x[1][1]))) \
									.filter(lambda x: x[1] != 0) \
									.sortBy(lambda x: -x[1])
	
					term_pair_save = term_pairs.saveAsTextFile('term_pairs_'+str(time.time()))