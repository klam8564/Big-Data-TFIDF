# -*- coding: UTF-8 -*-
from pyspark import SparkContext
from math import log10, pow, sqrt
import time
import itertools

#accepts only document strings from file
def doc_filter(str):
	if (str.startswith('doc')) and str[3:].isdigit():
		return str
'''
filtering words that meet the given conditions
they must either start with gene_ and end with _gene
or start with disease_ and end with _disease
'''
# filters documents for proper layout and readys them for mapreduce
def tf_word_filter(list):
	filtered = {}
	for word in list:
		if ((word.startswith('gene_') and word.endswith('_gene')) \
			or (word.startswith('disease_') and word.endswith('_disease'))):
			filtered[word] = filtered.get(word, 0) + 1
	return filtered
# filters document for proper layout and readys them for mapreduce
def df_word_filter(list):
	filtered = []
	for word in list:
		if ((word.startswith('gene_') and word.endswith('_gene')) \
			or (word.startswith('disease_') and word.endswith('_disease'))):
			filtered.append(word)
	return set(filtered)
	
#Multiplying the TF and IDF vectors to create the TFIDF vector
def tfidf_join(tf,idf):

	tfidf = []
	for tf_term in tf:
		tfidf.append((tf_term, tf[tf_term]*idf[tf_term]))
	return dict(tfidf)

#computing term term relevance between two given terms
def term_term_relevance(termA, termB):

	multiplication_list = []
	power_list_A = []
	power_list_B = []
	for elementA in termA:
		power_list_A.append(pow(elementA, 2))
		
	#to compute the numerator, must take sum of the product of all terms in A and B
	multiplication_list = [a*b for a,b in zip(termA, termB)]
	
	for elementB in termB:
		power_list_B.append(pow(elementB, 2))
		
	#to compute the denominator, must take the sum of powers of all terms in A and B, then they are square rooted and multiplied together
	numerator = sum(multiplication_list)
	bottomA = sum(power_list_A)
	bottomB = sum(power_list_B) 

	denominator = sqrt(bottomA) * sqrt(bottomB)

	#divide by 0 error catch
	#all relevancies <= 0 are ignored in output
	if denominator != 0:
		return (numerator / denominator)
	else:
		return 0

#sorts list by 3rd value, descendingly, in this case it will be their term-term relevancy score
def sort_descending(input):
	sorted_values = sorted(input, key=lambda tup: -tup[2])
	return sorted_values

#using a query term to determine term term relevancy with all other terms in the corpus
def term_term_query(tf_idf, query_term):
	f = open('query_result.txt','w')
	#this query list contains the tf-idf vector correlating to the query term
	query_list = []
	for element in tf_idf:
		query_list.append(element[1].get(query_term, 0))

	#dictionary_keys retrieves the term name of every term in the corpus, ex. gene_abc_gene
	dictionary_keys = []

	for element in tf_idf:
		for key in element[1]:
			if key != query_term:
				dictionary_keys.append(key)

	#take distinct values as they may appear multiple times
	dictionary_keys = list(set(dictionary_keys))
	key_list = []
	corpus_list = []
	
	#now, for each term, we need to retrieve their own tf-idf vector
	for key in dictionary_keys:
		score_list = []
		key_list.append(key)
		for element in tf_idf:
			score_list.append(element[1].get(key,0))
		corpus_list.append(score_list)
	#relevance_list keeps track of all relevancies between the query term and other terms in the corpus
	relevance_list = []
	for index, element in enumerate(corpus_list):
			relevance_list.append((query_term, key_list[index], term_term_relevance(list(element),query_list)))
			
	#nonzero_list eliminates all term term relevancy values that are <= 0
	nonzero_list = 	sc.parallelize(relevance_list) \
					.filter(lambda a: a[2] > 0) \
					.collect()

	descending_list = sort_descending(nonzero_list)
	f.write(str(descending_list))
	print("File written successfully.")
	
#filling tfidf with zeroes for proper m x n output	
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
	
#printing tfidf	in a legible format
def tfidf_print(tfidf, idf):
	print_spooler = []
	for row in tfidf:
		if row[1]:
			print_spooler.append((row[0], row[1]))
	return print_spooler
	
#retrieving tf idf vector for a given term	
def fetch_term_vector(tfidf, term):
	term_vector = []
	for row in tfidf:
		term_vector.append(row[1].get(term, 0))
	return term_vector
	
#computing similarity between two terms
def semantic_similarity(vectorA, vectorB):
	numerator = sum(a*b for a,b in zip(vectorA,vectorB))
	denominator = 	sqrt(sum(i ** 2 for i in vectorA)) + \
					sqrt(sum(i ** 2 for i in vectorB))
	if(denominator != 0):
		return numerator / denominator
		
if __name__ == "__main__":
	#initializing text and spark
	filename = "project2_relevance.txt"
	sc = SparkContext(appName="TF-IDF")
	
	#calculating tf matrix
	tf = 	sc.textFile(filename).map(lambda line: line.split(" ")) \
			.map(lambda x: (x[0], tf_word_filter(x[1:])))
	
	#getting total corpus size to use in idf calculation
	corpus_size = tf.count()
	
	#calculating df matrix
	df =	sc.textFile(filename).map(lambda line: line.split(" ")) \
			.flatMap(lambda x: list(df_word_filter(x[1:]))) \
			.map(lambda x: (x, 1)) \
			.reduceByKey(lambda x, y: x + y)
			
	idf =	dict(df.map(lambda x: (x[0], log10(corpus_size/x[1]))).collect())
		
	#merging tf and idf
	tfidf = tf.map(lambda x: (x[0], tfidf_join(x[1], idf)))
	
	# filepath = 'TFIDF_out.txt'
	# tfidf_output = tfidf_print(tfidf.collect(), idf)
	# with open(filepath, 'w') as file:
	# 	for item in tfidf_output:
	# 		file.write("{}\n".format(item))
		
	# save = 	tfidf.coalesce(1).saveAsTextFile('test_out_'+str(time.time()))
	
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
	
	# term_pair_save = term_pairs.saveAsTextFile('term_pairs_'+str(time.time()))
	term_term_query(tfidf.collect(), "gene_nmdars_gene")
