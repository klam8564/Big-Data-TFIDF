# -*- coding: UTF-8 -*-
from pyspark import SparkContext
from math import log10, sqrt
import time
import itertools
def doc_filter(str):
	if (str.startswith('doc')) and str[3:].isdigit():
		return str
		
def tf_word_filter(list):
	filtered = {}
	for word in list:
		if ((word.startswith('gene_') and word.endswith('_gene')) \
			or (word.startswith('disease_') and word.endswith('_disease'))):
			filtered[word] = filtered.get(word, 0) + 1
	return filtered
	
def df_word_filter(list):
	filtered = []
	for word in list:
		if ((word.startswith('gene_') and word.endswith('_gene')) \
			or (word.startswith('disease_') and word.endswith('_disease'))):
			filtered.append(word)
	return set(filtered)

def tfidf_join(tf,idf):

	tfidf = []
	for tf_term in tf:
		tfidf.append((tf_term, tf[tf_term]*idf[tf_term]))
	return dict(tfidf)
	
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
	
def tfidf_print(tfidf, idf):
	print_spooler = []
	for row in tfidf:
		if row[1]:
			print_spooler.append((row[0], row[1]))
	return print_spooler
	
def term_term_relevancy():
	return 1
	
def fetch_term_vector(tfidf, term):
	term_vector = []
	for row in tfidf:
		term_vector.append(row[1].get(term, 0))
	return term_vector
	
def semantic_similarity(vectorA, vectorB):
	numerator = sum(a*b for a,b in zip(vectorA,vectorB))
	denominator = 	sqrt(sum(i ** 2 for i in vectorA)) + \
					sqrt(sum(i ** 2 for i in vectorB))
	if(denominator != 0):
		return numerator / denominator
if __name__ == "__main__":
	filename = "project2_data.txt"
	sc = SparkContext(appName="TF-IDF")
	
	tf = 	sc.textFile(filename).map(lambda line: line.split(" ")) \
			.map(lambda x: (x[0], tf_word_filter(x[1:])))
				
	corpus_size = tf.count()
	
	df =	sc.textFile(filename).map(lambda line: line.split(" ")) \
			.flatMap(lambda x: list(df_word_filter(x[1:]))) \
			.map(lambda x: (x, 1)) \
			.reduceByKey(lambda x, y: x + y)
			
	idf =	dict(df.map(lambda x: (x[0], log10(corpus_size/x[1]))).collect())

	# idffilepath = 'IDF_out_txt'
	# with open(idffilepath, 'w') as file:
	# 	file.write("{}\n".format(idf))
		
	tfidf = tf.map(lambda x: (x[0], tfidf_join(x[1], idf)))
	
	# filepath = 'TFIDF_out.txt'
	# tfidf_output = tfidf_print(tfidf.collect(), idf)
	# with open(filepath, 'w') as file:
	# 	for item in tfidf_output:
	# 		file.write("{}\n".format(item))
		
	# save = 	tfidf.coalesce(1).saveAsTextFile('test_out_'+str(time.time()))
	
	term_term = sc.parallelize(idf)
	
	# term_term_save = term_term.coalesce(1).saveAsTextFile('term_term_'+str(time.time()))
	
	lookup = tfidf.collect()
	term_vectors = term_term.map(lambda x: (x, fetch_term_vector(lookup, x)))
	combinations =  []
	for pair in itertools.combinations(term_vectors.collect(), 2):
		combinations.append(pair)
	
	# for row in combinations:
	# 	print(row)	
	term_pairs = 	sc.parallelize(combinations) \
					.map(lambda x: ((x[0][0],x[1][0]), semantic_similarity(x[0][1], x[1][1]))) \
					.filter(lambda x: x[1] != 0) \
					.sortBy(lambda x: -x[1])
	
	term_pair_save = term_pairs.saveAsTextFile('term_pairs_'+str(time.time()))