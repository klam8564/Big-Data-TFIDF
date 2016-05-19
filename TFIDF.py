# -*- coding: UTF-8 -*-
from pyspark import SparkContext
from math import log10, pow, sqrt
import time
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


def term_term_relevance(termA, termB):

	multiplication_list = []
	power_list_A = []
	power_list_B = []
	for elementA in termA:
		power_list_A.append(pow(elementA, 2))
		
	multiplication_list = [a*b for a,b in zip(termA, termB)]
	
	for elementB in termB:
		power_list_B.append(pow(elementB, 2))

	numerator = sum(multiplication_list)

	bottomA = sum(power_list_A)

	bottomB = sum(power_list_B) 

	denominator = sqrt(bottomA) * sqrt(bottomB)

	#divide by 0 error catch
	if denominator != 0:
		return (numerator / denominator)
	else:
		return 0

def sort_descending(input):
	sorted_values = sorted(input, key=lambda tup: -tup[2])
	return sorted_values


def term_term_query(tf_idf, query_term):
	f = open('query_result.txt','w')
	query_list = []
	for element in tf_idf:
		query_list.append(element[1].get(query_term, 0))

	dictionary_keys = []

	for element in tf_idf:
		for key in element[1]:
			if key != query_term:
				dictionary_keys.append(key)

	#takes distinct values
	dictionary_keys = list(set(dictionary_keys))
	key_list = []
	corpus_list = []

	for key in dictionary_keys:
		score_list = []
		key_list.append(key)
		for element in tf_idf:
			score_list.append(element[1].get(key,0))
		corpus_list.append(score_list)

	relevance_list = []
	for index, element in enumerate(corpus_list):
			relevance_list.append((query_term, key_list[index], term_term_relevance(list(element),query_list)))

	nonzero_list = 	sc.parallelize(relevance_list) \
					.filter(lambda a: a[2] > 0) \
					.collect()

	descending_list = sort_descending(nonzero_list)
	f.write(str(descending_list))
	print("File written successfully.")



	
if __name__ == "__main__":
	filename = "project2_relevance.txt"
	sc = SparkContext(appName="TF-IDF")
	
	tf = 	sc.textFile(filename).map(lambda line: line.split(" ")) \
			.map(lambda x: (x[0], tf_word_filter(x[1:])))
				
	corpus_size = tf.count()
	
	df =	sc.textFile(filename).map(lambda line: line.split(" ")) \
			.flatMap(lambda x: list(df_word_filter(x[1:]))) \
			.map(lambda x: (x, 1)) \
			.reduceByKey(lambda x, y: x + y)
			
	idf =	dict(df.map(lambda x: (x[0], log10(corpus_size/x[1]))).collect())
			
	tfidf = tf.map(lambda x: (x[0], tfidf_join(x[1], idf)))
	
	
	term_term_query(tfidf.collect(), "gene_nmdars_gene")