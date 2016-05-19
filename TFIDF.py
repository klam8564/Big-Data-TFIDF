# -*- coding: UTF-8 -*-
from pyspark import SparkContext
import copy
from math import pow, log, sqrt

def detect_word(str):
	if (str.startswith(str + '_') and str.endswith('_' + str)):
		return True

def word_filter(str):
	if not(str.startswith('doc')):
		return str

def doc_filter(str):
	if (str.startswith('doc')):
		return str

def term_term_relevance(termA, termB):
	topA = 	sc.parallelize(termA) \
			.zipWithIndex() \
			.map (lambda a: (a[1], a[0]))

	topB = 	sc.parallelize(termB) \
			.zipWithIndex() \
			.map (lambda a: (a[1], a[0]))

	numerator = topA.union(topB) \
			.reduceByKey(lambda a,b: a*b) \
			.map(lambda a: a[1]) \
			.reduce(lambda x, y: x + y)

	bottomA = 	sc.parallelize(termA) \
				.map(lambda a: pow(a,2)) \
				.reduce(lambda x, y: x + y)

	bottomB = 	sc.parallelize(termB) \
				.map(lambda a: pow(a,2)) \
				.reduce(lambda x, y: x + y)

	denominator = sqrt(bottomA) * sqrt(bottomB)

	return (numerator / denominator)
	
def tf_idf_merge(tf_values, idf_vector):
	tf_dict = 	dict(tf_values)
	tf_idf 	= 	sc.parallelize(idf_vector) \
				.map(lambda pair: (pair[0], pair[1] * tf_dict.get(pair[0], 0))) \
				.collect()
	return tf_idf

def sort_descending(input):
	sorted_values = sc.parallelize(input) \
					.map(lambda a: (a[1], a[0])) \
					.sortByKey(False) \
					.map(lambda a: (a[1], a[0])) \
					.collect()
	return sorted_values

filename = "project2_sample.txt"
sc = SparkContext("local", "TF-IDF")

documents = sc.textFile(filename) \
			.map(lambda line: line.split(" ")) \
			.collect()
			
#Crunches TF-vector
tf_vector = []
idf_vector = []
for document in documents:
	tf_vector_row = []
	doc_value = sc.parallelize(document) \
				.map(lambda word: word_filter(word)) \
				.filter(lambda x: x!=None) \
				.map(lambda word: (word, 1)) \
				.reduceByKey(lambda a, b: a + b) \
				.collect()

#Extracts the document index
#<TO_DO> Better way to do it in spark
	doc_index = sc.parallelize(document) \
				.map(lambda word: doc_filter(word)) \
				.filter(lambda x: x!=None) \
				.collect()

	tf_vector_row.append(doc_index)
	tf_vector_row.append(doc_value)
	idf_vector.extend(doc_value)
	tf_vector.append(tf_vector_row)

#Crunches IDF-vector
idf_vector_flattened_dict = sc.parallelize(idf_vector) \
							.countByKey()	
								
idf_vector_flattened = [(k,v) for k,v in idf_vector_flattened_dict.items()]
		
document_count = len(tf_vector)

idf_vector_normalized = sc.parallelize(idf_vector_flattened) \
						.map(lambda x: (x[0], log(int(document_count) / int(x[1]) ))) \
						.collect()
						
tf_idf = []
for row in tf_vector:
	tf_idf.append([row[0], tf_idf_merge(row[1], idf_vector_normalized)])



