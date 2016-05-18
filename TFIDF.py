# -*- coding: UTF-8 -*-
from pyspark import SparkContext
import copy
import math

def detect_word(str):
	if (str.startswith(str + '_') and str.endswith('_' + str)):
		return True

def word_filter(str):
	if not(str.startswith('doc')):
		return str

def doc_filter(str):
	if (str.startswith('doc')):
		return str

'''
Compare term-term relevenacy by taking two lists as arguments
Used in both main problem and sub-problem
'''
def term_term_relevance():
	print('hello from the other side')

#<TO_DO> Reduce amount of .collect()'s
filename = "project2_sample.txt"
sc = SparkContext("local", "TF-IDF")

documents = sc.textFile(filename) \
			.map(lambda line: line.split(" ")) \
			.collect()


#Crunches TF-vector
#<TO_DO> Convert for loop to spark handling
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
		
word_count = len(idf_vector_flattened)

#Might need to change normalization formula
idf_vector_normalized = sc.parallelize(idf_vector_flattened) \
						.map(lambda x: (x[0], math.log(int(x[1]) / word_count))) \
						.collect()

idf_dictionary = dict(idf_vector_normalized)

'''
Multiply each row of the TF vector by IDF vector
HOW_TO: Take each element in TF vector's value section.
		Map each value to its corresponding key:value pair in IDF vector

FORMAT: TF：【 [DOC1], 【 (K1, V1), ... , (Kn, Vn) 】 】
								.
								.
			【 [DOCm], 【 (K1, V1), ... , (Kn, Vn) 】 】		
		IDF:【 (K1, V1), ... , (Kn, Vn) 】
'''
test_tf_list = [['doc1'], [('potato', 4), ('apple', 3)]]
test_idf_list = dict([('potato', .5), ('banana', 5), ('apple', 95)])

'''
Use array[n] to access python's n element in lists
Used for pairs as well
Extract each tf vector's value component
Map each of those values using Spark's RDD operations to a function 
Boiled down: 	Get only SECOND value of each row of tf-vector
				Parallelize into RDD, map each of those values from tf to tf-idf
'''

# for row in tf_vector:
# 	for pair in row[1]:
# 		pair[1] *= idf_dictionary['t1']

