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
#idf_vector = copy.deepcopy(tf_vector)

print(idf_vector)
#Remove first element of every row, first elemnent being the doc_index, second being 
#<TO_DO> Change methodology of getting the elements from (x, y)'s y vector
#idf_vector_flat = []
#for row in idf_vector:
	#row.pop(0)
	#for element in row:
		#idf_vector_flat.extend(element)

idf_vector_flattened = 	sc.parallelize(idf_vector) \
						.countByKey() \

word_count = len(idf_vector_flattened)

#Might need to change normalization formula
#idf_vector_normalized = sc.parallelize(idf_vector_flattened) \
						#.map(lambda x: (x[0], math.log(int(x[1]) / int(word_count)))) \
						#.collect()

'''
Multiply each row of the TF vector by IDF vector
HOW_TO: Take each element in TF vector's value section.
		Map each value to its corresponding key:value pair in IDF vector

FORMAT: TF： 	【 [DOC1], 【 (K1, V1), ... , (Kn, Vn) 】 】
								.
								.
			【 [DOCm], 【 (K1, V1), ... , (Kn, Vn) 】 】		
		IDF：	【 (K1, V1), ... , (Kn, Vn) 】
'''

print(idf_vector_flattened)