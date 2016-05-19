# -*- coding: UTF-8 -*-
from pyspark import SparkContext
from math import log10
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
	return tfidf
	
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
			
	tfidf = tf.map(lambda x: (x[0], tfidf_join(x[1], idf)))
	
	
	save = 	tfidf.coalesce(1).saveAsTextFile('test_out_'+str(time.time()))
	